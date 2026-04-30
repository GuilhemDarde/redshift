import argparse
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from analysis_utils import compute_regression_metrics, ensure_dir, write_rows_csv, z_to_bin_indices
from config import CONFIG
from data_loader import get_dataloaders
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MarieStyleBaseline(nn.Module):
    """Wrapper local inspire du modele Marie: branche image + branche magnitudes + tetes classification/regression."""

    def __init__(self, n_bins: int, meta_dim: int = 4) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(6, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.meta = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU(), nn.Linear(64, 96), nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(192, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU())
        self.class_head = nn.Linear(128, n_bins)
        self.reg_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([self.image(x), self.meta(meta)], dim=1)
        h = self.shared(h)
        return self.class_head(h), self.reg_head(h).squeeze(1)


def meta_from_cond(cond: torch.Tensor) -> torch.Tensor:
    return cond[:, 1:5]


def train_epoch(model, loader, optimizer, device, edges, limit_batches: Optional[int]) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    losses = []
    for batch_idx, (x, cond) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        labels = torch.tensor(z_to_bin_indices(cond[:, 0].detach().cpu().numpy(), edges), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, z_reg = model(x, meta_from_cond(cond))
        loss = ce(logits, labels) + huber(z_reg, cond[:, 0])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def predict(model, loader, device, limit_batches: Optional[int]):
    model.eval()
    z_true, z_pred = [], []
    with torch.no_grad():
        for batch_idx, (x, cond) in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            x = x.to(device, non_blocking=True)
            cond_device = cond.to(device, non_blocking=True)
            _, z_reg = model(x, meta_from_cond(cond_device))
            z_pred.append(z_reg.cpu().numpy())
            z_true.append(cond[:, 0].numpy())
    if not z_true:
        raise RuntimeError("Aucun batch evalue. Verifiez le dataset, le split test ou --limit_batches.")
    return np.concatenate(z_true), np.concatenate(z_pred)


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir or CONFIG.EXP_FOLDER)
    device = torch.device(CONFIG.DEVICE)
    edges = np.array(CONFIG.Z_BIN_EDGES, dtype=np.float64)
    train_loader, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        region=args.region,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
    )
    model = MarieStyleBaseline(n_bins=len(edges) - 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, edges, args.limit_batches)
        logger.info("Epoch %s/%s | Loss %.5f", epoch + 1, args.epochs, loss)

    suffix = f"_fold{args.fold_id}" if args.fold_id is not None else ""
    torch.save(model.state_dict(), os.path.join(output_dir, f"marie_baseline{suffix}.pt"))
    z_true, z_pred = predict(model, test_loader, device, args.limit_batches)
    metrics = compute_regression_metrics(z_true, z_pred)
    metrics.update({"fold_id": args.fold_id if args.fold_id is not None else -1, "region": args.region})
    np.savez(os.path.join(output_dir, f"predictions_marie_baseline{suffix}.npz"), z_true=z_true, z_pred=z_pred)
    write_rows_csv(os.path.join(output_dir, f"metrics_marie_baseline{suffix}.csv"), [metrics])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=CONFIG.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CONFIG.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    run(parser.parse_args())
