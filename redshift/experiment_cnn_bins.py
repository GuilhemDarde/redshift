import argparse
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from analysis_utils import compute_regression_metrics, ensure_dir, write_rows_csv, z_bin_centers, z_to_bin_indices
from config import CONFIG
from data_loader import get_dataloaders
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BinCNN(nn.Module):
    def __init__(self, n_bins: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def labels_from_cond(cond: torch.Tensor, edges: np.ndarray, device: torch.device) -> torch.Tensor:
    labels = z_to_bin_indices(cond[:, 0].detach().cpu().numpy(), edges)
    return torch.tensor(labels, dtype=torch.long, device=device)


def run_epoch(model, loader, optimizer, criterion, device, edges, limit_batches: Optional[int]) -> float:
    model.train()
    losses = []
    for batch_idx, (x, cond) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = labels_from_cond(cond, edges, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def predict(model, loader, device, edges, limit_batches: Optional[int]):
    model.eval()
    centers = z_bin_centers(edges)
    z_true, z_pred = [], []
    with torch.no_grad():
        for batch_idx, (x, cond) in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            logits = model(x.to(device, non_blocking=True))
            pred_bins = torch.argmax(logits, dim=1).cpu().numpy()
            z_pred.append(centers[pred_bins])
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
    model = BinCNN(n_bins=len(edges) - 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss = run_epoch(model, train_loader, optimizer, criterion, device, edges, args.limit_batches)
        logger.info("Epoch %s/%s | CE %.5f", epoch + 1, args.epochs, loss)

    suffix = f"_fold{args.fold_id}" if args.fold_id is not None else ""
    ckpt = os.path.join(output_dir, f"cnn_bins{suffix}.pt")
    torch.save(model.state_dict(), ckpt)
    z_true, z_pred = predict(model, test_loader, device, edges, args.limit_batches)
    metrics = compute_regression_metrics(z_true, z_pred)
    metrics.update({"fold_id": args.fold_id if args.fold_id is not None else -1, "region": args.region})
    np.savez(os.path.join(output_dir, f"predictions_cnn_bins{suffix}.npz"), z_true=z_true, z_pred=z_pred)
    write_rows_csv(os.path.join(output_dir, f"metrics_cnn_bins{suffix}.csv"), [metrics])


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
