import argparse
import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from analysis_utils import compute_regression_metrics, ensure_dir, write_rows_csv, z_to_bin_indices
from analyze_treyer_figure7 import make_figure7_report
from config import CONFIG
from data_loader import build_metadata, get_dataset_and_splits
from experiment_marie_baseline import MarieStyleBaseline, meta_from_cond
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MarieWideBaseline(nn.Module):
    '''
    actions : Baseline plus proche de l'esprit Marie/Treyer: branche image plus large + MLP magnitudes.
    inputs : n_bins (int), meta_dim (int)
    appels : Conv2d, BatchNorm2d, AdaptiveAvgPool2d
    outputs : logits de bins redshift et regression z
    '''
    def __init__(self, n_bins: int, meta_dim: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(192 * 4 * 4 + 128, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(256, n_bins)
        self.reg_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([self.image(x), self.meta(meta)], dim=1)
        h = self.shared(h)
        return self.class_head(h), self.reg_head(h).squeeze(1)


def make_model(architecture: str, n_bins: int, dropout: float) -> nn.Module:
    if architecture == "compact":
        return MarieStyleBaseline(n_bins=n_bins)
    if architecture == "wide":
        return MarieWideBaseline(n_bins=n_bins, dropout=dropout)
    raise ValueError(f"Architecture inconnue: {architecture}")


def maybe_data_parallel(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Activation DataParallel Marie/Treyer sur %s GPU visibles.", torch.cuda.device_count())
        return nn.DataParallel(model)
    if enabled:
        logger.info("DataParallel demandé mais un seul GPU est visible.")
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def loss_batch(model: nn.Module, x: torch.Tensor, cond: torch.Tensor, edges: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor(z_to_bin_indices(cond[:, 0].detach().cpu().numpy(), edges), dtype=torch.long, device=cond.device)
    logits, z_reg = model(x, meta_from_cond(cond))
    ce = nn.functional.cross_entropy(logits, labels)
    huber = nn.functional.smooth_l1_loss(z_reg, cond[:, 0])
    return ce + huber, logits, z_reg


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device, edges: np.ndarray, limit_batches: Optional[int]) -> float:
    model.train()
    losses = []
    for batch_idx, (x, cond) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = loss_batch(model, x, cond, edges)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, edges: np.ndarray, limit_batches: Optional[int]) -> float:
    model.eval()
    losses = []
    for batch_idx, (x, cond) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        loss, _, _ = loss_batch(model, x, cond, edges)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, limit_batches: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_true, z_pred = [], []
    for batch_idx, (x, cond) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        cond_device = cond.to(device, non_blocking=True)
        _, z_reg = model(x, meta_from_cond(cond_device))
        z_true.append(cond[:, 0].numpy())
        z_pred.append(z_reg.cpu().numpy())
    if not z_true:
        raise RuntimeError("Aucun batch évalué.")
    return np.concatenate(z_true), np.concatenate(z_pred)


def split_summary_rows(metadata: Dict[str, np.ndarray]) -> list:
    rows = []
    for split in ["train", "val", "test"]:
        mask = metadata["split"] == split
        rows.append({
            "split": split,
            "n": int(np.sum(mask)),
            "z_min": float(np.nanmin(metadata["z_true"][mask])) if np.any(mask) else float("nan"),
            "z_max": float(np.nanmax(metadata["z_true"][mask])) if np.any(mask) else float("nan"),
            "mag_i_min": float(np.nanmin(metadata["mag_i"][mask])) if np.any(mask) else float("nan"),
            "mag_i_max": float(np.nanmax(metadata["mag_i"][mask])) if np.any(mask) else float("nan"),
            "field_values": ";".join(sorted(set(np.asarray(metadata["field"])[mask].astype(str).tolist()))) if np.any(mask) else "",
            "label_values": ";".join(sorted(set(np.asarray(metadata["label_type"])[mask].astype(str).tolist()))) if np.any(mask) else "",
        })
    return rows


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir or CONFIG.exp_path("marie_treyer_baseline"))
    device = torch.device(CONFIG.DEVICE)
    edges = np.array(CONFIG.Z_BIN_EDGES, dtype=np.float64)

    dataset, split_indices = get_dataset_and_splits(
        region=args.region,
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
    )
    metadata = build_metadata(dataset, split_indices=split_indices)
    metadata_path = os.path.join(output_dir, "dataset_metadata_treyer.npz")
    np.savez(metadata_path, **metadata)
    write_rows_csv(os.path.join(output_dir, "split_summary.csv"), split_summary_rows(metadata))

    train_ds = Subset(dataset, split_indices["train"])
    val_ds = Subset(dataset, split_indices["val"])
    test_ds = Subset(dataset, split_indices["test"])
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = make_model(args.architecture, n_bins=len(edges) - 1, dropout=args.dropout).to(device)
    model = maybe_data_parallel(model, args.data_parallel)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience
    history = []
    ckpt_path = os.path.join(output_dir, "marie_treyer_baseline_best.pt")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, edges, args.limit_batches)
        val_loss = eval_loss(model, val_loader, device, edges, args.limit_batches)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %s/%s | train %.5f | val %.5f", epoch + 1, args.epochs, train_loss, val_loss)
        if np.isfinite(val_loss) and val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch + 1
            patience_left = args.patience
            torch.save(unwrap_model(model).state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                logger.info("Early stopping epoch %s. Best epoch=%s val=%.5f", epoch + 1, best_epoch, best_val)
                break

    write_rows_csv(os.path.join(output_dir, "training_history.csv"), history)
    if best_epoch < 0:
        best_epoch = len(history)
        best_val = history[-1]["val_loss"] if history else float("nan")
        torch.save(unwrap_model(model).state_dict(), ckpt_path)
    unwrap_model(model).load_state_dict(torch.load(ckpt_path, map_location=device))
    z_true, z_pred = predict(model, test_loader, device, args.limit_batches)
    metrics = compute_regression_metrics(z_true, z_pred)
    metrics.update({
        "seed": args.seed,
        "fold_id": args.fold_id if args.fold_id is not None else -1,
        "region": args.region,
        "field": args.field,
        "sample_filter": args.sample_filter,
        "architecture": args.architecture,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    })
    write_rows_csv(os.path.join(output_dir, "metrics_marie_treyer_baseline.csv"), [metrics])

    predictions_path = os.path.join(output_dir, "predictions_marie_treyer_baseline.npz")
    np.savez(
        predictions_path,
        z_true=z_true,
        z_pred=z_pred,
        test_indices=split_indices["test"][: len(z_true)],
        train_mag_i=metadata["mag_i"][split_indices["train"]],
    )
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    make_figure7_report(
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        output_dir=os.path.join(output_dir, "figure7_like"),
        z_bins=args.z_bins,
        mag_i_min=args.mag_i_min,
        mag_i_max=args.mag_i_max,
        mag_i_bins=args.mag_i_bins,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    logger.info("Baseline Marie/Treyer terminée: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["compact", "wide"], default="wide")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=CONFIG.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CONFIG.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="cosmos_ud")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--z_bins", type=int, default=20)
    parser.add_argument("--mag_i_min", type=float, default=CONFIG.I_MIN)
    parser.add_argument("--mag_i_max", type=float, default=CONFIG.I_MAX)
    parser.add_argument("--mag_i_bins", type=int, default=14)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--data_parallel", action="store_true")
    run(parser.parse_args())
