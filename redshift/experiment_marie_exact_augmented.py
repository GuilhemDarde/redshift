import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from analysis_utils import (
    compute_regression_metrics,
    ensure_dir,
    magnitude_bin_edges,
    magnitude_support_definition_rows,
    magnitude_support_mask,
    save_metadata_npz,
    split_labels,
    write_rows_csv,
)
from analyze_treyer_figure7 import make_figure7_report
from config import CONFIG
from data_loader import build_metadata, get_dataset_and_splits
from marie_treyer_exact import (
    build_marie_treyer_model,
    marie_point_estimate,
    marie_z_centers,
    marie_z_edges,
    z_to_marie_bins,
)
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def maybe_data_parallel(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Activation DataParallel modèle Marie exact sur %s GPU visibles.", torch.cuda.device_count())
        return nn.DataParallel(model)
    if enabled:
        logger.info("DataParallel demandé mais un seul GPU est visible.")
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _random_marie_aug(x: np.ndarray) -> np.ndarray:
    alea = np.random.randint(0, 8)
    if alea == 1:
        x = np.rot90(x, axes=(1, 2), k=1)
    elif alea == 2:
        x = np.rot90(x, axes=(1, 2), k=2)
    elif alea == 3:
        x = np.rot90(x, axes=(1, 2), k=3)
    elif alea == 4:
        x = np.flip(x, axis=1)
    elif alea == 5:
        x = np.flip(x, axis=2)
    elif alea == 6:
        x = np.rot90(np.flip(x, axis=1).copy(), axes=(1, 2), k=3)
    elif alea == 7:
        x = np.rot90(np.flip(x, axis=2).copy(), axes=(1, 2), k=3)
    return x.copy()


def _mags_for_marie(data: Dict[str, np.ndarray]) -> np.ndarray:
    if "mags_marie" in data:
        mags = np.asarray(data["mags_marie"], dtype=np.float32)
    else:
        mags = np.asarray(data["mags"], dtype=np.float32)
    return np.nan_to_num(mags, nan=0.0, posinf=0.0, neginf=0.0)


class MarieArrayDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        z_true: np.ndarray,
        mag_i: np.ndarray,
        ebv: np.ndarray,
        mags: np.ndarray,
        edges: np.ndarray,
        indices: Optional[np.ndarray] = None,
        augment: bool = False,
        is_synthetic: bool = False,
    ) -> None:
        self.x = x
        self.z_true = np.asarray(z_true, dtype=np.float32)
        self.mag_i = np.asarray(mag_i, dtype=np.float32)
        self.ebv = np.asarray(ebv, dtype=np.float32)
        self.mags = np.asarray(mags, dtype=np.float32)
        self.edges = np.asarray(edges, dtype=np.float64)
        self.indices = np.arange(len(self.z_true), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.is_synthetic = is_synthetic

    def set_indices(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = np.asarray(self.x[real_idx], dtype=np.float32)
        if self.augment:
            x = _random_marie_aug(x)
        z = np.float32(self.z_true[real_idx])
        zbin = np.int64(z_to_marie_bins(np.asarray([z]), self.edges)[0])
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(zbin, dtype=torch.long),
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(self.ebv[real_idx], dtype=torch.float32),
            torch.tensor(self.mags[real_idx], dtype=torch.float32),
            torch.tensor(self.mag_i[real_idx], dtype=torch.float32),
            torch.tensor(real_idx, dtype=torch.long),
            torch.tensor(1 if self.is_synthetic else 0, dtype=torch.long),
        )


def _synthetic_mags_from_cond(cond: np.ndarray) -> np.ndarray:
    mag_i = cond[:, 1] * 2.0 + 22.0
    r = mag_i + cond[:, 3]
    g = r + cond[:, 2]
    z = mag_i - cond[:, 4]
    u = g
    y = z
    return np.stack([u, g, r, mag_i, z, y], axis=1).astype(np.float32)


def load_synthetic_marie_dataset(
    path: str,
    real_data: Dict[str, np.ndarray],
    edges: np.ndarray,
    train_indices: np.ndarray,
    max_samples: Optional[int],
    mode_filter: Optional[str],
    seed: int,
    augment: bool,
) -> MarieArrayDataset:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset synthétique introuvable: {path}")
    data = np.load(path, allow_pickle=False)
    if "x" not in data or "cond" not in data:
        raise KeyError("Le dataset synthétique doit contenir x et cond.")
    mask = np.ones(len(data["x"]), dtype=bool)
    if mode_filter is not None and "mode" in data.files:
        mask &= data["mode"].astype(str) == mode_filter
    if "source_index" in data.files:
        train_lookup = np.zeros(len(real_data["z_true"]), dtype=bool)
        train_lookup[np.asarray(train_indices, dtype=np.int64)] = True
        valid_source = (data["source_index"] >= 0) & (data["source_index"] < len(train_lookup))
        mask &= valid_source & train_lookup[data["source_index"].clip(0, len(train_lookup) - 1)]
    selected = np.where(mask)[0]
    if selected.size == 0:
        raise ValueError("Aucune augmentation synthétique compatible avec le fold courant.")
    if max_samples is not None and selected.size > max_samples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(selected, size=max_samples, replace=False))

    x = data["x"][selected].astype(np.float32)
    cond = data["cond"][selected].astype(np.float32)
    z_true = cond[:, 0]
    mag_i = cond[:, 1] * 2.0 + 22.0
    ebv = np.zeros(len(selected), dtype=np.float32)
    if "source_index" in data.files:
        source_idx = data["source_index"][selected].astype(np.int64)
        mags = _mags_for_marie(real_data)[source_idx]
        if "ebv" in real_data:
            ebv = np.asarray(real_data["ebv"], dtype=np.float32)[source_idx]
    else:
        mags = _synthetic_mags_from_cond(cond)
    return MarieArrayDataset(
        x=x,
        z_true=z_true,
        mag_i=mag_i,
        ebv=ebv,
        mags=mags,
        edges=edges,
        augment=augment,
        is_synthetic=True,
    )


def make_real_dataset(data: Dict[str, np.ndarray], indices: np.ndarray, edges: np.ndarray, augment: bool) -> MarieArrayDataset:
    ebv = np.asarray(data.get("ebv", np.zeros(len(data["z_true"]), dtype=np.float32)), dtype=np.float32)
    return MarieArrayDataset(
        x=np.asarray(data["x"], dtype=np.float32),
        z_true=np.asarray(data["z_true"], dtype=np.float32),
        mag_i=np.asarray(data["mag_i"], dtype=np.float32),
        ebv=ebv,
        mags=_mags_for_marie(data),
        edges=edges,
        indices=indices,
        augment=augment,
        is_synthetic=False,
    )


def smooth_indices_like_marie(train_indices: np.ndarray, z_true: np.ndarray, mag_i: np.ndarray, edges: np.ndarray, seed: int) -> np.ndarray:
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(train_indices).copy()
        rng.shuffle(shuffled)
        return shuffled

    rng = np.random.default_rng(seed)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    train_mag = mag_i[train_indices]
    faint = np.sort(train_mag[train_mag >= 22.0])
    if faint.size < 10:
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        return shuffled

    divide_by = 6
    cut_pos = np.asarray([int(faint.size / divide_by * k) for k in range(divide_by + 1)])
    cut_pos[cut_pos >= faint.size] = faint.size - 1
    i_values = np.hstack((np.asarray([18.0, 20.0, 21.0, 22.0]), faint[cut_pos][1:]))
    selected_parts: List[np.ndarray] = []
    bin_mid = 0.5 * (edges[:-1] + edges[1:])

    for lo, hi in zip(i_values[:-1], i_values[1:]):
        selected = train_indices[(train_mag >= lo) & (train_mag <= hi)]
        if selected.size < 2:
            if selected.size:
                selected_parts.append(selected)
            continue
        z_sel = z_true[selected]
        hist, _ = np.histogram(z_sel, bins=edges)
        if np.count_nonzero(hist) < 2:
            selected_parts.append(selected)
            continue
        try:
            kde_values = gaussian_kde(z_sel, bw_method=0.5)(bin_mid)
        except Exception:
            selected_parts.append(selected)
            continue
        candidates = []
        hist_max = max(int(hist.max()), 1)
        for bin_id in range(len(edges) - 1):
            in_bin = selected[(z_true[selected] >= edges[bin_id]) & (z_true[selected] < edges[bin_id + 1])]
            if in_bin.size == 0:
                continue
            new_size = max(1, int(kde_values[bin_id] * hist_max))
            candidates.append(rng.choice(in_bin, size=new_size, replace=True))
        if not candidates:
            selected_parts.append(selected)
            continue
        pool = np.concatenate(candidates)
        selected_parts.append(rng.choice(pool, size=selected.size, replace=pool.size < selected.size))

    if not selected_parts:
        return train_indices
    smoothed = np.concatenate(selected_parts).astype(np.int64)
    if smoothed.size != train_indices.size:
        smoothed = rng.choice(smoothed, size=train_indices.size, replace=smoothed.size < train_indices.size)
    rng.shuffle(smoothed)
    return smoothed


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    bin_centers: torch.Tensor,
    limit_batches: Optional[int],
) -> float:
    model.train()
    losses = []
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x, zbin, z_true, ebv, mags = [item.to(device, non_blocking=True) for item in batch[:5]]
        optimizer.zero_grad(set_to_none=True)
        logits, z_reg = model(x, ebv, mags=mags)
        loss = ce(logits, zbin) + l1(z_reg.flatten(), z_true)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bin_centers: torch.Tensor,
    limit_batches: Optional[int],
) -> Dict[str, np.ndarray]:
    model.eval()
    z_true_parts, z_pred_parts, mag_parts, index_parts, synth_parts = [], [], [], [], []
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x, _, z_true, ebv, mags, mag_i, index, is_synth = batch
        x = x.to(device, non_blocking=True)
        ebv = ebv.to(device, non_blocking=True)
        mags = mags.to(device, non_blocking=True)
        logits, _ = model(x, ebv, mags=mags)
        z_pred = marie_point_estimate(logits, bin_centers)
        z_true_parts.append(z_true.numpy())
        z_pred_parts.append(z_pred.cpu().numpy())
        mag_parts.append(mag_i.numpy())
        index_parts.append(index.numpy())
        synth_parts.append(is_synth.numpy())
    if not z_true_parts:
        raise RuntimeError("Aucun batch évalué.")
    return {
        "z_true": np.concatenate(z_true_parts),
        "z_pred": np.concatenate(z_pred_parts),
        "mag_i": np.concatenate(mag_parts),
        "index": np.concatenate(index_parts),
        "is_synthetic": np.concatenate(synth_parts),
    }


def build_train_dataset(
    ablation: str,
    real_train: MarieArrayDataset,
    dataset_data: Dict[str, np.ndarray],
    edges: np.ndarray,
    train_indices: np.ndarray,
    args: argparse.Namespace,
) -> Dataset:
    if ablation == "real":
        return real_train
    if ablation in {"i2i", "classic_i2i"}:
        synthetic = load_synthetic_marie_dataset(
            path=args.synthetic_i2i,
            real_data=dataset_data,
            edges=edges,
            train_indices=train_indices,
            max_samples=args.max_synthetic,
            mode_filter="i2i" if args.filter_synthetic_mode else None,
            seed=args.seed,
            augment=args.augment_synthetic,
        )
        return ConcatDataset([real_train, synthetic])
    raise ValueError(f"Ablation inconnue: {ablation}")


def eval_subsets(
    ablation: str,
    pred: Dict[str, np.ndarray],
    train_mag_i: np.ndarray,
    args: argparse.Namespace,
    output_dir: str,
) -> Dict[str, float]:
    edges = magnitude_bin_edges(args.mag_i_min, args.mag_i_max, args.mag_i_bins)
    low, threshold, support, mag_bin, counts = magnitude_support_mask(pred["mag_i"], train_mag_i, edges, quantile=args.low_mag_support_quantile)
    rows = []
    for name, mask in [
        ("global", np.ones(len(pred["z_true"]), dtype=bool)),
        ("low_mag_support", low),
        ("normal_mag_support", np.isfinite(support) & ~low),
        ("faint_mag", pred["mag_i"] >= args.faint_mag_threshold),
        ("normal_faint_mag", pred["mag_i"] < args.faint_mag_threshold),
    ]:
        row = compute_regression_metrics(pred["z_true"][mask], pred["z_pred"][mask])
        row.update({"ablation": ablation, "subset": name})
        rows.append(row)
    write_rows_csv(os.path.join(output_dir, f"metrics_subsets_{ablation}.csv"), rows)
    write_rows_csv(
        os.path.join(output_dir, "mag_support_definition.csv"),
        magnitude_support_definition_rows(edges, counts, threshold),
    )
    return rows[0]


def run_single_ablation(
    ablation: str,
    real_train: MarieArrayDataset,
    eval_dataset: MarieArrayDataset,
    dataset_data: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    metadata: Dict[str, np.ndarray],
    args: argparse.Namespace,
    output_dir: str,
) -> Dict[str, float]:
    device = torch.device(CONFIG.DEVICE)
    edges = marie_z_edges(args.n_bins, args.z_min, args.z_max)
    bin_centers = torch.tensor(marie_z_centers(edges), dtype=torch.float32, device=device)
    real_train.set_indices(train_indices)

    train_dataset = build_train_dataset(ablation, real_train, dataset_data, edges, train_indices, args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False)

    model = build_marie_treyer_model(n_bins=args.n_bins, mags_input_size=6).to(device)
    model = maybe_data_parallel(model, args.data_parallel)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    history = []
    for epoch in range(args.epochs):
        if args.smooth_distribution:
            smoothed = smooth_indices_like_marie(train_indices, dataset_data["z_true"], dataset_data["mag_i"], edges, seed=args.seed + epoch)
            real_train.set_indices(smoothed)
        loss = train_epoch(model, train_loader, optimizer, device, bin_centers, args.limit_batches)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": loss, "lr": scheduler.get_last_lr()[0], "ablation": ablation})
        logger.info("[%s] Epoch %s/%s | loss %.5f | lr %.3g", ablation, epoch + 1, args.epochs, loss, scheduler.get_last_lr()[0])

    write_rows_csv(os.path.join(output_dir, f"training_history_{ablation}.csv"), history)
    torch.save(unwrap_model(model).state_dict(), os.path.join(output_dir, f"marie_exact_{ablation}.pt"))

    pred = predict(model, eval_loader, device, bin_centers, args.limit_batches)
    pred_path = os.path.join(output_dir, f"predictions_marie_exact_{ablation}.npz")
    np.savez(
        pred_path,
        z_true=pred["z_true"],
        z_pred=pred["z_pred"],
        mag_i=pred["mag_i"],
        test_indices=eval_indices[: len(pred["z_true"])],
        train_mag_i=dataset_data["mag_i"][train_indices],
    )

    metrics = eval_subsets(ablation, pred, dataset_data["mag_i"][train_indices], args, output_dir)
    metrics.update({"ablation": ablation, "fold_id": args.fold_id, "epochs": args.epochs})

    metadata_path = os.path.join(output_dir, "dataset_metadata_marie_exact.npz")
    make_figure7_report(
        predictions_path=pred_path,
        metadata_path=metadata_path,
        output_dir=os.path.join(output_dir, f"figure7_{ablation}"),
        z_bins=args.z_bins,
        mag_i_min=args.mag_i_min,
        mag_i_max=args.mag_i_max,
        mag_i_bins=args.mag_i_bins,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    return metrics


def resolve_ablations(values: Sequence[str]) -> List[str]:
    if "all" in values:
        values = ["real", "i2i"]
    valid = {"real", "i2i", "classic_i2i"}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Ablations inconnues: {unknown}")
    return list(values)


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir or CONFIG.exp_path(f"marie_exact_augmented_fold{args.fold_id}"))
    edges = marie_z_edges(args.n_bins, args.z_min, args.z_max)
    dataset, split_indices = get_dataset_and_splits(
        region=args.region,
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
        split_strategy="marie_regular",
    )
    metadata = build_metadata(dataset, split_indices=split_indices)
    metadata["split"] = split_labels(len(dataset), split_indices)
    metadata_path = os.path.join(output_dir, "dataset_metadata_marie_exact.npz")
    save_metadata_npz(metadata_path, metadata)

    train_indices = split_indices["train"]
    eval_indices = split_indices["test"]
    real_train = make_real_dataset(dataset.data, train_indices, edges, augment=True)
    eval_dataset = make_real_dataset(dataset.data, eval_indices, edges, augment=False)

    split_rows = []
    for split_name, idx in split_indices.items():
        split_rows.append({
            "split": split_name,
            "n": len(idx),
            "z_min": float(np.nanmin(dataset.data["z_true"][idx])) if len(idx) else float("nan"),
            "z_max": float(np.nanmax(dataset.data["z_true"][idx])) if len(idx) else float("nan"),
            "mag_i_min": float(np.nanmin(dataset.data["mag_i"][idx])) if len(idx) else float("nan"),
            "mag_i_max": float(np.nanmax(dataset.data["mag_i"][idx])) if len(idx) else float("nan"),
        })
    write_rows_csv(os.path.join(output_dir, "split_summary.csv"), split_rows)

    summary_rows = []
    real_ref = None
    for ablation in resolve_ablations(args.ablations):
        metrics = run_single_ablation(
            ablation=ablation,
            real_train=real_train,
            eval_dataset=eval_dataset,
            dataset_data=dataset.data,
            train_indices=train_indices,
            eval_indices=eval_indices,
            metadata=metadata,
            args=args,
            output_dir=output_dir,
        )
        if ablation == "real":
            real_ref = metrics
            metrics["sigma_nmad_relative_to_real_pct"] = 0.0
            metrics["outlier_relative_to_real_pct"] = 0.0
        elif real_ref is not None:
            metrics["sigma_nmad_relative_to_real_pct"] = 100.0 * (metrics["sigma_nmad"] - real_ref["sigma_nmad"]) / real_ref["sigma_nmad"]
            metrics["outlier_relative_to_real_pct"] = 100.0 * (metrics["outlier_rate"] - real_ref["outlier_rate"]) / real_ref["outlier_rate"]
        summary_rows.append(metrics)

    write_rows_csv(os.path.join(output_dir, "metrics_marie_exact_augmented_summary.csv"), summary_rows)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    logger.info("Expérience Marie exacte augmentée terminée: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", default=["real", "i2i"])
    parser.add_argument("--synthetic_i2i", type=str, default=None)
    parser.add_argument("--filter_synthetic_mode", action="store_true")
    parser.add_argument("--max_synthetic", type=int, default=None)
    parser.add_argument("--augment_synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="cosmos_ud")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--n_bins", type=int, default=360)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=6.0)
    parser.add_argument("--mag_i_min", type=float, default=18.0)
    parser.add_argument("--mag_i_max", type=float, default=25.0)
    parser.add_argument("--mag_i_bins", type=int, default=14)
    parser.add_argument("--low_mag_support_quantile", type=float, default=0.20)
    parser.add_argument("--faint_mag_threshold", type=float, default=23.5)
    parser.add_argument("--smooth_distribution", action="store_true")
    parser.add_argument("--milestones", nargs="+", type=int, default=[35, 45])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--z_bins", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_parallel", action="store_true")
    run(parser.parse_args())
