import argparse
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset

from analysis_utils import aggregate_by_bins, compute_regression_metrics, ensure_dir, write_rows_csv, z_to_bin_indices
from config import CONFIG
from data_loader import build_metadata, get_dataset_and_splits
from density_utils import compute_catalog_knn_density, low_density_mask
from experiment_marie_baseline import MarieStyleBaseline, meta_from_cond
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def maybe_data_parallel(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Activation DataParallel Marie sur %s GPU visibles.", torch.cuda.device_count())
        return torch.nn.DataParallel(model)
    if enabled:
        logger.info("DataParallel Marie demandé mais un seul GPU est visible.")
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


class ClassicAugmentDataset(Dataset):
    '''
    actions : Produit des augmentations classiques label-preserving pour comparer la valeur ajoutée du CFM.
    inputs : base_dataset (Dataset), copies (int), noise_std (float)
    appels : torch.rot90, torch.flip, torch.randn_like
    outputs : Instance de Dataset
    '''
    def __init__(self, base_dataset: Dataset, copies: int = 1, noise_std: float = 0.02) -> None:
        self.base_dataset = base_dataset
        self.copies = max(1, copies)
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.base_dataset) * self.copies

    def __getitem__(self, idx: int):
        base_idx = idx % len(self.base_dataset)
        variant = (idx // len(self.base_dataset)) % 4
        x, cond = self.base_dataset[base_idx]
        x = x.clone()
        if variant == 1:
            x = torch.flip(x, dims=[2])
        elif variant == 2:
            x = torch.flip(x, dims=[1])
        elif variant == 3:
            x = torch.rot90(x, k=1, dims=[1, 2])
        if self.noise_std > 0.0:
            x = x + self.noise_std * torch.randn_like(x)
        return x, cond


def load_synthetic_dataset(
    path: str,
    max_samples: Optional[int] = None,
    seed: int = CONFIG.SEED,
    mode_filter: Optional[str] = None,
) -> TensorDataset:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset synthétique introuvable: {path}")
    data = np.load(path, allow_pickle=False)
    if "x" not in data or "cond" not in data:
        raise KeyError("Le dataset synthétique doit contenir les clés x et cond.")

    mask = np.ones(len(data["x"]), dtype=bool)
    if mode_filter is not None and "mode" in data.files:
        mask &= data["mode"] == mode_filter
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"Aucun échantillon synthétique disponible pour mode={mode_filter}.")
    if max_samples is not None and indices.size > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))

    x = torch.tensor(data["x"][indices], dtype=torch.float32)
    cond = torch.tensor(data["cond"][indices], dtype=torch.float32)
    return TensorDataset(x, cond)


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
        raise RuntimeError("Aucun batch évalué. Vérifiez le dataset, le split test ou --limit_batches.")
    return np.concatenate(z_true), np.concatenate(z_pred)


def density_edges(train_density: np.ndarray, n_bins: int = 5) -> np.ndarray:
    finite = train_density[np.isfinite(train_density)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    if np.unique(edges).size < len(edges):
        edges = np.linspace(float(np.min(finite)), float(np.max(finite)), n_bins + 1)
    if edges[0] == edges[-1]:
        edges = np.linspace(edges[0] - 0.5, edges[-1] + 0.5, n_bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def synthetic_path_for_ablation(args: argparse.Namespace, ablation: str) -> Optional[str]:
    return {
        "global": args.synthetic_global,
        "targeted_global": args.synthetic_targeted_global,
        "i2i": args.synthetic_i2i,
        "interp": args.synthetic_interp,
        "classic_i2i": args.synthetic_i2i,
    }.get(ablation)


def build_training_dataset(train_real: Dataset, ablation: str, args: argparse.Namespace) -> Dataset:
    if ablation == "real":
        return train_real
    if ablation == "classic":
        classic = ClassicAugmentDataset(train_real, copies=args.classic_copies, noise_std=args.classic_noise_std)
        return ConcatDataset([train_real, classic])
    if ablation == "classic_i2i":
        classic = ClassicAugmentDataset(train_real, copies=args.classic_copies, noise_std=args.classic_noise_std)
        synthetic = load_synthetic_dataset(
            synthetic_path_for_ablation(args, ablation),
            max_samples=args.max_synthetic,
            seed=args.seed,
            mode_filter="i2i" if args.filter_synthetic_mode else None,
        )
        return ConcatDataset([train_real, classic, synthetic])

    path = synthetic_path_for_ablation(args, ablation)
    synthetic = load_synthetic_dataset(
        path,
        max_samples=args.max_synthetic,
        seed=args.seed,
        mode_filter=ablation if args.filter_synthetic_mode else None,
    )
    return ConcatDataset([train_real, synthetic])


def evaluate_subsets(
    ablation: str,
    z_true: np.ndarray,
    z_pred: np.ndarray,
    test_density: np.ndarray,
    density_threshold: float,
    output_dir: str,
) -> Dict[str, float]:
    rows = []
    subsets = {
        "global": np.ones_like(z_true, dtype=bool),
        "low_density": test_density <= density_threshold,
        "normal_density": test_density > density_threshold,
    }
    global_metrics: Dict[str, float] = {}
    for subset_name, mask in subsets.items():
        metrics = compute_regression_metrics(z_true[mask], z_pred[mask])
        metrics.update({"ablation": ablation, "subset": subset_name})
        rows.append(metrics)
        if subset_name == "global":
            global_metrics = metrics
    write_rows_csv(os.path.join(output_dir, f"metrics_subsets_{ablation}.csv"), rows)
    return global_metrics


def run_single_ablation(
    ablation: str,
    train_real: Dataset,
    test_loader: DataLoader,
    test_indices: np.ndarray,
    density: np.ndarray,
    density_threshold: float,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
) -> Dict[str, float]:
    logger.info("=== Ablation Marie: %s ===", ablation)
    train_dataset = build_training_dataset(train_real, ablation, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    edges = np.array(CONFIG.Z_BIN_EDGES, dtype=np.float64)
    model = MarieStyleBaseline(n_bins=len(edges) - 1).to(device)
    model = maybe_data_parallel(model, args.data_parallel)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, edges, args.limit_batches)
        logger.info("[%s] Epoch %s/%s | Loss %.5f", ablation, epoch + 1, args.epochs, loss)

    torch.save(unwrap_model(model).state_dict(), os.path.join(output_dir, f"marie_augmented_{ablation}.pt"))
    z_true, z_pred = predict(model, test_loader, device, args.limit_batches)
    test_density = density[test_indices][: len(z_true)]

    np.savez(
        os.path.join(output_dir, f"predictions_marie_augmented_{ablation}.npz"),
        z_true=z_true,
        z_pred=z_pred,
        density=test_density,
    )
    global_metrics = evaluate_subsets(ablation, z_true, z_pred, test_density, density_threshold, output_dir)

    d_edges = density_edges(density[np.isfinite(density)], n_bins=args.density_bins)
    d_rows = aggregate_by_bins(test_density, d_edges, z_true, z_pred)
    for row in d_rows:
        row.update({"ablation": ablation})
    write_rows_csv(os.path.join(output_dir, f"metrics_by_density_{ablation}.csv"), d_rows)

    z_rows = aggregate_by_bins(z_true, edges, z_true, z_pred)
    for row in z_rows:
        row.update({"ablation": ablation})
    write_rows_csv(os.path.join(output_dir, f"metrics_by_z_true_{ablation}.csv"), z_rows)
    return global_metrics


def resolve_ablations(args: argparse.Namespace) -> List[str]:
    requested = args.ablations
    if "all" in requested:
        requested = ["real", "classic", "global", "targeted_global", "i2i", "interp", "classic_i2i"]
    valid = {"real", "classic", "global", "targeted_global", "i2i", "interp", "classic_i2i"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"Ablations inconnues: {unknown}")
    return requested


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir or CONFIG.exp_path("marie_augmented"))
    device = torch.device(CONFIG.DEVICE)
    dataset, split_indices = get_dataset_and_splits(
        region=args.region,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
    )
    metadata = build_metadata(dataset, split_indices=split_indices)
    density, _ = compute_catalog_knn_density(metadata["ra"], metadata["dec"], k=args.knn_k)
    _, density_threshold = low_density_mask(density, split_indices["train"], quantile=args.low_density_quantile)
    n_low_test = int(np.sum(density[split_indices["test"]] <= density_threshold))
    n_test = int(len(split_indices["test"]))
    logger.info(
        "Densité catalogue kNN: seuil train q=%.2f -> test low=%s normal=%s",
        args.low_density_quantile,
        n_low_test,
        n_test - n_low_test,
    )

    train_real = Subset(dataset, split_indices["train"])
    test_real = Subset(dataset, split_indices["test"])
    test_loader = DataLoader(
        test_real,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    summary_rows = []
    real_reference = None
    for ablation in resolve_ablations(args):
        try:
            metrics = run_single_ablation(
                ablation,
                train_real,
                test_loader,
                split_indices["test"],
                density,
                density_threshold,
                args,
                device,
                output_dir,
            )
        except FileNotFoundError as exc:
            logger.warning("Ablation %s ignorée: %s", ablation, exc)
            continue
        metrics = dict(metrics)
        if ablation == "real":
            real_reference = metrics
            metrics["sigma_nmad_relative_to_real_pct"] = 0.0
            metrics["outlier_relative_to_real_pct"] = 0.0
        elif real_reference is not None:
            metrics["sigma_nmad_relative_to_real_pct"] = (
                100.0 * (metrics["sigma_nmad"] - real_reference["sigma_nmad"]) / real_reference["sigma_nmad"]
                if real_reference["sigma_nmad"] != 0.0 else float("nan")
            )
            metrics["outlier_relative_to_real_pct"] = (
                100.0 * (metrics["outlier_rate"] - real_reference["outlier_rate"]) / real_reference["outlier_rate"]
                if real_reference["outlier_rate"] != 0.0 else float("nan")
            )
        summary_rows.append(metrics)

    write_rows_csv(os.path.join(output_dir, "metrics_marie_augmented_summary.csv"), summary_rows)
    logger.info("Expérience Marie augmentée terminée: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", default=["all"])
    parser.add_argument("--synthetic_global", type=str, default=None)
    parser.add_argument("--synthetic_targeted_global", type=str, default=None)
    parser.add_argument("--synthetic_i2i", type=str, default=None)
    parser.add_argument("--synthetic_interp", type=str, default=None)
    parser.add_argument("--filter_synthetic_mode", action="store_true")
    parser.add_argument("--max_synthetic", type=int, default=None)
    parser.add_argument("--classic_copies", type=int, default=1)
    parser.add_argument("--classic_noise_std", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=CONFIG.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CONFIG.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--low_density_quantile", type=float, default=0.20)
    parser.add_argument("--density_bins", type=int, default=5)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_parallel", action="store_true")
    run(parser.parse_args())
