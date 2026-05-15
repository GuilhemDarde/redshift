import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import CONFIG


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def stripe82_mask(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_min: float = CONFIG.STRIPE82_RA_MIN,
    ra_max: float = CONFIG.STRIPE82_RA_MAX,
    dec_min: float = CONFIG.STRIPE82_DEC_MIN,
    dec_max: float = CONFIG.STRIPE82_DEC_MAX,
) -> np.ndarray:
    ra_norm = np.mod(ra, 360.0)
    if ra_min <= ra_max:
        ra_ok = (ra_norm >= ra_min) & (ra_norm <= ra_max)
    else:
        ra_ok = (ra_norm >= ra_min) | (ra_norm <= ra_max)
    dec_ok = (dec >= dec_min) & (dec <= dec_max)
    return ra_ok & dec_ok


def apply_region_mask(metadata: Dict[str, np.ndarray], region: str = "all") -> np.ndarray:
    n = len(metadata["z_true"])
    if region == "all":
        return np.ones(n, dtype=bool)
    if region == "stripe82":
        return stripe82_mask(metadata["ra"], metadata["dec"])
    raise ValueError(f"Region inconnue: {region}. Valeurs attendues: all, stripe82")


def compute_split_indices(
    ra_values: np.ndarray,
    n_folds: Optional[int] = None,
    fold_id: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    sorted_indices = np.argsort(ra_values)
    total = len(sorted_indices)

    if n_folds is not None and fold_id is not None:
        if n_folds < 3:
            raise ValueError("n_folds doit etre >= 3 pour avoir train/val/test.")
        if fold_id < 0 or fold_id >= n_folds:
            raise ValueError("fold_id doit etre dans [0, n_folds).")

        folds = np.array_split(sorted_indices, n_folds)
        test_idx = folds[fold_id]
        val_idx = folds[(fold_id + 1) % n_folds]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i not in {fold_id, (fold_id + 1) % n_folds}])
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    train_size = int(0.80 * total)
    val_size = int(0.10 * total)
    return {
        "train": sorted_indices[:train_size],
        "val": sorted_indices[train_size : train_size + val_size],
        "test": sorted_indices[train_size + val_size :],
    }


def compute_marie_regular_cv_indices(
    n_samples: int,
    n_folds: int = CONFIG.N_FOLDS,
    fold_id: int = 0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Reproduit le split cross-validation regulier vu dans le code Marie.

    Marie melange les indices valides avec seed=42, les coupe en 5 blocs, puis
    utilise un bloc comme validation/test et tous les autres comme train.
    """
    if n_samples <= 0:
        raise ValueError("n_samples doit etre strictement positif.")
    if n_folds < 2:
        raise ValueError("n_folds doit etre >= 2 pour le split Marie.")
    if fold_id < 0 or fold_id >= n_folds:
        raise ValueError("fold_id doit etre dans [0, n_folds).")
    indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    test_idx = np.asarray(folds[fold_id], dtype=np.int64)
    train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_id]).astype(np.int64)
    return {"train": train_idx, "val": test_idx, "test": test_idx}


def split_labels(n_samples: int, split_indices: Dict[str, np.ndarray]) -> np.ndarray:
    labels = np.full(n_samples, "unassigned", dtype="<U10")
    for split_name, idx in split_indices.items():
        labels[idx] = split_name
    return labels


def residuals_normalized(z_true: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    return (z_pred - z_true) / (1.0 + z_true)


def sigma_nmad(values: np.ndarray) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def compute_regression_metrics(z_true: np.ndarray, z_pred: np.ndarray, z_sigma: Optional[np.ndarray] = None) -> Dict[str, float]:
    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)
    mask = np.isfinite(z_true) & np.isfinite(z_pred)
    if not np.any(mask):
        return {
            "n": 0,
            "bias": float("nan"),
            "sigma_nmad": float("nan"),
            "rmse": float("nan"),
            "outlier_rate": float("nan"),
            "median_sigma": float("nan"),
        }

    dz = residuals_normalized(z_true[mask], z_pred[mask])
    metrics = {
        "n": int(np.sum(mask)),
        "bias": float(np.mean(dz)),
        "sigma_nmad": sigma_nmad(dz),
        "rmse": float(np.sqrt(np.mean((z_pred[mask] - z_true[mask]) ** 2))),
        "outlier_rate": float(np.mean(np.abs(dz) > 0.15) * 100.0),
        "median_sigma": float("nan"),
    }
    if z_sigma is not None:
        sigma = np.asarray(z_sigma)[mask]
        sigma = sigma[np.isfinite(sigma)]
        if sigma.size:
            metrics["median_sigma"] = float(np.median(sigma))
    return metrics


def bin_edges_from_range(values: np.ndarray, n_bins: int, value_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if value_range is None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.linspace(0.0, 1.0, n_bins + 1)
        value_range = (float(np.min(finite)), float(np.max(finite)))
    return np.linspace(value_range[0], value_range[1], n_bins + 1)


def aggregate_by_bins(
    values: np.ndarray,
    edges: np.ndarray,
    z_true: np.ndarray,
    z_pred: np.ndarray,
    z_sigma: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        metrics = compute_regression_metrics(z_true[mask], z_pred[mask], z_sigma[mask] if z_sigma is not None else None)
        metrics.update({"bin": i, "bin_min": lo, "bin_max": hi, "bin_center": 0.5 * (lo + hi)})
        rows.append(metrics)
    return rows


def magnitude_bin_edges(
    mag_i_min: float = CONFIG.I_MIN,
    mag_i_max: float = CONFIG.I_MAX,
    n_bins: int = 14,
) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins doit etre strictement positif.")
    if mag_i_max <= mag_i_min:
        raise ValueError("mag_i_max doit etre superieur a mag_i_min.")
    return np.linspace(float(mag_i_min), float(mag_i_max), int(n_bins) + 1)


def magnitude_support_mask(
    mag_i: np.ndarray,
    reference_mag_i: np.ndarray,
    edges: np.ndarray,
    quantile: float = 0.20,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifie les objets situes dans les bins de magnitude i les moins representes.

    Le support d'un objet est le nombre d'objets du set de reference dans son bin
    de magnitude. Les bins rares sont ceux dont le support est inferieur ou egal
    au quantile demande, calcule uniquement sur les bins non vides du reference.
    """
    mag_i = np.asarray(mag_i, dtype=np.float64)
    reference_mag_i = np.asarray(reference_mag_i, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges doit contenir au moins deux bornes.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile doit etre dans [0, 1].")

    ref_valid = reference_mag_i[np.isfinite(reference_mag_i)]
    counts, _ = np.histogram(ref_valid, bins=edges)
    nonzero = counts[counts > 0]
    threshold = float(np.quantile(nonzero, quantile)) if nonzero.size else float("nan")

    bin_id = np.digitize(mag_i, edges) - 1
    bin_id[np.isfinite(mag_i) & (mag_i == edges[-1])] = len(counts) - 1
    valid = np.isfinite(mag_i) & (bin_id >= 0) & (bin_id < len(counts))
    support = np.full(len(mag_i), np.nan, dtype=np.float64)
    support[valid] = counts[bin_id[valid]]

    low = valid & np.isfinite(threshold) & (support <= threshold)
    return low, threshold, support, bin_id.astype(np.int64), counts.astype(np.int64)


def magnitude_support_definition_rows(edges: np.ndarray, counts: np.ndarray, threshold: float) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for idx, count in enumerate(np.asarray(counts, dtype=np.int64)):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        rows.append({
            "bin": idx,
            "bin_min": lo,
            "bin_max": hi,
            "bin_center": 0.5 * (lo + hi),
            "train_count": int(count),
            "low_mag_support": bool(np.isfinite(threshold) and count <= threshold),
            "train_count_threshold": threshold,
        })
    return rows


def write_rows_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    metadata_path = path or CONFIG.METADATA_PATH or CONFIG.exp_path(CONFIG.DATASET_METADATA)
    data = np.load(metadata_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def save_metadata_npz(path: str, metadata: Dict[str, np.ndarray]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.savez(path, **metadata)


def save_metadata_csv(path: str, metadata: Dict[str, np.ndarray]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    n = len(metadata["z_true"])
    rows = []
    scalar_keys = [k for k, v in metadata.items() if np.asarray(v).ndim == 1 and len(v) == n]
    for i in range(n):
        row = {}
        for key in scalar_keys:
            val = metadata[key][i]
            row[key] = val.item() if hasattr(val, "item") else val
        rows.append(row)
    write_rows_csv(path, rows)


def z_bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def z_to_bin_indices(z_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(z_values, edges) - 1, 0, len(edges) - 2)
