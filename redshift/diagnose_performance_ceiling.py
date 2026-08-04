import argparse
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import (
    compute_regression_metrics,
    ensure_dir,
    load_metadata,
    residuals_normalized,
    sigma_nmad,
    write_rows_csv,
)
from config import CONFIG

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback for minimal environments
    cKDTree = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "z_true" not in data or "z_pred" not in data:
        raise KeyError("Le fichier predictions doit contenir z_true et z_pred.")
    return {key: data[key] for key in data.files}


def split_indices_from_metadata(metadata: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "split" not in metadata:
        raise KeyError("Metadata sans colonne split: impossible de calculer un support train-only strict.")
    split = np.asarray(metadata["split"]).astype(str)
    return {
        "train": np.where(split == "train")[0].astype(np.int64),
        "val": np.where(split == "val")[0].astype(np.int64),
        "test": np.where(split == "test")[0].astype(np.int64),
    }


def prediction_indices(predictions: Dict[str, np.ndarray], metadata: Dict[str, np.ndarray]) -> np.ndarray:
    n_pred = len(predictions["z_true"])
    if "test_indices" in predictions:
        indices = np.asarray(predictions["test_indices"], dtype=np.int64)
        if len(indices) < n_pred:
            raise ValueError("test_indices est plus court que z_true dans le fichier predictions.")
        return indices[:n_pred]
    if "split" in metadata:
        test_indices = np.where(np.asarray(metadata["split"]).astype(str) == "test")[0].astype(np.int64)
        if len(test_indices) >= n_pred:
            return test_indices[:n_pred]
    if len(metadata["z_true"]) == n_pred:
        logger.warning(
            "Predictions et metadata ont la meme taille, mais aucun test_indices explicite n'est present. "
            "Le support train-only reste impossible sans split complet."
        )
        return np.arange(n_pred, dtype=np.int64)
    raise ValueError(
        f"Impossible d'aligner metadata ({len(metadata['z_true'])}) et predictions ({n_pred}). "
        "Sauvegardez test_indices ou fournissez metadata avec split=test."
    )


def _first_present(metadata: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in metadata:
            return np.asarray(metadata[key], dtype=np.float64)
    raise KeyError(f"Aucune des cles metadata suivantes n'est disponible: {list(keys)}")


def required_magnitudes(metadata: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mag_i = _first_present(metadata, ["mag_i", "mag_marie_i"])
    return {
        "u": _first_present(metadata, ["mag_marie_us", "mag_marie_u", "mag_us", "mag_u"]),
        "g": _first_present(metadata, ["mag_marie_g", "mag_g"]),
        "r": _first_present(metadata, ["mag_marie_r", "mag_r"]),
        "i": mag_i,
        "z": _first_present(metadata, ["mag_marie_z", "mag_z"]),
        "y": _first_present(metadata, ["mag_marie_y", "mag_y"]),
    }


def photometric_features_from_metadata(metadata: Dict[str, np.ndarray], feature_space: str) -> Tuple[np.ndarray, List[str]]:
    mags = required_magnitudes(metadata)
    mag_i = mags["i"]
    g_r = mags["g"] - mags["r"]
    r_i = mags["r"] - mag_i
    i_z = mag_i - mags["z"]

    if feature_space == "classic_colors":
        return np.column_stack([mag_i, g_r, r_i, i_z]), ["mag_i", "g-r", "r-i", "i-z"]
    if feature_space == "marie_magnitudes":
        return np.column_stack([mags["u"], mags["g"], mags["r"], mag_i, mags["z"], mags["y"]]), [
            "mag_us",
            "mag_g",
            "mag_r",
            "mag_i",
            "mag_z",
            "mag_y",
        ]
    if feature_space == "marie_magnitudes_colors":
        return np.column_stack([mags["u"], mags["g"], mags["r"], mag_i, mags["z"], mags["y"], g_r, r_i, i_z]), [
            "mag_us",
            "mag_g",
            "mag_r",
            "mag_i",
            "mag_z",
            "mag_y",
            "g-r",
            "r-i",
            "i-z",
        ]
    raise ValueError("feature_space doit valoir classic_colors, marie_magnitudes ou marie_magnitudes_colors.")


def color_features_from_metadata(metadata: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mags = required_magnitudes(metadata)
    return {
        "mag_i": mags["i"],
        "g_r": mags["g"] - mags["r"],
        "r_i": mags["r"] - mags["i"],
        "i_z": mags["i"] - mags["z"],
        "z_y": mags["z"] - mags["y"],
    }


def standardize_on_train(features: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("features doit etre une matrice 2D.")
    if train_indices.size < 2:
        raise ValueError("Le split train doit contenir au moins deux objets.")
    ref = features[train_indices]
    mean = np.nanmean(ref, axis=0)
    std = np.nanstd(ref, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    standardized = (features - mean[None, :]) / std[None, :]
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def _query_reference_knn(
    standardized_features: np.ndarray,
    reference_indices: np.ndarray,
    query_indices: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if k <= 0:
        raise ValueError("k doit etre strictement positif.")
    if reference_indices.size < 2:
        raise ValueError("reference_indices doit contenir au moins deux objets.")

    reference = standardized_features[reference_indices]
    query = standardized_features[query_indices]
    n_query = min(reference_indices.size, k + 1)
    if cKDTree is not None:
        tree = cKDTree(reference)
        distances, positions = tree.query(query, k=n_query)
    else:
        diff = query[:, None, :] - reference[None, :, :]
        distance_matrix = np.sqrt(np.sum(diff * diff, axis=2))
        positions = np.argsort(distance_matrix, axis=1)[:, :n_query]
        distances = np.take_along_axis(distance_matrix, positions, axis=1)

    distances = np.asarray(distances, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        positions = positions[:, None]

    neighbor_indices = np.full((len(query_indices), k), -1, dtype=np.int64)
    neighbor_distances = np.full((len(query_indices), k), np.nan, dtype=np.float64)
    for row, query_index in enumerate(query_indices):
        actual = reference_indices[positions[row]]
        keep = actual != query_index
        actual = actual[keep]
        dist = distances[row][keep]
        n_keep = min(k, actual.size)
        if n_keep:
            neighbor_indices[row, :n_keep] = actual[:n_keep]
            neighbor_distances[row, :n_keep] = dist[:n_keep]
    return neighbor_distances, neighbor_indices


def local_redshift_ambiguity(z_train_reference: np.ndarray, neighbor_indices: np.ndarray) -> Dict[str, np.ndarray]:
    z_train_reference = np.asarray(z_train_reference, dtype=np.float64)
    local_std = np.full(neighbor_indices.shape[0], np.nan, dtype=np.float64)
    local_nmad = np.full(neighbor_indices.shape[0], np.nan, dtype=np.float64)
    local_nmad_norm = np.full(neighbor_indices.shape[0], np.nan, dtype=np.float64)
    local_median_z = np.full(neighbor_indices.shape[0], np.nan, dtype=np.float64)

    for row, indices in enumerate(neighbor_indices):
        valid = indices[indices >= 0]
        z_values = z_train_reference[valid]
        z_values = z_values[np.isfinite(z_values)]
        if z_values.size == 0:
            continue
        median_z = float(np.median(z_values))
        denom = 1.0 + median_z
        local_median_z[row] = median_z
        local_std[row] = float(np.std(z_values))
        local_nmad[row] = sigma_nmad(z_values)
        local_nmad_norm[row] = sigma_nmad((z_values - median_z) / denom)

    return {
        "local_z_std": local_std,
        "local_z_nmad": local_nmad,
        "local_z_nmad_norm": local_nmad_norm,
        "local_median_z": local_median_z,
    }


def finite_quantile_range(values: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.nanquantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        lo, hi = finite_quantile_range(finite, 0.0, 100.0)
        edges = np.linspace(lo, hi, n_bins + 1)
    return edges


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if np.nanstd(x_valid) <= 0.0 or np.nanstd(y_valid) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    return _pearson_corr(_rankdata(x[mask]), _rankdata(y[mask]))


def aggregate_by_edges(
    name: str,
    values: np.ndarray,
    edges: np.ndarray,
    z_true: np.ndarray,
    z_pred: np.ndarray,
    abs_dz: np.ndarray,
    support_radius: np.ndarray,
    local_ambiguity: np.ndarray,
    mag_i: np.ndarray,
    error_to_ambiguity: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    values = np.asarray(values, dtype=np.float64)
    for bin_id in range(len(edges) - 1):
        lo = float(edges[bin_id])
        hi = float(edges[bin_id + 1])
        if bin_id == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        metrics = compute_regression_metrics(z_true[mask], z_pred[mask])
        finite_abs = abs_dz[mask]
        finite_support = support_radius[mask]
        finite_ambiguity = local_ambiguity[mask]
        finite_mag_i = mag_i[mask]
        finite_ratio = error_to_ambiguity[mask]
        metrics.update({
            "axis": name,
            "bin": bin_id,
            "bin_min": lo,
            "bin_max": hi,
            "bin_center": 0.5 * (lo + hi),
            "median_abs_dz_norm": _safe_nanmedian(finite_abs),
            "median_support_radius": _safe_nanmedian(finite_support),
            "median_local_z_nmad_norm": _safe_nanmedian(finite_ambiguity),
            "median_mag_i": _safe_nanmedian(finite_mag_i),
            "median_error_to_ambiguity": _safe_nanmedian(finite_ratio),
        })
        rows.append(metrics)
    return rows


def _safe_nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def grid_statistic_2d(
    x: np.ndarray,
    y: np.ndarray,
    values: Optional[np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    statistic: str,
    min_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    x_bin[np.isfinite(x) & (x == x_edges[-1])] = len(x_edges) - 2
    y_bin[np.isfinite(y) & (y == y_edges[-1])] = len(y_edges) - 2
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x_bin >= 0)
        & (x_bin < len(x_edges) - 1)
        & (y_bin >= 0)
        & (y_bin < len(y_edges) - 1)
    )
    if values is not None:
        values = np.asarray(values, dtype=np.float64)
        valid &= np.isfinite(values)

    ny = len(y_edges) - 1
    nx = len(x_edges) - 1
    counts = np.zeros((ny, nx), dtype=np.int64)
    stats = np.full((ny, nx), np.nan, dtype=np.float64)
    if not np.any(valid):
        return stats, counts

    flat = y_bin[valid] * nx + x_bin[valid]
    unique = np.unique(flat)
    valid_values = values[valid] if values is not None else None
    for cell in unique:
        cell_mask = flat == cell
        count = int(np.sum(cell_mask))
        row = int(cell // nx)
        col = int(cell % nx)
        counts[row, col] = count
        if count < min_count:
            continue
        if values is None or statistic == "count":
            stats[row, col] = float(count)
        elif statistic == "median":
            stats[row, col] = float(np.median(valid_values[cell_mask]))
        elif statistic == "mean":
            stats[row, col] = float(np.mean(valid_values[cell_mask]))
        else:
            raise ValueError("statistic doit valoir count, median ou mean.")
    return stats, counts


def plot_grid(
    ax: plt.Axes,
    grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(image, ax=ax, label=colorbar_label)


def robust_vmax(values: np.ndarray, percentile: float = 95.0) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.nanpercentile(finite, percentile))


def plot_color_color_surfaces(
    output_path: str,
    train_colors: Dict[str, np.ndarray],
    eval_colors: Dict[str, np.ndarray],
    abs_dz: np.ndarray,
    local_ambiguity: np.ndarray,
    error_to_ambiguity: np.ndarray,
    bins: int,
    min_cell_count: int,
) -> None:
    x_all = np.concatenate([train_colors["g_r"], eval_colors["g_r"]])
    y_all = np.concatenate([train_colors["r_i"], eval_colors["r_i"]])
    x_edges = np.linspace(*finite_quantile_range(x_all, 1, 99), bins + 1)
    y_edges = np.linspace(*finite_quantile_range(y_all, 1, 99), bins + 1)

    train_count, _ = grid_statistic_2d(train_colors["g_r"], train_colors["r_i"], None, x_edges, y_edges, "count", 1)
    ambiguity_grid, _ = grid_statistic_2d(eval_colors["g_r"], eval_colors["r_i"], local_ambiguity, x_edges, y_edges, "median", min_cell_count)
    error_grid, _ = grid_statistic_2d(eval_colors["g_r"], eval_colors["r_i"], abs_dz, x_edges, y_edges, "median", min_cell_count)
    ratio_grid, _ = grid_statistic_2d(eval_colors["g_r"], eval_colors["r_i"], error_to_ambiguity, x_edges, y_edges, "median", min_cell_count)

    count_grid = np.where(train_count > 0, np.log10(train_count), np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_grid(axes[0, 0], count_grid, x_edges, y_edges, "Densite train", "log10(N train)", "magma")
    plot_grid(
        axes[0, 1],
        ambiguity_grid,
        x_edges,
        y_edges,
        "Ambiguite locale du redshift",
        "sigma_NMAD locale normalisee",
        "viridis",
        vmax=robust_vmax(ambiguity_grid),
    )
    plot_grid(
        axes[1, 0],
        error_grid,
        x_edges,
        y_edges,
        "Erreur modele",
        "median |dz|/(1+z)",
        "inferno",
        vmax=robust_vmax(error_grid),
    )
    plot_grid(
        axes[1, 1],
        ratio_grid,
        x_edges,
        y_edges,
        "Erreur / ambiguite locale",
        "ratio median",
        "plasma",
        vmax=robust_vmax(ratio_grid),
    )
    for ax in axes.ravel():
        ax.set_xlabel("g-r")
        ax.set_ylabel("r-i")
    fig.suptitle("Plafond de performance dans l'espace couleur-couleur", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_support_magnitude_surfaces(
    output_path: str,
    support_radius: np.ndarray,
    mag_i: np.ndarray,
    abs_dz: np.ndarray,
    outlier: np.ndarray,
    local_ambiguity: np.ndarray,
    error_to_ambiguity: np.ndarray,
    bins: int,
    min_cell_count: int,
) -> None:
    x_edges = np.linspace(*finite_quantile_range(support_radius, 1, 99), bins + 1)
    y_edges = np.linspace(*finite_quantile_range(mag_i, 1, 99), bins + 1)
    count_grid, _ = grid_statistic_2d(support_radius, mag_i, None, x_edges, y_edges, "count", 1)
    ambiguity_grid, _ = grid_statistic_2d(support_radius, mag_i, local_ambiguity, x_edges, y_edges, "median", min_cell_count)
    error_grid, _ = grid_statistic_2d(support_radius, mag_i, abs_dz, x_edges, y_edges, "median", min_cell_count)
    outlier_grid, _ = grid_statistic_2d(support_radius, mag_i, outlier.astype(float) * 100.0, x_edges, y_edges, "mean", min_cell_count)
    ratio_grid, _ = grid_statistic_2d(support_radius, mag_i, error_to_ambiguity, x_edges, y_edges, "median", min_cell_count)

    count_grid = np.where(count_grid > 0, np.log10(count_grid), np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_grid(axes[0, 0], count_grid, x_edges, y_edges, "Densite evaluation", "log10(N eval)", "magma")
    plot_grid(
        axes[0, 1],
        ambiguity_grid,
        x_edges,
        y_edges,
        "Ambiguite locale",
        "sigma_NMAD locale normalisee",
        "viridis",
        vmax=robust_vmax(ambiguity_grid),
    )
    plot_grid(
        axes[1, 0],
        error_grid,
        x_edges,
        y_edges,
        "Erreur modele",
        "median |dz|/(1+z)",
        "inferno",
        vmax=robust_vmax(error_grid),
    )
    plot_grid(
        axes[1, 1],
        ratio_grid,
        x_edges,
        y_edges,
        "Erreur / ambiguite",
        "ratio median",
        "plasma",
        vmax=robust_vmax(ratio_grid),
    )
    for ax in axes.ravel():
        ax.set_xlabel("Rayon kNN photometrique train")
        ax.set_ylabel("mag_i")
        ax.invert_yaxis()
    fig.suptitle("Plafond de performance par support photometrique et magnitude", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_grid(
        ax,
        outlier_grid,
        x_edges,
        y_edges,
        "Taux d'outliers par support photometrique et magnitude",
        "outliers (%)",
        "magma",
        vmax=robust_vmax(outlier_grid),
    )
    ax.set_xlabel("Rayon kNN photometrique train")
    ax.set_ylabel("mag_i")
    ax.invert_yaxis()
    fig.tight_layout()
    outlier_path = os.path.splitext(output_path)[0] + "_outlier_rate.png"
    fig.savefig(outlier_path, dpi=160)
    plt.close(fig)


def plot_ambiguity_error_relation(
    output_path: str,
    local_ambiguity: np.ndarray,
    abs_dz: np.ndarray,
    rows: List[Dict[str, object]],
    max_points: int,
    seed: int,
) -> None:
    x = np.asarray(local_ambiguity, dtype=np.float64)
    y = np.asarray(abs_dz, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return

    if x.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x_plot = x[idx]
        y_plot = y[idx]
    else:
        x_plot = x
        y_plot = y

    fig, ax = plt.subplots(figsize=(8, 6))
    x_max = robust_vmax(x, 99) or float(np.max(x))
    y_max = robust_vmax(y, 99) or float(np.max(y))
    ax.hexbin(x_plot, y_plot, gridsize=60, mincnt=1, cmap="magma", extent=(0, x_max, 0, y_max))
    centers = np.array([row["bin_center"] for row in rows], dtype=np.float64)
    med_error = np.array([row["median_abs_dz_norm"] for row in rows], dtype=np.float64)
    valid_curve = np.isfinite(centers) & np.isfinite(med_error)
    ax.plot(centers[valid_curve], med_error[valid_curve], color="cyan", marker="o", linewidth=1.8, label="mediane par bin")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Ambiguite locale train: sigma_NMAD(z voisins)/(1+z local)")
    ax.set_ylabel("|z_pred - z_true|/(1+z_true)")
    ax.set_title("Erreur modele vs ambiguite locale du probleme")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_support_curves(output_path: str, rows: List[Dict[str, object]]) -> None:
    centers = np.array([row["bin_center"] for row in rows], dtype=np.float64)
    sigma = np.array([row["sigma_nmad"] for row in rows], dtype=np.float64)
    outlier = np.array([row["outlier_rate"] for row in rows], dtype=np.float64)
    ambiguity = np.array([row["median_local_z_nmad_norm"] for row in rows], dtype=np.float64)
    median_mag = np.array([row["median_mag_i"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(centers, sigma, marker="o")
    axes[0, 0].set_ylabel("sigma_NMAD modele")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(centers, outlier, marker="o", color="tab:red")
    axes[0, 1].set_ylabel("outliers (%)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(centers, ambiguity, marker="o", color="tab:green")
    axes[1, 0].set_ylabel("ambiguite locale")
    axes[1, 0].set_xlabel("Rayon kNN photometrique train")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(centers, median_mag, marker="o", color="tab:purple")
    axes[1, 1].set_ylabel("mag_i median")
    axes[1, 1].set_xlabel("Rayon kNN photometrique train")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle("Courbes de diagnostic par support photometrique", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_markdown_report(output_path: str, summary: Dict[str, object], feature_names: Sequence[str]) -> None:
    lines = [
        "# Diagnostic du plafond de performance",
        "",
        "Ce rapport diagnostique compare l'erreur du modele au support photometrique train-only et a l'ambiguite locale du redshift.",
        "",
        "## Definition",
        "",
        "- Support photometrique: rayon au k-ieme voisin dans l'espace de features standardise sur le train.",
        "- Ambiguite locale: dispersion robuste des redshifts des voisins train autour de chaque objet evalue.",
        "- Ratio erreur/ambiguite: erreur absolue normalisee divisee par l'ambiguite locale.",
        "",
        "Features utilisees:",
        "",
        "```text",
        ", ".join(feature_names),
        "```",
        "",
        "## Resume numerique",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Figures",
        "",
        "- `ceiling_color_color_surfaces.png`: densite train, ambiguite locale, erreur modele et ratio dans le plan g-r / r-i.",
        "- `ceiling_support_magnitude_surfaces.png`: meme diagnostic dans le plan support photometrique / mag_i.",
        "- `ceiling_support_magnitude_surfaces_outlier_rate.png`: taux d'outliers par support et magnitude.",
        "- `ceiling_ambiguity_vs_error.png`: relation directe entre ambiguite locale et erreur du modele.",
        "- `ceiling_support_curves.png`: courbes de metriques par bins de support photometrique.",
        "",
        "Interpretation attendue: si l'erreur augmente surtout avec l'ambiguite locale et le faible support, le plafond vient en grande partie des donnees et du probleme photo-z. Si l'erreur reste forte dans des zones bien supportees et peu ambigues, le modele garde une marge d'amelioration.",
        "",
    ])
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    predictions = load_predictions(args.predictions)
    metadata = load_metadata(args.metadata)
    splits = split_indices_from_metadata(metadata)
    train_indices = splits["train"]
    eval_indices = prediction_indices(predictions, metadata)

    z_true = np.asarray(predictions["z_true"], dtype=np.float64)
    z_pred = np.asarray(predictions["z_pred"], dtype=np.float64)
    n_eval = min(len(z_true), len(z_pred), len(eval_indices))
    z_true = z_true[:n_eval]
    z_pred = z_pred[:n_eval]
    eval_indices = eval_indices[:n_eval]

    if np.max(eval_indices, initial=-1) >= len(metadata["z_true"]):
        raise ValueError("eval_indices contient des indices hors metadata.")
    metadata_z = np.asarray(metadata["z_true"], dtype=np.float64)[eval_indices]
    mismatch = np.nanmedian(np.abs(metadata_z - z_true)) if n_eval else 0.0
    if np.isfinite(mismatch) and mismatch > 1e-5:
        logger.warning("z_true predictions et metadata ne sont pas parfaitement alignes (median abs diff=%s).", mismatch)

    features, feature_names = photometric_features_from_metadata(metadata, args.feature_space)
    standardized = standardize_on_train(features, train_indices)

    train_dist, train_neighbors = _query_reference_knn(standardized, train_indices, train_indices, args.k)
    eval_dist, eval_neighbors = _query_reference_knn(standardized, train_indices, eval_indices, args.k)
    train_radius = train_dist[:, -1]
    eval_radius = eval_dist[:, -1]
    radius_threshold = float(np.nanquantile(train_radius[np.isfinite(train_radius)], 1.0 - args.low_fraction))
    low_support_eval = np.isfinite(eval_radius) & (eval_radius >= radius_threshold)

    ambiguity = local_redshift_ambiguity(np.asarray(metadata["z_true"], dtype=np.float64), eval_neighbors)
    local_z_nmad_norm = ambiguity["local_z_nmad_norm"]

    dz = residuals_normalized(z_true, z_pred)
    abs_dz = np.abs(dz)
    outlier = abs_dz > 0.15
    colors_all = color_features_from_metadata(metadata)
    train_colors = {key: values[train_indices] for key, values in colors_all.items()}
    eval_colors = {key: values[eval_indices] for key, values in colors_all.items()}
    mag_i_eval = eval_colors["mag_i"]
    error_to_ambiguity = abs_dz / np.clip(local_z_nmad_norm, args.ambiguity_floor, None)

    global_metrics = compute_regression_metrics(z_true, z_pred)
    low_metrics = compute_regression_metrics(z_true[low_support_eval], z_pred[low_support_eval])
    summary = {
        "feature_space": args.feature_space,
        "feature_names": ",".join(feature_names),
        "k": int(args.k),
        "n_train": int(len(train_indices)),
        "n_eval": int(n_eval),
        "low_support_fraction": float(args.low_fraction),
        "low_support_radius_threshold": radius_threshold,
        "eval_low_support_n": int(np.sum(low_support_eval)),
        "global_sigma_nmad": global_metrics["sigma_nmad"],
        "global_rmse": global_metrics["rmse"],
        "global_outlier_rate": global_metrics["outlier_rate"],
        "low_support_sigma_nmad": low_metrics["sigma_nmad"],
        "low_support_rmse": low_metrics["rmse"],
        "low_support_outlier_rate": low_metrics["outlier_rate"],
        "median_abs_dz_norm": _safe_nanmedian(abs_dz),
        "median_support_radius": _safe_nanmedian(eval_radius),
        "median_local_z_nmad_norm": _safe_nanmedian(local_z_nmad_norm),
        "median_error_to_ambiguity": _safe_nanmedian(error_to_ambiguity),
        "spearman_abs_error_vs_support_radius": _spearman_corr(abs_dz, eval_radius),
        "spearman_abs_error_vs_local_ambiguity": _spearman_corr(abs_dz, local_z_nmad_norm),
        "pearson_abs_error_vs_support_radius": _pearson_corr(abs_dz, eval_radius),
        "pearson_abs_error_vs_local_ambiguity": _pearson_corr(abs_dz, local_z_nmad_norm),
    }
    write_rows_csv(os.path.join(output_dir, "ceiling_summary.csv"), [summary])

    support_rows = aggregate_by_edges(
        "support_radius",
        eval_radius,
        quantile_edges(eval_radius, args.curve_bins),
        z_true,
        z_pred,
        abs_dz,
        eval_radius,
        local_z_nmad_norm,
        mag_i_eval,
        error_to_ambiguity,
    )
    ambiguity_rows = aggregate_by_edges(
        "local_z_nmad_norm",
        local_z_nmad_norm,
        quantile_edges(local_z_nmad_norm, args.curve_bins),
        z_true,
        z_pred,
        abs_dz,
        eval_radius,
        local_z_nmad_norm,
        mag_i_eval,
        error_to_ambiguity,
    )
    write_rows_csv(os.path.join(output_dir, "ceiling_by_support_radius.csv"), support_rows)
    write_rows_csv(os.path.join(output_dir, "ceiling_by_local_ambiguity.csv"), ambiguity_rows)

    plot_color_color_surfaces(
        os.path.join(output_dir, "ceiling_color_color_surfaces.png"),
        train_colors,
        eval_colors,
        abs_dz,
        local_z_nmad_norm,
        error_to_ambiguity,
        bins=args.surface_bins,
        min_cell_count=args.min_cell_count,
    )
    plot_support_magnitude_surfaces(
        os.path.join(output_dir, "ceiling_support_magnitude_surfaces.png"),
        eval_radius,
        mag_i_eval,
        abs_dz,
        outlier,
        local_z_nmad_norm,
        error_to_ambiguity,
        bins=args.surface_bins,
        min_cell_count=args.min_cell_count,
    )
    plot_ambiguity_error_relation(
        os.path.join(output_dir, "ceiling_ambiguity_vs_error.png"),
        local_z_nmad_norm,
        abs_dz,
        ambiguity_rows,
        max_points=args.max_scatter_points,
        seed=args.seed,
    )
    plot_support_curves(os.path.join(output_dir, "ceiling_support_curves.png"), support_rows)
    write_markdown_report(os.path.join(output_dir, "ceiling_diagnostic_report.md"), summary, feature_names)
    logger.info("Diagnostic du plafond de performance sauvegarde dans %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--feature_space",
        choices=["classic_colors", "marie_magnitudes", "marie_magnitudes_colors"],
        default="classic_colors",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--low_fraction", type=float, default=0.20)
    parser.add_argument("--curve_bins", type=int, default=10)
    parser.add_argument("--surface_bins", type=int, default=55)
    parser.add_argument("--min_cell_count", type=int, default=20)
    parser.add_argument("--max_scatter_points", type=int, default=50000)
    parser.add_argument("--ambiguity_floor", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    run(parser.parse_args())
