import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import compute_regression_metrics, ensure_dir, load_metadata, residuals_normalized, write_rows_csv
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "z_true" not in data or "z_pred" not in data:
        raise KeyError("Le fichier predictions doit contenir z_true et z_pred.")
    return {key: data[key] for key in data.files}


def align_metadata_for_predictions(metadata: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n = len(predictions["z_true"])
    if "test_indices" in predictions:
        idx = predictions["test_indices"].astype(np.int64)
        return {key: value[idx] if np.asarray(value).shape[:1] == (len(metadata["z_true"]),) else value for key, value in metadata.items()}
    if "split" in metadata and np.sum(metadata["split"] == "test") == n:
        mask = metadata["split"] == "test"
        return {key: value[mask] if np.asarray(value).shape[:1] == (len(mask),) else value for key, value in metadata.items()}
    if len(metadata["z_true"]) == n:
        return metadata
    raise ValueError(
        f"Impossible d'aligner metadata ({len(metadata['z_true'])}) et predictions ({n}). "
        "Sauvegardez test_indices dans le fichier predictions ou utilisez une metadata avec split=test."
    )


def median_absolute_deviation(values: np.ndarray) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def _row_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> Dict[str, float]:
    metrics = compute_regression_metrics(z_true, z_pred)
    dz = residuals_normalized(np.asarray(z_true), np.asarray(z_pred))
    metrics["mad"] = median_absolute_deviation(dz)
    return metrics


def _bootstrap_errors(z_true: np.ndarray, z_pred: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, float]:
    if n_bootstrap <= 0 or len(z_true) < 2:
        return {"bias_err": float("nan"), "mad_err": float("nan"), "sigma_nmad_err": float("nan"), "outlier_rate_err": float("nan")}
    rng = np.random.default_rng(seed)
    values = {"bias": [], "mad": [], "sigma_nmad": [], "outlier_rate": []}
    n = len(z_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        row = _row_metrics(z_true[idx], z_pred[idx])
        for key in values:
            values[key].append(row[key])
    return {f"{key}_err": float(np.nanstd(vals, ddof=1)) for key, vals in values.items()}


def aggregate_treyer_bins(
    values: np.ndarray,
    edges: np.ndarray,
    z_true: np.ndarray,
    z_pred: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = CONFIG.SEED,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    values = np.asarray(values)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        row = _row_metrics(z_true[mask], z_pred[mask])
        row.update(_bootstrap_errors(z_true[mask], z_pred[mask], n_bootstrap=n_bootstrap, seed=seed + i))
        row.update({"bin": i, "bin_min": lo, "bin_max": hi, "bin_center": 0.5 * (lo + hi)})
        rows.append(row)
    return rows


def magnitude_support_rows(
    mag_i: np.ndarray,
    train_mag_i: np.ndarray,
    z_true: np.ndarray,
    z_pred: np.ndarray,
    edges: np.ndarray,
    quantile: float = 0.20,
) -> List[Dict[str, float]]:
    counts, _ = np.histogram(train_mag_i, bins=edges)
    nonzero = counts[counts > 0]
    threshold = float(np.quantile(nonzero, quantile)) if len(nonzero) else float("nan")
    bin_id = np.digitize(mag_i, edges) - 1
    valid = (bin_id >= 0) & (bin_id < len(counts))
    support = np.full(len(mag_i), np.nan, dtype=np.float64)
    support[valid] = counts[bin_id[valid]]
    low = valid & (support <= threshold)
    normal = valid & (support > threshold)
    rows = []
    for name, mask in [("global", valid), ("low_mag_support", low), ("normal_mag_support", normal)]:
        row = _row_metrics(z_true[mask], z_pred[mask])
        row.update({"subset": name, "train_count_threshold": threshold, "n_eval_valid_mag": int(np.sum(valid))})
        rows.append(row)
    return rows


def _plot_metric(ax, rows: List[Dict[str, float]], y_key: str, err_key: Optional[str], ylabel: str) -> None:
    x = np.asarray([row["bin_center"] for row in rows], dtype=float)
    y = np.asarray([row[y_key] for row in rows], dtype=float)
    yerr = np.asarray([row.get(err_key, np.nan) for row in rows], dtype=float) if err_key else None
    if yerr is not None and np.isfinite(yerr).any():
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.4, capsize=2)
    else:
        ax.plot(x, y, marker="o", linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4) if y_key == "bias" else None
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def plot_figure7_like(z_rows: List[Dict[str, float]], mag_rows: List[Dict[str, float]], output_path: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex="col")
    groups = [(z_rows, "z true"), (mag_rows, "mag i")]
    for col, (rows, xlabel) in enumerate(groups):
        x = np.asarray([row["bin_center"] for row in rows], dtype=float)
        n = np.asarray([row["n"] for row in rows], dtype=float)
        width = np.nanmedian(np.diff(x)) * 0.8 if len(x) > 1 else 0.1
        axes[0, col].bar(x, n, width=width, color="#8aa6c1")
        axes[0, col].set_ylabel("N test")
        axes[0, col].grid(True, alpha=0.25)
        _plot_metric(axes[1, col], rows, "bias", "bias_err", "bias")
        _plot_metric(axes[2, col], rows, "sigma_nmad", "sigma_nmad_err", "sigma_NMAD")
        _plot_metric(axes[3, col], rows, "outlier_rate", "outlier_rate_err", "outliers (%)")
        axes[3, col].set_xlabel(xlabel)
    axes[0, 0].set_title("Binning par redshift label")
    axes[0, 1].set_title("Binning par magnitude i")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def make_figure7_report(
    predictions_path: str,
    metadata_path: str,
    output_dir: str,
    z_bins: int = 20,
    mag_i_min: float = CONFIG.I_MIN,
    mag_i_max: float = CONFIG.I_MAX,
    mag_i_bins: int = 14,
    bootstrap: int = 100,
    seed: int = CONFIG.SEED,
) -> None:
    ensure_dir(output_dir)
    predictions = load_predictions(predictions_path)
    metadata = align_metadata_for_predictions(load_metadata(metadata_path), predictions)
    z_true = predictions["z_true"]
    z_pred = predictions["z_pred"]
    mag_i = metadata["mag_i"][: len(z_true)]

    z_edges = np.linspace(0.0, CONFIG.Z_MAX, z_bins + 1)
    mag_edges = np.linspace(mag_i_min, mag_i_max, mag_i_bins + 1)
    z_rows = aggregate_treyer_bins(z_true, z_edges, z_true, z_pred, n_bootstrap=bootstrap, seed=seed)
    mag_rows = aggregate_treyer_bins(mag_i, mag_edges, z_true, z_pred, n_bootstrap=bootstrap, seed=seed + 10_000)

    write_rows_csv(os.path.join(output_dir, "metrics_by_z_true.csv"), z_rows)
    write_rows_csv(os.path.join(output_dir, "metrics_by_mag_i.csv"), mag_rows)
    write_rows_csv(os.path.join(output_dir, "metrics_global.csv"), [_row_metrics(z_true, z_pred)])
    plot_figure7_like(z_rows, mag_rows, os.path.join(output_dir, "figure7_like_treyer.png"))

    if "train_mag_i" in predictions:
        support = magnitude_support_rows(
            mag_i,
            predictions["train_mag_i"],
            z_true,
            z_pred,
            mag_edges,
        )
        write_rows_csv(os.path.join(output_dir, "metrics_by_mag_support.csv"), support)

    logger.info("Analyse Figure 7-like sauvegardee: %s", output_dir)


def run(args: argparse.Namespace) -> None:
    make_figure7_report(
        predictions_path=args.predictions,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        z_bins=args.z_bins,
        mag_i_min=args.mag_i_min,
        mag_i_max=args.mag_i_max,
        mag_i_bins=args.mag_i_bins,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=CONFIG.exp_path("treyer_figure7"))
    parser.add_argument("--z_bins", type=int, default=20)
    parser.add_argument("--mag_i_min", type=float, default=CONFIG.I_MIN)
    parser.add_argument("--mag_i_max", type=float, default=CONFIG.I_MAX)
    parser.add_argument("--mag_i_bins", type=int, default=14)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    run(parser.parse_args())
