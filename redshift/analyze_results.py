import argparse
import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import (
    aggregate_by_bins,
    bin_edges_from_range,
    compute_regression_metrics,
    ensure_dir,
    load_metadata,
    residuals_normalized,
    write_rows_csv,
)
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_results(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    if "z_true" not in data or "z_pred" not in data:
        raise KeyError("Le fichier resultats doit contenir z_true et z_pred.")
    result = {key: data[key] for key in data.files}
    if "z_sigma" not in result:
        result["z_sigma"] = np.full_like(result["z_true"], np.nan, dtype=np.float64)
    return result


def align_metadata(metadata: Optional[Dict[str, np.ndarray]], n_results: int) -> Optional[Dict[str, np.ndarray]]:
    if metadata is None:
        return None
    if "split" in metadata and np.sum(metadata["split"] == "test") == n_results:
        mask = metadata["split"] == "test"
        return {k: v[mask] if np.asarray(v).ndim == 1 and len(v) == len(mask) else v for k, v in metadata.items()}
    if len(metadata["z_true"]) == n_results:
        return metadata
    n = min(len(metadata["z_true"]), n_results)
    logger.warning("Taille metadata (%s) != resultats (%s). Troncature a %s.", len(metadata["z_true"]), n_results, n)
    return {k: v[:n] if np.asarray(v).ndim == 1 and len(v) >= n else v for k, v in metadata.items()}


def plot_heatmap(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, output_path: str, bins: int = 80) -> None:
    plt.figure(figsize=(7, 6))
    plt.hist2d(x, y, bins=bins, cmap="magma")
    plt.colorbar(label="N")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_curve(rows, x_key: str, y_key: str, xlabel: str, ylabel: str, title: str, output_path: str) -> None:
    x = np.array([r[x_key] for r in rows], dtype=float)
    y = np.array([r[y_key] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    for xi, yi, ni in zip(x, y, n):
        if np.isfinite(yi):
            plt.text(xi, yi, f"{int(ni)}", fontsize=7, ha="center", va="bottom")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_radec_error(metadata: Dict[str, np.ndarray], dz: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(7, 7))
    sc = plt.scatter(metadata["ra"], metadata["dec"], c=np.abs(dz), s=3, cmap="viridis", vmin=0, vmax=np.nanpercentile(np.abs(dz), 95))
    plt.colorbar(sc, label="|dz| / (1+z)")
    plt.xlabel("RA (deg)")
    plt.ylabel("DEC (deg)")
    plt.title("Erreur spatiale RA/DEC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir or CONFIG.exp_path("results_report"))
    results = load_results(args.results)
    metadata = None
    if args.metadata or CONFIG.METADATA_PATH or os.path.exists(CONFIG.exp_path(CONFIG.DATASET_METADATA)):
        metadata = align_metadata(load_metadata(args.metadata), len(results["z_true"]))

    n = len(results["z_true"])
    if metadata is not None:
        n = min(n, len(metadata["z_true"]))
    z_true = results["z_true"][:n]
    z_pred = results["z_pred"][:n]
    z_sigma = results["z_sigma"][:n]
    dz = residuals_normalized(z_true, z_pred)

    metrics = compute_regression_metrics(z_true, z_pred, z_sigma)
    write_rows_csv(os.path.join(output_dir, "metrics_global.csv"), [metrics])

    plot_heatmap(z_true, z_pred, "z true", "z pred", "Densite z_true vs z_pred", os.path.join(output_dir, "heatmap_ztrue_zpred.png"))
    plot_heatmap(z_pred, dz, "z pred", "dz/(1+z)", "Residus vs z_pred", os.path.join(output_dir, "heatmap_residual_zpred.png"))

    z_edges = bin_edges_from_range(z_pred, args.z_bins)
    z_rows = aggregate_by_bins(z_pred, z_edges, z_true, z_pred, z_sigma)
    write_rows_csv(os.path.join(output_dir, "metrics_by_z_pred.csv"), z_rows)
    plot_curve(z_rows, "bin_center", "sigma_nmad", "z_pred", "sigma_NMAD", "Sigma_NMAD par bin de z_pred", os.path.join(output_dir, "sigma_by_z_pred.png"))

    if metadata is not None:
        mag_i = metadata["mag_i"][:n]
        plot_heatmap(mag_i, dz, "mag I", "dz/(1+z)", "Residus vs bande I", os.path.join(output_dir, "heatmap_residual_mag_i.png"))
        i_edges = np.linspace(args.i_min, args.i_max, args.i_bins + 1)
        i_mask = (mag_i >= args.i_min) & (mag_i <= args.i_max)
        i_rows = aggregate_by_bins(mag_i[i_mask], i_edges, z_true[i_mask], z_pred[i_mask], z_sigma[i_mask])
        write_rows_csv(os.path.join(output_dir, "metrics_by_mag_i.csv"), i_rows)
        plot_curve(i_rows, "bin_center", "sigma_nmad", "mag I", "sigma_NMAD", "Sigma_NMAD par bin de bande I", os.path.join(output_dir, "sigma_by_mag_i.png"))
        plot_radec_error(metadata, dz, os.path.join(output_dir, "radec_error_map.png"))
    else:
        logger.warning("Metadata non disponibles: analyses bande I et RA/DEC ignorees.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=CONFIG.exp_path(CONFIG.SOTA_RESULTS))
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--z_bins", type=int, default=30)
    parser.add_argument("--i_min", type=float, default=CONFIG.I_ANALYSIS_RANGE[0])
    parser.add_argument("--i_max", type=float, default=CONFIG.I_ANALYSIS_RANGE[1])
    parser.add_argument("--i_bins", type=int, default=CONFIG.I_ANALYSIS_BINS)
    run(parser.parse_args())
