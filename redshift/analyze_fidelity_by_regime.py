import argparse
import csv
import logging
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from config import CONFIG


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BANDS = ("u", "g", "r", "i", "z", "y")


def load_band_metrics(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Fichier de metriques vide: {path}")

    columns: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = []
        for row in rows:
            raw = row.get(key, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(np.nan)
        columns[key] = np.asarray(values, dtype=np.float64)
    return columns


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.nanquantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        edges = np.linspace(float(np.nanmin(finite)), float(np.nanmax(finite)) + 1e-6, n_bins + 1)
    return edges


def _median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def aggregate_1d(
    axis_name: str,
    axis_values: np.ndarray,
    edges: np.ndarray,
    metrics: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for b in range(len(edges) - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (axis_values >= lo) & (axis_values <= hi if b == len(edges) - 2 else axis_values < hi)
        row: Dict[str, object] = {
            "axis": axis_name,
            "bin": b,
            "bin_min": lo,
            "bin_max": hi,
            "bin_center": 0.5 * (lo + hi),
            "n": int(np.sum(mask)),
        }
        for band in BANDS:
            l1 = metrics.get(f"relative_l1_{band}")
            corr = metrics.get(f"corr_{band}")
            if l1 is not None:
                row[f"median_relative_l1_{band}"] = _median(l1[mask])
            if corr is not None:
                row[f"median_corr_{band}"] = _median(corr[mask])
        rows.append(row)
    return rows


def grid_median(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_count: int,
) -> np.ndarray:
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    for row in range(ny):
        for col in range(nx):
            cell = valid & (x_bin == col) & (y_bin == row)
            if int(np.sum(cell)) >= min_count:
                grid[row, col] = float(np.median(values[cell]))
    return grid


def plot_curves(output_path: str, z_rows: Sequence[Dict[str, object]], mag_rows: Sequence[Dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for band in BANDS:
        key = f"median_relative_l1_{band}"
        centers = np.array([r["bin_center"] for r in z_rows], dtype=np.float64)
        values = np.array([r.get(key, np.nan) for r in z_rows], dtype=np.float64)
        axes[0].plot(centers, values, marker="o", label=band)

        centers = np.array([r["bin_center"] for r in mag_rows], dtype=np.float64)
        values = np.array([r.get(key, np.nan) for r in mag_rows], dtype=np.float64)
        axes[1].plot(centers, values, marker="o", label=band)

    axes[0].set_xlabel("z spectroscopique")
    axes[1].set_xlabel("mag_i")
    for ax in axes:
        ax.set_ylabel("L1 relative mediane")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
    fig.suptitle("Fidelite de reconstruction par regime", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_heatmap(output_path: str, grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, band: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="inferno",
    )
    ax.set_xlabel("z spectroscopique")
    ax.set_ylabel("mag_i")
    ax.invert_yaxis()
    ax.set_title(f"L1 relative mediane, bande {band}")
    plt.colorbar(image, ax=ax, label="L1 relative mediane")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    metrics = load_band_metrics(args.band_metrics)
    for required in ("z", "mag_i"):
        if required not in metrics:
            raise KeyError(f"Colonne manquante dans {args.band_metrics}: {required}")

    z = metrics["z"]
    mag_i = metrics["mag_i"]
    z_edges = quantile_edges(z, args.bins)
    mag_edges = quantile_edges(mag_i, args.bins)

    z_rows = aggregate_1d("z", z, z_edges, metrics)
    mag_rows = aggregate_1d("mag_i", mag_i, mag_edges, metrics)
    write_rows_csv(os.path.join(output_dir, "fidelity_by_z.csv"), z_rows)
    write_rows_csv(os.path.join(output_dir, "fidelity_by_mag_i.csv"), mag_rows)

    plot_curves(os.path.join(output_dir, "fidelity_curves.png"), z_rows, mag_rows)

    for band in args.heatmap_bands.split(","):
        band = band.strip()
        key = f"relative_l1_{band}"
        if key not in metrics:
            logger.warning("Bande absente des metriques, heatmap ignoree: %s", band)
            continue
        grid = grid_median(z, mag_i, metrics[key], z_edges, mag_edges, args.min_cell_count)
        plot_heatmap(os.path.join(output_dir, f"fidelity_heatmap_{band}.png"), grid, z_edges, mag_edges, band)

    worst = max(z_rows, key=lambda r: r.get("median_relative_l1_g", 0.0) or 0.0)
    logger.info(
        "Regime le plus degrade en bande g: z dans [%.3f, %.3f], L1=%.3f sur n=%d",
        worst["bin_min"], worst["bin_max"], worst.get("median_relative_l1_g", float("nan")), worst["n"],
    )
    logger.info("Analyse de fidelite par regime sauvegardee dans %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fidelite de reconstruction i2i en fonction du redshift et de la magnitude."
    )
    parser.add_argument("--band_metrics", required=True, help="visual_band_metrics.csv produit par visual_band_inspection.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--heatmap_bands", type=str, default="g,r,i")
    parser.add_argument("--min_cell_count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    run(parser.parse_args())
