import argparse
import glob
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from config import CONFIG
from data_loader import build_metadata, export_metadata, get_dataset_and_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_hist_by_split(values: np.ndarray, split: np.ndarray, title: str, xlabel: str, output_path: str, bins: int = 50) -> None:
    plt.figure(figsize=(8, 5))
    for split_name in ["train", "val", "test"]:
        mask = (split == split_name) & np.isfinite(values)
        if np.any(mask):
            plt.hist(values[mask], bins=bins, histtype="step", density=True, linewidth=1.8, label=f"{split_name} (n={np.sum(mask)})")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Densite normalisee")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scatter_split(metadata: Dict[str, np.ndarray], output_path: str) -> None:
    plt.figure(figsize=(7, 7))
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    for split_name, color in colors.items():
        mask = metadata["split"] == split_name
        if np.any(mask):
            plt.scatter(metadata["ra"][mask], metadata["dec"][mask], s=2, alpha=0.35, label=split_name, color=color)
    plt.xlabel("RA (deg)")
    plt.ylabel("DEC (deg)")
    plt.title("Repartition spatiale train/val/test")
    plt.legend(markerscale=4)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_flag_distribution(metadata: Dict[str, np.ndarray], output_path: str) -> None:
    band_names = CONFIG.BAND_NAMES
    means = []
    for band in band_names:
        vals = metadata.get(f"flag_{band}")
        if vals is None:
            means.append(np.nan)
        else:
            finite = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite != 0)) if finite.size else np.nan)

    plt.figure(figsize=(8, 5))
    plt.bar(band_names, means)
    plt.ylim(0, 1)
    plt.xlabel("Bande")
    plt.ylabel("Fraction flag != 0")
    plt.title("Distribution des flags par bande apres selection")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def scan_raw_flag_distribution(data_path: str, max_files: int = None) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
    if max_files is not None:
        files = files[:max_files]

    counts = np.zeros(len(CONFIG.BAND_NAMES), dtype=np.int64)
    totals = np.zeros(len(CONFIG.BAND_NAMES), dtype=np.int64)
    files_with_flags = 0
    for path in files:
        try:
            with np.load(path) as raw:
                if "flag" not in raw.files:
                    continue
                flags = raw["flag"][:, CONFIG.CHANNELS]
                n_bands = min(flags.shape[1], len(CONFIG.BAND_NAMES))
                finite = np.isfinite(flags[:, :n_bands])
                counts[:n_bands] += np.sum((flags[:, :n_bands] != 0) & finite, axis=0)
                totals[:n_bands] += np.sum(finite, axis=0)
                files_with_flags += 1
        except Exception as exc:
            logger.warning("Impossible de lire les flags bruts de %s: %s", path, exc)

    for i, band in enumerate(CONFIG.BAND_NAMES):
        fraction = float(counts[i] / totals[i]) if totals[i] else float("nan")
        rows.append({"band": band, "n_flagged": int(counts[i]), "n_total": int(totals[i]), "fraction_flagged": fraction})
    logger.info("Audit flags bruts: %s fichiers avec flags sur %s fichiers inspectes.", files_with_flags, len(files))
    return rows


def plot_raw_flag_distribution(rows: List[Dict[str, float]], output_path: str) -> None:
    bands = [row["band"] for row in rows]
    values = [row["fraction_flagged"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.bar(bands, values)
    plt.ylim(0, 1)
    plt.xlabel("Bande")
    plt.ylabel("Fraction flag != 0")
    plt.title("Distribution des flags bruts avant selection")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_summary(metadata: Dict[str, np.ndarray], output_path: str) -> None:
    lines = ["metric,value"]
    lines.append(f"n_total,{len(metadata['z_true'])}")
    for split_name in ["train", "val", "test"]:
        lines.append(f"n_{split_name},{int(np.sum(metadata['split'] == split_name))}")
    for key in ["z_true", "mag_u", "mag_i", "mag_z", "ra", "dec"]:
        vals = metadata.get(key)
        if vals is None:
            continue
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lines.append(f"{key}_min,{float(np.min(vals))}")
            lines.append(f"{key}_median,{float(np.median(vals))}")
            lines.append(f"{key}_max,{float(np.max(vals))}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir or CONFIG.exp_path("dataset_report"))
    dataset, split_indices = get_dataset_and_splits(
        region=args.region,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
    )

    metadata_npz = args.metadata_output or CONFIG.exp_path(CONFIG.DATASET_METADATA)
    metadata_csv = os.path.splitext(metadata_npz)[0] + ".csv"
    metadata = export_metadata(dataset, metadata_npz, split_indices=split_indices, csv_path=metadata_csv)
    metadata = build_metadata(dataset, split_indices=split_indices)

    logger.info("Metadata exportee: %s et %s", metadata_npz, metadata_csv)
    write_summary(metadata, os.path.join(output_dir, "dataset_summary.csv"))
    plot_scatter_split(metadata, os.path.join(output_dir, "ra_dec_split.png"))
    plot_hist_by_split(metadata["z_true"], metadata["split"], "Distribution redshift par split", "z spec", os.path.join(output_dir, "z_distribution_by_split.png"))
    plot_flag_distribution(metadata, os.path.join(output_dir, "flags_by_band.png"))
    raw_flag_rows = scan_raw_flag_distribution(CONFIG.DATA_PATH, max_files=args.max_files)
    if raw_flag_rows:
        write_rows_csv(os.path.join(output_dir, "raw_flags_by_band.csv"), raw_flag_rows)
        plot_raw_flag_distribution(raw_flag_rows, os.path.join(output_dir, "raw_flags_by_band.png"))

    for band in CONFIG.BAND_NAMES:
        key = f"mag_{band}"
        if key in metadata and np.isfinite(metadata[key]).any():
            plot_hist_by_split(
                metadata[key],
                metadata["split"],
                f"Distribution magnitude {band}",
                f"mag {band}",
                os.path.join(output_dir, f"mag_{band}_distribution_by_split.png"),
            )

    for band in ["u", "i", "z"]:
        key = f"mag_{band}"
        if key in metadata and np.isfinite(metadata[key]).any():
            plot_hist_by_split(
                metadata[key],
                metadata["split"],
                f"Focus bande {band.upper()}",
                f"mag {band}",
                os.path.join(output_dir, f"focus_mag_{band}.png"),
            )
        else:
            logger.warning("Bande %s absente ou non finie dans les metadata.", band)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--metadata_output", type=str, default=None)
    run(parser.parse_args())
