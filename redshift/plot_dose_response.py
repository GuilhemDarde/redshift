import argparse
import csv
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import ensure_dir, write_rows_csv


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

METRICS = ("sigma_nmad", "rmse", "outlier_rate")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dose_from_run(run_dir: str, n_available: Optional[int]) -> float:
    """Dose effective: max_synthetic, ou le pool complet quand l'option est absente."""
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        value = config.get("max_synthetic")
        if value is not None:
            return float(value)
    if n_available is not None:
        return float(n_available)
    basename = os.path.basename(run_dir.rstrip("/"))
    digits = "".join(ch for ch in basename if ch.isdigit())
    if not digits:
        raise ValueError(f"Dose indeterminable pour {run_dir}: ni max_synthetic ni chiffres dans le nom.")
    return float(digits)


def collect_doses(run_dirs: Sequence[str], ablation: str, n_available: Optional[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, f"metrics_subsets_{ablation}.csv")
        if not os.path.exists(path):
            logger.warning("Metriques absentes, run ignore: %s", path)
            continue
        dose = dose_from_run(run_dir, n_available)
        for row in read_csv_rows(path):
            entry: Dict[str, object] = {
                "run": os.path.basename(run_dir.rstrip("/")),
                "dose": dose,
                "subset": row["subset"],
                "n": int(float(row["n"])),
            }
            for metric in METRICS:
                value = row.get(metric, "")
                entry[metric] = float(value) if value not in ("", "nan") else float("nan")
            rows.append(entry)
    rows.sort(key=lambda r: (str(r["subset"]), float(r["dose"])))
    return rows


def baseline_reference(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Moyenne et ecart-type inter-graines de la baseline, par sous-groupe."""
    if not path or not os.path.exists(path):
        return {}
    reference: Dict[str, Dict[str, float]] = {}
    for row in read_csv_rows(path):
        subset = row["subset"]
        stats: Dict[str, float] = {}
        for metric in METRICS:
            for suffix in ("mean", "std"):
                key = f"{metric}_{suffix}"
                if row.get(key) not in (None, "", "nan"):
                    stats[key] = float(row[key])
        reference[subset] = stats
    return reference


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty_like(values, dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    rx, ry = rank(x[mask]), rank(y[mask])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_dose_response(
    output_path: str,
    series: Sequence[Tuple[str, Sequence[Dict[str, object]]]],
    reference: Dict[str, Dict[str, float]],
    subsets: Sequence[str],
) -> None:
    colors = ["tab:blue", "tab:green", "tab:purple"]
    fig, axes = plt.subplots(len(METRICS), len(subsets), figsize=(5.2 * len(subsets), 3.6 * len(METRICS)), squeeze=False)
    for col, subset in enumerate(subsets):
        first_rows = [r for r in series[0][1] if r["subset"] == subset]
        for line, metric in enumerate(METRICS):
            ax = axes[line][col]
            for s, (label, rows) in enumerate(series):
                subset_rows = [r for r in rows if r["subset"] == subset]
                doses = np.array([r["dose"] for r in subset_rows], dtype=np.float64)
                values = np.array([r[metric] for r in subset_rows], dtype=np.float64)
                ax.plot(doses, values, marker="o", color=colors[s % len(colors)], label=label)

            subset_rows = first_rows
            stats = reference.get(subset, {})
            mean = stats.get(f"{metric}_mean")
            std = stats.get(f"{metric}_std")
            if mean is not None:
                ax.axhline(mean, color="tab:red", linewidth=1.2, label="baseline (moyenne graines)")
                if std is not None:
                    ax.axhspan(mean - 2 * std, mean + 2 * std, color="tab:red", alpha=0.15, label="baseline +/- 2 sigma")
            if line == 0:
                ax.set_title(f"{subset}\n(n={subset_rows[0]['n'] if subset_rows else 0})", fontsize=10)
            if col == 0:
                ax.set_ylabel(metric)
            if line == len(METRICS) - 1:
                ax.set_xlabel("images synthetiques ajoutees")
            ax.grid(True, alpha=0.3)
            if line == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Courbe dose-reponse de l'augmentation i2i", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    run_dirs = [d for d in args.run_dirs if os.path.isdir(d)]
    if len(run_dirs) < 3:
        raise ValueError("Il faut au moins trois doses pour lire une tendance.")

    labels = [s.strip() for s in args.labels.split(",")]
    series: List[Tuple[str, List[Dict[str, object]]]] = [
        (labels[0] if labels else "serie 1", collect_doses(run_dirs, args.ablation, args.n_available))
    ]
    if args.compare_run_dirs:
        compare_dirs = [d for d in args.compare_run_dirs if os.path.isdir(d)]
        label = labels[1] if len(labels) > 1 else "serie 2"
        series.append((label, collect_doses(compare_dirs, args.ablation, args.n_available)))

    reference = baseline_reference(args.baseline_variance)
    for label, rows in series:
        suffix = "".join(ch if ch.isalnum() else "_" for ch in label)
        write_rows_csv(os.path.join(output_dir, f"dose_response_{suffix}.csv"), rows)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    plot_dose_response(os.path.join(output_dir, "dose_response.png"), series, reference, subsets)

    trend_rows: List[Dict[str, object]] = []
    print(f"{'serie':<18} {'subset':<36} {'metrique':<13} {'spearman':>9} {'min':>11} {'max':>11} {'2 sigma':>10}")
    print("-" * 114)
    for label, rows in series:
        for subset in subsets:
            subset_rows = [r for r in rows if r["subset"] == subset]
            if not subset_rows:
                continue
            doses = np.array([r["dose"] for r in subset_rows], dtype=np.float64)
            for metric in METRICS:
                values = np.array([r[metric] for r in subset_rows], dtype=np.float64)
                rho = spearman(doses, values)
                std = reference.get(subset, {}).get(f"{metric}_std")
                two_sigma = 2.0 * std if std is not None else float("nan")
                spread = float(np.nanmax(values) - np.nanmin(values)) if values.size else float("nan")
                trend_rows.append({
                    "series": label,
                    "subset": subset,
                    "metric": metric,
                    "spearman_dose_vs_metric": rho,
                    "min": float(np.nanmin(values)) if values.size else float("nan"),
                    "max": float(np.nanmax(values)) if values.size else float("nan"),
                    "spread": spread,
                    "baseline_two_sigma": two_sigma,
                    "spread_exceeds_two_sigma": bool(np.isfinite(two_sigma) and spread > two_sigma),
                })
                print(
                    f"{label:<18} {subset:<36} {metric:<13} {rho:>9.3f} "
                    f"{np.nanmin(values):>11.6f} {np.nanmax(values):>11.6f} {two_sigma:>10.6f}"
                )
        print()
    write_rows_csv(os.path.join(output_dir, "dose_response_trends.csv"), trend_rows)

    print(
        "\nLecture: un effet reel est monotone en dose (|spearman| proche de 1) ET d'amplitude "
        "superieure a 2 sigma. Une variation non monotone sous 2 sigma est du bruit."
    )
    logger.info("Courbe dose-reponse sauvegardee dans %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble la courbe dose-reponse des augmentations i2i.")
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ablation", type=str, default="i2i")
    parser.add_argument("--compare_run_dirs", nargs="+", default=None, help="Seconde serie de doses, superposee a la premiere.")
    parser.add_argument("--labels", type=str, default="sans correction,bruit restaure")
    parser.add_argument("--baseline_variance", type=str, default=None, help="seed_variance_real.csv pour la bande +/- 2 sigma.")
    parser.add_argument("--n_available", type=int, default=None, help="Taille du pool, utilisee quand max_synthetic est absent.")
    parser.add_argument(
        "--subsets",
        type=str,
        default="global,low_photometric_support,faint_and_low_photometric_support",
    )
    run(parser.parse_args())
