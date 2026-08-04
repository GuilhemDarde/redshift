import argparse
import csv
import logging
import os
import statistics
from typing import Dict, List, Sequence

from analysis_utils import ensure_dir, write_rows_csv


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


METRICS = ("sigma_nmad", "rmse", "outlier_rate")


def read_subset_metrics(run_dir: str, ablation: str) -> List[Dict[str, str]]:
    path = os.path.join(run_dir, f"metrics_subsets_{ablation}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de metriques introuvable: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect(run_dirs: Sequence[str], ablation: str) -> Dict[str, Dict[str, List[float]]]:
    """Regroupe les metriques par sous-groupe, une valeur par run."""
    collected: Dict[str, Dict[str, List[float]]] = {}
    for run_dir in run_dirs:
        for row in read_subset_metrics(run_dir, ablation):
            subset = row["subset"]
            bucket = collected.setdefault(subset, {metric: [] for metric in METRICS})
            for metric in METRICS:
                value = row.get(metric, "")
                if value in ("", "nan"):
                    continue
                bucket[metric].append(float(value))
    return collected


def summarize(collected: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for subset, metrics in collected.items():
        row: Dict[str, object] = {"subset": subset}
        for metric, values in metrics.items():
            if not values:
                continue
            mean = statistics.fmean(values)
            # Ecart-type d'echantillon: on estime la dispersion inter-graines,
            # pas celle d'une population complete.
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_n_seeds"] = len(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
            row[f"{metric}_spread"] = max(values) - min(values)
            row[f"{metric}_cv_pct"] = 100.0 * std / mean if mean else float("nan")
            # Seuil indicatif: un ecart inferieur a 2 sigma inter-graines
            # ne peut pas etre presente comme un effet.
            row[f"{metric}_detectable_delta"] = 2.0 * std
        rows.append(row)
    rows.sort(key=lambda r: str(r["subset"]))
    return rows


def print_table(rows: Sequence[Dict[str, object]]) -> None:
    for metric in METRICS:
        print(f"\n=== {metric} ===")
        header = f"{'subset':<40} {'seeds':>5} {'mean':>12} {'std':>12} {'spread':>12} {'CV %':>8} {'2 sigma':>12}"
        print(header)
        print("-" * len(header))
        for row in rows:
            if f"{metric}_mean" not in row:
                continue
            print(
                f"{str(row['subset']):<40} "
                f"{row[f'{metric}_n_seeds']:>5} "
                f"{row[f'{metric}_mean']:>12.6f} "
                f"{row[f'{metric}_std']:>12.6f} "
                f"{row[f'{metric}_spread']:>12.6f} "
                f"{row[f'{metric}_cv_pct']:>8.2f} "
                f"{row[f'{metric}_detectable_delta']:>12.6f}"
            )


def run(args: argparse.Namespace) -> None:
    run_dirs = [d for d in args.run_dirs if os.path.isdir(d)]
    missing = sorted(set(args.run_dirs) - set(run_dirs))
    if missing:
        logger.warning("Repertoires ignores car introuvables: %s", missing)
    if len(run_dirs) < 2:
        raise ValueError("Il faut au moins deux runs pour estimer une variance inter-graines.")

    collected = collect(run_dirs, args.ablation)
    rows = summarize(collected)

    output_dir = ensure_dir(args.output_dir)
    output_path = os.path.join(output_dir, f"seed_variance_{args.ablation}.csv")
    write_rows_csv(output_path, rows)
    print_table(rows)

    print(
        "\nLecture: un ecart entre deux variantes inferieur a la colonne '2 sigma' "
        "n'est pas distinguable du bruit d'initialisation et ne doit pas etre "
        "presente comme un gain."
    )
    logger.info("Variance inter-graines sauvegardee dans %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agrege les metriques de plusieurs runs ne differant que par la graine."
    )
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--ablation", type=str, default="real")
    parser.add_argument("--output_dir", required=True)
    run(parser.parse_args())
