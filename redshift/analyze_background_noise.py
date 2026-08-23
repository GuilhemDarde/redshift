import argparse
import logging
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from config import CONFIG
from data_loader import get_dataset_and_splits
from utils import set_global_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def outer_annulus_mask(height: int, width: int, inner_fraction: float) -> np.ndarray:
    '''
    actions : Construit un masque des pixels exterieurs, loin de l'objet central.
    inputs : height (int), width (int), inner_fraction (float)
    appels : np.meshgrid, np.sqrt
    outputs : np.ndarray
    '''
    cy = 0.5 * (height - 1)
    cx = 0.5 * (width - 1)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return radius >= inner_fraction * 0.5 * min(height, width)


def robust_sigma(values: np.ndarray, axis: int = -1) -> np.ndarray:
    '''
    actions : Estime un ecart-type robuste par MAD, insensible aux objets voisins brillants.
    inputs : values (np.ndarray), axis (int)
    appels : np.median
    outputs : np.ndarray
    '''
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis)
    return 1.4826 * mad


def background_statistics(images: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    '''
    actions : Calcule le niveau et la dispersion du fond par image et par bande.
    inputs : images (np.ndarray), mask (np.ndarray)
    appels : robust_sigma, np.median
    outputs : Dict[str, np.ndarray]
    '''
    if images.ndim != 4:
        raise ValueError("images doit avoir la forme (n, bandes, hauteur, largeur).")
    flat = images[:, :, mask]
    return {
        "sigma": robust_sigma(flat, axis=-1),
        "level": np.median(flat, axis=-1),
    }


def summarize(
    band_names: List[str],
    source_stats: Dict[str, np.ndarray],
    aug_stats: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for b, band in enumerate(band_names):
        src_sigma = source_stats["sigma"][:, b]
        aug_sigma = aug_stats["sigma"][:, b]
        finite = np.isfinite(src_sigma) & np.isfinite(aug_sigma) & (src_sigma > 0)
        ratio = np.full(src_sigma.shape, np.nan, dtype=np.float64)
        ratio[finite] = aug_sigma[finite] / src_sigma[finite]
        rows.append({
            "band": band,
            "n": int(np.sum(finite)),
            "median_sigma_source": float(np.median(src_sigma[finite])) if np.any(finite) else float("nan"),
            "median_sigma_augmented": float(np.median(aug_sigma[finite])) if np.any(finite) else float("nan"),
            "median_sigma_ratio": float(np.median(ratio[finite])) if np.any(finite) else float("nan"),
            "q16_sigma_ratio": float(np.nanquantile(ratio[finite], 0.16)) if np.any(finite) else float("nan"),
            "q84_sigma_ratio": float(np.nanquantile(ratio[finite], 0.84)) if np.any(finite) else float("nan"),
            "median_level_source": float(np.median(source_stats["level"][:, b])),
            "median_level_augmented": float(np.median(aug_stats["level"][:, b])),
        })
    return rows


def plot_summary(output_path: str, rows: List[Dict[str, object]]) -> None:
    bands = [str(r["band"]) for r in rows]
    ratio = np.array([r["median_sigma_ratio"] for r in rows], dtype=np.float64)
    low = ratio - np.array([r["q16_sigma_ratio"] for r in rows], dtype=np.float64)
    high = np.array([r["q84_sigma_ratio"] for r in rows], dtype=np.float64) - ratio

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(np.arange(len(bands)), ratio, yerr=[low, high], fmt="o", capsize=4)
    ax.axhline(1.0, color="grey", linewidth=1.0)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_ylabel("sigma(fond) augmentee / source")
    ax.set_title("Conservation du bruit de fond par bande")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_distributions(output_path: str, band_names: List[str], source_stats, aug_stats) -> None:
    n = len(band_names)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 7))
    for b, ax in zip(range(n), axes.ravel()):
        src = source_stats["sigma"][:, b]
        aug = aug_stats["sigma"][:, b]
        both = np.concatenate([src[np.isfinite(src)], aug[np.isfinite(aug)]])
        if both.size == 0:
            continue
        lo, hi = np.nanquantile(both, [0.01, 0.99])
        bins = np.linspace(lo, hi, 60)
        ax.hist(src, bins=bins, histtype="step", label="source")
        ax.hist(aug, bins=bins, histtype="step", label="augmentee")
        ax.set_title(f"bande {band_names[b]}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Distribution de sigma(fond), source vs augmentee", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    try:
        candidates = np.load(args.augmentations, allow_pickle=False)
    except ValueError as exc:
        # Certains artefacts intermediaires contiennent des tableaux objets.
        # Ces fichiers sont produits localement par le pipeline, donc depickler est sans risque ici.
        logger.warning("Lecture stricte impossible (%s), nouvelle tentative avec allow_pickle=True.", exc)
        candidates = np.load(args.augmentations, allow_pickle=True)
    x = candidates["x"]
    source_index = candidates["source_index"] if "source_index" in candidates.files else None
    if source_index is None:
        raise KeyError("Le fichier d'augmentations doit contenir source_index pour comparer au fond source.")

    dataset, _ = get_dataset_and_splits(
        region=args.region,
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
        split_strategy=args.split_strategy,
    )

    valid = (source_index >= 0) & (source_index < len(dataset.data["x"]))
    indices = np.where(valid)[0]
    if indices.size == 0:
        raise RuntimeError("Aucun candidat avec source_index valide.")
    if args.max_samples is not None and indices.size > args.max_samples:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(indices, size=args.max_samples, replace=False))

    aug_images = np.asarray(x[indices], dtype=np.float64)
    source_images = np.asarray(dataset.data["x"][source_index[indices]], dtype=np.float64)
    mask = outer_annulus_mask(aug_images.shape[2], aug_images.shape[3], args.inner_fraction)
    logger.info("Fond mesure sur %d pixels par image et par bande (%d candidats).", int(mask.sum()), len(indices))

    band_names = list(CONFIG.BAND_NAMES)[: aug_images.shape[1]]
    source_stats = background_statistics(source_images, mask)
    aug_stats = background_statistics(aug_images, mask)
    rows = summarize(band_names, source_stats, aug_stats)

    write_rows_csv(os.path.join(output_dir, "background_noise_summary.csv"), rows)
    plot_summary(os.path.join(output_dir, "background_noise_ratio.png"), rows)
    plot_distributions(os.path.join(output_dir, "background_noise_distributions.png"), band_names, source_stats, aug_stats)

    print(f"{'bande':>6} {'sigma source':>14} {'sigma augm.':>14} {'ratio':>8}")
    print("-" * 46)
    for row in rows:
        print(
            f"{str(row['band']):>6} {row['median_sigma_source']:>14.6g} "
            f"{row['median_sigma_augmented']:>14.6g} {row['median_sigma_ratio']:>8.4f}"
        )
    print(
        "\nLecture: un ratio nettement inferieur a 1 signifie que l'augmentation lisse le bruit. "
        "Un CNN peut alors distinguer image reelle et image augmentee, et exploiter ce raccourci."
    )
    logger.info("Analyse du bruit de fond sauvegardee dans %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare les statistiques de bruit de fond entre images sources et augmentations."
    )
    parser.add_argument("--augmentations", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--inner_fraction", type=float, default=0.75, help="Rayon interieur de la couronne, en fraction du demi-cote.")
    parser.add_argument("--max_samples", type=int, default=4000)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="all")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--split_strategy", choices=["spatial", "marie_regular", "marie_strict"], default="spatial")
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    run(parser.parse_args())
