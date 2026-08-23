import argparse
import logging
import os
from typing import Dict, List, Tuple

import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from analyze_background_noise import outer_annulus_mask, robust_sigma
from config import CONFIG
from data_loader import get_dataset_and_splits
from utils import set_global_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def indices_subsample(n: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples is None or n <= max_samples:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def load_candidates(path: str):
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as exc:
        logger.warning("Lecture stricte impossible (%s), nouvelle tentative avec allow_pickle=True.", exc)
        return np.load(path, allow_pickle=True)


def background_sigma(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''
    actions : Estime sigma(fond) par image et par bande dans la couronne exterieure.
    inputs : images (np.ndarray), mask (np.ndarray)
    appels : robust_sigma
    outputs : np.ndarray de forme (n, bandes)
    '''
    return robust_sigma(np.asarray(images, dtype=np.float64)[:, :, mask], axis=-1)


def corner_patches(images: np.ndarray, size: int) -> np.ndarray:
    '''
    actions : Extrait quatre coins de fond, rectangulaires, pour mesurer l'autocorrelation.
    inputs : images (np.ndarray), size (int)
    appels : np.concatenate
    outputs : np.ndarray de forme (n, bandes, 4, size, size)
    '''
    return np.stack(
        [
            images[:, :, :size, :size],
            images[:, :, :size, -size:],
            images[:, :, -size:, :size],
            images[:, :, -size:, -size:],
        ],
        axis=2,
    )


def lag1_autocorrelation(images: np.ndarray, size: int) -> np.ndarray:
    '''
    actions : Mesure la correlation pixel-a-pixel horizontale du fond, par bande.
    inputs : images (np.ndarray), size (int)
    appels : corner_patches, np.mean, np.std
    outputs : np.ndarray de forme (bandes,)
    '''
    patches = np.asarray(corner_patches(images, size), dtype=np.float64)
    left = patches[..., :, :-1]
    right = patches[..., :, 1:]
    n_bands = patches.shape[1]
    result = np.full(n_bands, np.nan, dtype=np.float64)
    for b in range(n_bands):
        a = left[:, b].ravel()
        c = right[:, b].ravel()
        finite = np.isfinite(a) & np.isfinite(c)
        if np.sum(finite) < 2:
            continue
        a, c = a[finite], c[finite]
        sa, sc = np.std(a), np.std(c)
        if sa <= 0 or sc <= 0:
            continue
        result[b] = float(np.mean((a - np.mean(a)) * (c - np.mean(c))) / (sa * sc))
    return result


def missing_sigma(source_sigma: np.ndarray, aug_sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    actions : Calcule l'ecart-type du bruit a reinjecter, en quadrature, borne a zero.
    inputs : source_sigma (np.ndarray), aug_sigma (np.ndarray)
    appels : np.clip, np.sqrt
    outputs : Tuple[np.ndarray, np.ndarray]
    '''
    variance = source_sigma ** 2 - aug_sigma ** 2
    already_noisy = variance <= 0.0
    return np.sqrt(np.clip(variance, 0.0, None)), already_noisy


def _median_ratio_squared(numerator: np.ndarray, denominator: np.ndarray, band: int) -> float:
    finite = np.isfinite(numerator[:, band]) & np.isfinite(denominator[:, band]) & (denominator[:, band] > 0)
    if not np.any(finite):
        return float("nan")
    return float(np.median(numerator[finite, band] / denominator[finite, band]) ** 2)


def calibrate_scales(
    aug_images: np.ndarray,
    src_sigma: np.ndarray,
    aug_sigma: np.ndarray,
    sigma_add: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    target_ratio: float,
    iterations: int,
) -> np.ndarray:
    '''
    actions : Ajuste un facteur par bande pour que sigma(fond) restaure atteigne la cible.
    inputs : aug_images, src_sigma, aug_sigma, sigma_add, mask, rng, target_ratio (float), iterations (int)
    appels : background_sigma, _median_ratio_squared
    outputs : np.ndarray de forme (bandes,)

    L'addition en quadrature n'est pas exacte ici: sigma est estime par MAD sur un fond
    spatialement correle, ce qui sous-estime la composante lissee et fait depasser la cible.
    On resout donc empiriquement scale a partir d'une mesure, sur un sous-echantillon.
    '''
    n_bands = aug_images.shape[1]
    scales = np.ones(n_bands, dtype=np.float64)
    target = float(target_ratio) ** 2

    for iteration in range(max(1, iterations)):
        noise = rng.standard_normal(size=aug_images.shape) * sigma_add[:, :, None, None] * scales[None, :, None, None]
        measured = background_sigma(aug_images + noise, mask)
        for band in range(n_bands):
            a = _median_ratio_squared(aug_sigma, src_sigma, band)
            m = _median_ratio_squared(measured, src_sigma, band)
            if not np.isfinite(a) or not np.isfinite(m) or m <= a or target <= a:
                continue
            scales[band] *= float(np.sqrt((target - a) / (m - a)))
        logger.info(
            "Calibration iteration %d: facteurs = %s",
            iteration + 1, np.array2string(scales, precision=4, floatmode="fixed"),
        )
    return scales


def summarize(
    band_names: List[str],
    src: np.ndarray,
    aug: np.ndarray,
    fixed: np.ndarray,
    already_noisy: np.ndarray,
    autocorr: Dict[str, np.ndarray],
    scales: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for b, band in enumerate(band_names):
        finite = np.isfinite(src[:, b]) & (src[:, b] > 0)
        rows.append({
            "band": band,
            "n": int(np.sum(finite)),
            "median_sigma_source": float(np.median(src[finite, b])),
            "median_sigma_before": float(np.median(aug[finite, b])),
            "median_sigma_after": float(np.median(fixed[finite, b])),
            "ratio_before": float(np.median(aug[finite, b] / src[finite, b])),
            "ratio_after": float(np.median(fixed[finite, b] / src[finite, b])),
            "fraction_already_noisy": float(np.mean(already_noisy[finite, b])),
            "calibrated_scale": float(scales[b]),
            "autocorr_source": float(autocorr["source"][b]),
            "autocorr_before": float(autocorr["before"][b]),
            "autocorr_after": float(autocorr["after"][b]),
        })
    return rows


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    rng = np.random.default_rng(args.seed)

    candidates = load_candidates(args.augmentations)
    payload = {key: candidates[key] for key in candidates.files}
    if "source_index" not in payload:
        raise KeyError("Le fichier d'augmentations doit contenir source_index.")

    x = np.asarray(payload["x"])
    source_index = np.asarray(payload["source_index"], dtype=np.int64)

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
    if not np.all(valid):
        logger.warning("%d augmentations sans source valide seront laissees inchangees.", int(np.sum(~valid)))
    indices = np.where(valid)[0]
    if indices.size == 0:
        raise RuntimeError("Aucune augmentation avec source_index valide.")

    source_images = np.asarray(dataset.data["x"][source_index[indices]], dtype=np.float64)
    aug_images = np.asarray(x[indices], dtype=np.float64)
    mask = outer_annulus_mask(aug_images.shape[2], aug_images.shape[3], args.inner_fraction)
    band_names = list(CONFIG.BAND_NAMES)[: aug_images.shape[1]]

    src_sigma = background_sigma(source_images, mask)
    aug_sigma = background_sigma(aug_images, mask)
    sigma_add, already_noisy = missing_sigma(src_sigma, aug_sigma)

    logger.info(
        "Bruit a reinjecter calcule sur %d augmentations. Fraction deja assez bruitee: %.3f",
        indices.size, float(np.mean(already_noisy)),
    )

    if args.calibrate_iters > 0:
        sample = indices_subsample(aug_images.shape[0], args.calibrate_samples, rng)
        scales = calibrate_scales(
            aug_images[sample],
            src_sigma[sample],
            aug_sigma[sample],
            sigma_add[sample],
            mask,
            rng,
            args.target_ratio,
            args.calibrate_iters,
        )
    else:
        scales = np.full(aug_images.shape[1], float(args.scale), dtype=np.float64)

    # Bruit blanc: restaure la variance, pas la structure spatiale. Le diagnostic
    # d'autocorrelation ci-dessous mesure precisement l'ecart restant.
    noise = rng.standard_normal(size=aug_images.shape) * sigma_add[:, :, None, None] * scales[None, :, None, None]
    fixed_images = aug_images + noise
    fixed_sigma = background_sigma(fixed_images, mask)

    autocorr = {
        "source": lag1_autocorrelation(source_images, args.corner_size),
        "before": lag1_autocorrelation(aug_images, args.corner_size),
        "after": lag1_autocorrelation(fixed_images, args.corner_size),
    }
    rows = summarize(band_names, src_sigma, aug_sigma, fixed_sigma, already_noisy, autocorr, scales)
    write_rows_csv(os.path.join(output_dir, "noise_restoration_summary.csv"), rows)

    updated = np.array(x, copy=True)
    updated[indices] = fixed_images.astype(x.dtype, copy=False)
    payload["x"] = updated
    payload["noise_restoration_scale"] = np.asarray(scales, dtype=np.float64)
    payload["noise_restoration_inner_fraction"] = np.array(args.inner_fraction, dtype=np.float64)
    np.savez(args.output, **payload)

    print(f"{'bande':>6} {'ratio avant':>12} {'ratio apres':>12} {'ac source':>10} {'ac avant':>10} {'ac apres':>10}")
    print("-" * 64)
    for row in rows:
        print(
            f"{str(row['band']):>6} {row['ratio_before']:>12.4f} {row['ratio_after']:>12.4f} "
            f"{row['autocorr_source']:>10.4f} {row['autocorr_before']:>10.4f} {row['autocorr_after']:>10.4f}"
        )
    print(
        "\nLecture: 'ratio apres' doit approcher 1.000. Les colonnes 'ac' donnent l'autocorrelation\n"
        "pixel-a-pixel du fond: si 'ac source' est nettement superieure a 'ac apres', le bruit blanc\n"
        "restaure la variance mais pas la structure spatiale, et cette limite doit etre rapportee."
    )
    logger.info("Augmentations a bruit restaure sauvegardees: %s", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinjecte un bruit par bande pour restaurer sigma(fond) au niveau des images sources."
    )
    parser.add_argument("--augmentations", required=True, help="Augmentations DEJA filtrees visuellement.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="Facteur fixe, utilise seulement si --calibrate_iters 0.")
    parser.add_argument("--target_ratio", type=float, default=1.0, help="Ratio sigma(fond) augmentee/source vise.")
    parser.add_argument("--calibrate_iters", type=int, default=3, help="Iterations de calibration du facteur par bande, 0 pour desactiver.")
    parser.add_argument("--calibrate_samples", type=int, default=2000)
    parser.add_argument("--inner_fraction", type=float, default=0.75)
    parser.add_argument("--corner_size", type=int, default=16)
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
