import argparse
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis_utils import (
    ensure_dir,
    magnitude_bin_edges,
    magnitude_support_definition_rows,
    magnitude_support_mask,
    write_rows_csv,
)
from config import CONFIG
from data_loader import build_metadata, get_dataset_and_splits
from density_utils import compute_train_knn_density, low_density_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PHOTO_KEYS = ("mag_i", "g_r", "r_i", "i_z")
PHOTO_FEATURE_LIMITS = {
    "mag_i": (CONFIG.I_MIN - 1.0, CONFIG.I_MAX + 1.5),
    "g_r": (-3.0, 3.0),
    "r_i": (-3.0, 3.0),
    "i_z": (-3.0, 3.0),
}


def denormalize_images(x: np.ndarray, asinh_norm: bool = CONFIG.ASINH_NORM) -> np.ndarray:
    '''
    actions : Ramène les images normalisées dans un espace de flux linéaire.
    inputs : x (np.ndarray), asinh_norm (bool)
    appels : np.asarray, np.sinh
    outputs : np.ndarray
    '''
    x = np.asarray(x, dtype=np.float64)
    return np.sinh(np.clip(x, -20.0, 20.0)) if asinh_norm else x


def images_to_fluxes(x: np.ndarray, asinh_norm: bool = CONFIG.ASINH_NORM) -> np.ndarray:
    '''
    actions : Intègre le flux par bande sur chaque timbre multi-bande.
    inputs : x (np.ndarray), asinh_norm (bool)
    appels : denormalize_images, np.sum
    outputs : np.ndarray
    '''
    linear = denormalize_images(x, asinh_norm=asinh_norm)
    return np.sum(linear, axis=(2, 3))


def catalog_magnitudes_from_metadata(metadata: Dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    '''
    actions : Extrait les magnitudes catalogue dans l'ordre CONFIG.BAND_NAMES.
    inputs : metadata (Dict[str, np.ndarray]), indices (np.ndarray)
    appels : np.stack
    outputs : np.ndarray
    '''
    return np.stack([metadata[f"mag_{band}"][indices] for band in CONFIG.BAND_NAMES], axis=1)


def calibrate_zero_points(images: np.ndarray, catalog_mags: np.ndarray, min_flux: float = 1e-8) -> np.ndarray:
    '''
    actions : Calibre un zéro-point empirique par bande via les vraies galaxies de référence.
    inputs : images (np.ndarray), catalog_mags (np.ndarray), min_flux (float)
    appels : images_to_fluxes, np.nanmedian, np.log10
    outputs : np.ndarray
    '''
    fluxes = images_to_fluxes(images)
    zero_points = np.full(len(CONFIG.BAND_NAMES), np.nan, dtype=np.float64)
    for band_idx in range(len(CONFIG.BAND_NAMES)):
        mask = (
            np.isfinite(fluxes[:, band_idx])
            & np.isfinite(catalog_mags[:, band_idx])
            & (fluxes[:, band_idx] > min_flux)
        )
        if np.any(mask):
            zero_points[band_idx] = float(np.nanmedian(catalog_mags[mask, band_idx] + 2.5 * np.log10(fluxes[mask, band_idx])))

    finite = np.isfinite(zero_points)
    if not np.all(finite):
        fill_value = float(np.nanmedian(zero_points[finite])) if np.any(finite) else CONFIG.MAG_MEAN
        zero_points[~finite] = fill_value
    return zero_points


def image_magnitudes(images: np.ndarray, zero_points: np.ndarray, min_flux: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    '''
    actions : Convertit les flux intégrés par bande en magnitudes avec zéro-points empiriques.
    inputs : images (np.ndarray), zero_points (np.ndarray), min_flux (float)
    appels : images_to_fluxes, np.clip, np.log10
    outputs : Tuple[np.ndarray, np.ndarray]
    '''
    fluxes = images_to_fluxes(images)
    safe_fluxes = np.clip(fluxes, min_flux, None)
    mags = zero_points[None, :] - 2.5 * np.log10(safe_fluxes)
    mags[~np.isfinite(fluxes) | (fluxes <= min_flux)] = np.nan
    return mags, fluxes


def photometric_features_from_magnitudes(mags: np.ndarray) -> Dict[str, np.ndarray]:
    '''
    actions : Calcule les observables photométriques utilisées pour le conditionnement et la validation.
    inputs : mags (np.ndarray)
    appels : dict
    outputs : Dict[str, np.ndarray]
    '''
    return {
        "mag_i": mags[:, 3],
        "g_r": mags[:, 1] - mags[:, 2],
        "r_i": mags[:, 2] - mags[:, 3],
        "i_z": mags[:, 3] - mags[:, 4],
    }


def photometric_features_from_conditions(cond: np.ndarray) -> Dict[str, np.ndarray]:
    '''
    actions : Récupère les cibles photométriques depuis le vecteur conditionnel 7D.
    inputs : cond (np.ndarray)
    appels : dict
    outputs : Dict[str, np.ndarray]
    '''
    return {
        "mag_i": cond[:, 1] * 2.0 + 22.0,
        "g_r": cond[:, 2],
        "r_i": cond[:, 3],
        "i_z": cond[:, 4],
    }


def residuals_from_features(observed: Dict[str, np.ndarray], target: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(observed[key]) - np.asarray(target[key]) for key in PHOTO_KEYS}


def feature_limits_mask(features: Dict[str, np.ndarray], limits: Dict[str, Tuple[float, float]] = PHOTO_FEATURE_LIMITS) -> np.ndarray:
    '''
    actions : Rejette les cibles photométriques manifestement non physiques avant calibration et filtrage.
    inputs : features (Dict[str, np.ndarray]), limits (Dict[str, Tuple[float, float]])
    appels : np.asarray, np.isfinite
    outputs : np.ndarray
    '''
    n = len(next(iter(features.values())))
    mask = np.ones(n, dtype=bool)
    for key, (lo, hi) in limits.items():
        values = np.asarray(features[key])
        mask &= np.isfinite(values) & (values >= lo) & (values <= hi)
    return mask


def residual_thresholds(real_residuals: Dict[str, np.ndarray], quantile: float = 0.95, min_threshold: float = 1e-3) -> Dict[str, float]:
    '''
    actions : Construit l'enveloppe empirique d'acceptation à partir des résidus des vraies galaxies.
    inputs : real_residuals (Dict[str, np.ndarray]), quantile (float), min_threshold (float)
    appels : np.quantile, np.abs
    outputs : Dict[str, float]
    '''
    thresholds: Dict[str, float] = {}
    for key in PHOTO_KEYS:
        values = np.asarray(real_residuals[key])
        values = values[np.isfinite(values)]
        if values.size == 0:
            thresholds[key] = float("inf")
        else:
            thresholds[key] = max(float(np.quantile(np.abs(values), quantile)), min_threshold)
    return thresholds


def negative_flux_fraction(images: np.ndarray) -> np.ndarray:
    linear = denormalize_images(images)
    negative = np.sum(np.abs(np.minimum(linear, 0.0)), axis=(2, 3))
    total = np.sum(np.abs(linear), axis=(2, 3)) + 1e-12
    return np.max(negative / total, axis=1)


def acceptance_mask(
    residuals: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    fluxes: np.ndarray,
    images: np.ndarray,
    target_features: Optional[Dict[str, np.ndarray]] = None,
    min_flux: float = 1e-8,
    max_negative_fraction: float = 0.45,
) -> np.ndarray:
    '''
    actions : Accepte uniquement les augmentations photométriquement cohérentes et numériquement stables.
    inputs : residuals, thresholds, fluxes, images, min_flux, max_negative_fraction
    appels : np.isfinite, negative_flux_fraction
    outputs : np.ndarray
    '''
    n = len(fluxes)
    mask = np.ones(n, dtype=bool)
    mask &= np.all(np.isfinite(fluxes) & (fluxes > min_flux), axis=1)
    mask &= negative_flux_fraction(images) <= max_negative_fraction
    if target_features is not None:
        mask &= feature_limits_mask(target_features)
    for key in PHOTO_KEYS:
        values = np.asarray(residuals[key])
        mask &= np.isfinite(values) & (np.abs(values) <= thresholds[key])
    return mask


def compute_morphology_i(images: np.ndarray) -> Dict[str, np.ndarray]:
    '''
    actions : Calcule des métriques morphologiques simples sur la bande i.
    inputs : images (np.ndarray)
    appels : denormalize_images, np.meshgrid, np.argsort
    outputs : Dict[str, np.ndarray]
    '''
    linear = denormalize_images(images)
    band = np.clip(linear[:, 3], 0.0, None)
    n, h, w = band.shape
    yy, xx = np.mgrid[:h, :w]
    radius = np.full(n, np.nan, dtype=np.float64)
    concentration = np.full(n, np.nan, dtype=np.float64)
    asymmetry = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        img = band[i]
        total = float(np.sum(img))
        if not np.isfinite(total) or total <= 0.0:
            continue
        cx = float(np.sum(xx * img) / total)
        cy = float(np.sum(yy * img) / total)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radius[i] = float(np.sqrt(np.sum((r ** 2) * img) / total))

        flat_r = r.ravel()
        flat_flux = img.ravel()
        order = np.argsort(flat_r)
        cumulative = np.cumsum(flat_flux[order])
        r20 = flat_r[order][np.searchsorted(cumulative, 0.2 * total, side="left")]
        r80 = flat_r[order][np.searchsorted(cumulative, 0.8 * total, side="left")]
        concentration[i] = float(5.0 * np.log10((r80 + 1e-6) / (r20 + 1e-6)))
        asymmetry[i] = float(np.sum(np.abs(img - np.rot90(img, 2))) / (total + 1e-12))

    return {"morph_radius_i": radius, "morph_concentration_i": concentration, "morph_asymmetry_i": asymmetry}


def _sample_indices(indices: np.ndarray, max_count: Optional[int], seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_count is None or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_count, replace=False))


def _plot_histograms(real_features: Dict[str, np.ndarray], accepted_features: Dict[str, np.ndarray], output_dir: str, reference_label: str) -> None:
    for key in PHOTO_KEYS:
        plt.figure(figsize=(7, 5))
        real = np.asarray(real_features[key])
        accepted = np.asarray(accepted_features[key])
        real = real[np.isfinite(real)]
        accepted = accepted[np.isfinite(accepted)]
        if real.size:
            plt.hist(real, bins=50, density=True, histtype="step", linewidth=2, label=reference_label)
        if accepted.size:
            plt.hist(accepted, bins=50, density=True, histtype="step", linewidth=2, label="Accepted generated")
        combined = np.concatenate([real, accepted]) if real.size and accepted.size else real if real.size else accepted
        if combined.size:
            lo, hi = np.nanquantile(combined, [0.005, 0.995])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                plt.xlim(lo, hi)
        plt.xlabel(key)
        plt.ylabel("Density")
        plt.title(f"Photometric validation: {key}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"photometry_hist_{key}.png"), dpi=150)
        plt.close()


def _plot_color_color(real_features: Dict[str, np.ndarray], accepted_features: Dict[str, np.ndarray], output_dir: str, reference_label: str) -> None:
    pairs = [("g_r", "r_i"), ("r_i", "i_z")]
    for x_key, y_key in pairs:
        plt.figure(figsize=(6, 6))
        plt.scatter(real_features[x_key], real_features[y_key], s=4, alpha=0.25, label=reference_label)
        plt.scatter(accepted_features[x_key], accepted_features[y_key], s=4, alpha=0.25, label="Accepted generated")
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        plt.title(f"Color-color: {x_key} vs {y_key}")
        x = np.concatenate([real_features[x_key], accepted_features[x_key]])
        y = np.concatenate([real_features[y_key], accepted_features[y_key]])
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size and y.size:
            xlo, xhi = np.nanquantile(x, [0.005, 0.995])
            ylo, yhi = np.nanquantile(y, [0.005, 0.995])
            if xlo < xhi:
                plt.xlim(xlo, xhi)
            if ylo < yhi:
                plt.ylim(ylo, yhi)
        plt.grid(True, alpha=0.3)
        plt.legend(markerscale=3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"color_color_{x_key}_{y_key}.png"), dpi=150)
        plt.close()


def _image_preview(image: np.ndarray) -> np.ndarray:
    linear = denormalize_images(image[None])[0]
    return np.sum(linear[1:4], axis=0)


def _plot_examples(dataset_images: np.ndarray, candidate_images: np.ndarray, source_indices: np.ndarray, accepted_mask: np.ndarray, output_dir: str) -> None:
    accepted = np.where(accepted_mask)[0][:4]
    rejected = np.where(~accepted_mask)[0][:4]
    slots = [("accepted", accepted), ("rejected", rejected)]
    for label, idxs in slots:
        if len(idxs) == 0:
            continue
        fig, axes = plt.subplots(len(idxs), 2, figsize=(5, 2.5 * len(idxs)))
        if len(idxs) == 1:
            axes = np.asarray([axes])
        for row, cand_idx in enumerate(idxs):
            src_idx = int(source_indices[cand_idx]) if source_indices is not None and source_indices[cand_idx] >= 0 else -1
            if src_idx >= 0:
                axes[row, 0].imshow(_image_preview(dataset_images[src_idx]), cmap="magma", origin="lower")
                axes[row, 0].set_title("Source")
            else:
                axes[row, 0].axis("off")
            axes[row, 1].imshow(_image_preview(candidate_images[cand_idx]), cmap="magma", origin="lower")
            axes[row, 1].set_title(label)
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"examples_{label}.png"), dpi=150)
        plt.close()


def _distribution_rows(real_features: Dict[str, np.ndarray], accepted_features: Dict[str, np.ndarray], thresholds: Dict[str, float]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for key in PHOTO_KEYS:
        real = np.asarray(real_features[key])
        accepted = np.asarray(accepted_features[key])
        real = real[np.isfinite(real)]
        accepted = accepted[np.isfinite(accepted)]
        row = {
            "feature": key,
            "real_n": int(real.size),
            "accepted_n": int(accepted.size),
            "threshold_abs_residual": thresholds[key],
            "real_median": float(np.median(real)) if real.size else float("nan"),
            "accepted_median": float(np.median(accepted)) if accepted.size else float("nan"),
            "median_shift": float(np.median(accepted) - np.median(real)) if real.size and accepted.size else float("nan"),
            "ks_stat": float("nan"),
            "ks_pvalue": float("nan"),
            "wasserstein": float("nan"),
        }
        if real.size and accepted.size:
            ks = stats.ks_2samp(real, accepted)
            row["ks_stat"] = float(ks.statistic)
            row["ks_pvalue"] = float(ks.pvalue)
            row["wasserstein"] = float(stats.wasserstein_distance(real, accepted))
        rows.append(row)
    return rows


def _write_report(output_dir: str, rows: Iterable[Dict[str, float]], n_candidates: int, n_accepted: int, zero_points: np.ndarray, args: argparse.Namespace, reference_label: str) -> None:
    acceptance_rate = 100.0 * n_accepted / max(n_candidates, 1)
    lines = [
        "# Photometric Validation Report",
        "",
        f"- Candidates: `{n_candidates}`",
        f"- Accepted: `{n_accepted}`",
        f"- Acceptance rate: `{acceptance_rate:.2f}%`",
        f"- Reference selection: `{args.selection_target}`",
        f"- Reference label: `{reference_label}`",
        f"- Low magnitude support quantile: `{args.low_mag_support_quantile}`",
        f"- kNN density k, legacy only: `{args.knn_k}`",
        f"- Low-density quantile, legacy only: `{args.low_density_quantile}`",
        f"- Residual quantile: `{args.residual_quantile}`",
        "",
        "## Empirical zero-points",
        "",
    ]
    for band, zp in zip(CONFIG.BAND_NAMES, zero_points):
        lines.append(f"- `{band}`: `{zp:.6f}`")
    lines.extend(["", "## Distribution checks", ""])
    for row in rows:
        lines.append(
            f"- `{row['feature']}`: threshold `{row['threshold_abs_residual']:.4f}`, "
            f"median shift `{row['median_shift']:.4f}`, KS `{row['ks_stat']:.4f}`, "
            f"Wasserstein `{row['wasserstein']:.4f}`"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "- `photometry_hist_*.png`",
        "- `color_color_*.png`",
        "- `examples_accepted.png` / `examples_rejected.png` when available",
    ])
    with open(os.path.join(output_dir, "photometry_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def select_reference_indices(metadata: Dict[str, np.ndarray], split_indices: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[np.ndarray, str, Dict[str, np.ndarray]]:
    train_indices = split_indices["train"]
    context: Dict[str, np.ndarray] = {}
    if args.selection_target == "all_train":
        return train_indices, "Real train reference", context
    if args.selection_target == "faint_mag":
        faint = metadata["mag_i"] >= args.faint_mag_threshold
        context["faint_mag_threshold"] = np.array(args.faint_mag_threshold)
        return train_indices[faint[train_indices]], f"Real train mag_i >= {args.faint_mag_threshold:.2f}", context
    if args.selection_target == "low_density":
        density, _ = compute_train_knn_density(metadata["ra"], metadata["dec"], train_indices, k=args.knn_k)
        low_mask_all, threshold = low_density_mask(density, train_indices, quantile=args.low_density_quantile)
        context["density_threshold"] = np.array(threshold)
        return train_indices[low_mask_all[train_indices]], "Real legacy RA/DEC low-density", context
    if args.selection_target != "low_mag_support":
        raise ValueError(f"selection_target inconnu: {args.selection_target}")

    edges = magnitude_bin_edges(args.mag_i_min, args.mag_i_max, args.mag_i_bins)
    low_mask_all, threshold, support_count, mag_bin, train_counts = magnitude_support_mask(
        metadata["mag_i"],
        metadata["mag_i"][train_indices],
        edges,
        quantile=args.low_mag_support_quantile,
    )
    context.update({
        "mag_i_edges": edges,
        "low_mag_support_threshold": np.array(threshold),
        "mag_support_count": support_count,
        "mag_bin": mag_bin,
        "mag_bin_train_counts": train_counts,
    })
    return train_indices[low_mask_all[train_indices]], "Real low-magnitude-support", context


def validate_candidates(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    dataset, split_indices = get_dataset_and_splits(
        region=args.region,
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
        split_strategy=args.split_strategy,
    )
    metadata = build_metadata(dataset, split_indices=split_indices)
    ref_indices, reference_label, reference_context = select_reference_indices(metadata, split_indices, args)
    ref_indices = _sample_indices(ref_indices, args.max_reference, args.seed)
    if len(ref_indices) == 0:
        raise RuntimeError("Aucun objet de référence disponible pour calibrer la validation.")

    ref_catalog_mags = catalog_magnitudes_from_metadata(metadata, ref_indices)
    ref_target_features = photometric_features_from_magnitudes(ref_catalog_mags)
    ref_valid = feature_limits_mask(ref_target_features)
    ref_indices = ref_indices[ref_valid]
    ref_catalog_mags = ref_catalog_mags[ref_valid]
    if len(ref_indices) == 0:
        raise RuntimeError("Aucune référence photometriquement valide pour calibrer la validation.")

    zero_points = calibrate_zero_points(dataset.data["x"][ref_indices], ref_catalog_mags)
    ref_img_mags, _ = image_magnitudes(dataset.data["x"][ref_indices], zero_points)
    ref_img_features = photometric_features_from_magnitudes(ref_img_mags)
    ref_target_features = photometric_features_from_magnitudes(ref_catalog_mags)
    real_residuals = residuals_from_features(ref_img_features, ref_target_features)
    thresholds = residual_thresholds(real_residuals, quantile=args.residual_quantile)

    candidates = np.load(args.candidates, allow_pickle=False)
    x = candidates["x"]
    cond = candidates["cond"]
    source_index = candidates["source_index"] if "source_index" in candidates.files else np.full(len(x), -1, dtype=np.int64)

    cand_mags, cand_fluxes = image_magnitudes(x, zero_points)
    cand_features = photometric_features_from_magnitudes(cand_mags)
    target_features = photometric_features_from_conditions(cond)
    cand_residuals = residuals_from_features(cand_features, target_features)
    accepted_mask = acceptance_mask(
        cand_residuals,
        thresholds,
        cand_fluxes,
        x,
        target_features=target_features,
        max_negative_fraction=args.max_negative_fraction,
    )

    accepted_features = {key: values[accepted_mask] for key, values in cand_features.items()}
    rows = _distribution_rows(ref_target_features, accepted_features, thresholds)
    write_rows_csv(os.path.join(args.output_dir, "photometry_distribution_checks.csv"), rows)

    residual_rows = []
    for key in PHOTO_KEYS:
        values = cand_residuals[key]
        residual_rows.append({
            "feature": key,
            "candidate_median_abs_residual": float(np.nanmedian(np.abs(values))),
            "accepted_median_abs_residual": float(np.nanmedian(np.abs(values[accepted_mask]))) if np.any(accepted_mask) else float("nan"),
            "threshold_abs_residual": thresholds[key],
        })
    write_rows_csv(os.path.join(args.output_dir, "photometry_residual_summary.csv"), residual_rows)

    morph = compute_morphology_i(x)
    morph_rows = []
    for key, values in morph.items():
        morph_rows.append({
            "metric": key,
            "candidate_median": float(np.nanmedian(values)),
            "accepted_median": float(np.nanmedian(values[accepted_mask])) if np.any(accepted_mask) else float("nan"),
        })
    write_rows_csv(os.path.join(args.output_dir, "morphology_summary.csv"), morph_rows)

    _plot_histograms(ref_target_features, accepted_features, args.output_dir, reference_label)
    _plot_color_color(ref_target_features, accepted_features, args.output_dir, reference_label)
    _plot_examples(dataset.data["x"], x, source_index, accepted_mask, args.output_dir)
    _write_report(args.output_dir, rows, len(x), int(np.sum(accepted_mask)), zero_points, args, reference_label)

    output_filtered = args.output_filtered or os.path.join(args.output_dir, "cfm_i2i_accepted.npz")
    payload = {}
    for key in candidates.files:
        arr = candidates[key]
        if np.asarray(arr).shape[:1] == (len(x),):
            payload[key] = arr[accepted_mask]
        else:
            payload[key] = arr
    payload["acceptance_mask_candidates"] = accepted_mask
    payload["zero_points"] = zero_points
    payload["selection_target"] = np.array(args.selection_target)
    for key, value in reference_context.items():
        payload[f"reference_{key}"] = value
    for key in PHOTO_KEYS:
        payload[f"residual_{key}"] = cand_residuals[key][accepted_mask]
        payload[f"threshold_{key}"] = np.array(thresholds[key])
    np.savez(output_filtered, **payload)
    if args.selection_target == "low_mag_support":
        write_rows_csv(
            os.path.join(args.output_dir, "mag_support_definition.csv"),
            magnitude_support_definition_rows(
                reference_context["mag_i_edges"],
                reference_context["mag_bin_train_counts"],
                float(reference_context["low_mag_support_threshold"]),
            ),
        )
    logger.info("Validation terminee: %s/%s acceptes -> %s", int(np.sum(accepted_mask)), len(x), output_filtered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--output_filtered", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=CONFIG.exp_path("photometry_validation"))
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="all")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--split_strategy", choices=["spatial", "marie_regular", "marie_strict"], default="spatial")
    parser.add_argument("--selection_target", choices=["low_mag_support", "faint_mag", "all_train", "low_density"], default="low_mag_support")
    parser.add_argument("--faint_mag_threshold", type=float, default=23.5)
    parser.add_argument("--mag_i_min", type=float, default=CONFIG.I_MIN)
    parser.add_argument("--mag_i_max", type=float, default=CONFIG.I_MAX)
    parser.add_argument("--mag_i_bins", type=int, default=14)
    parser.add_argument("--low_mag_support_quantile", type=float, default=0.20)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--low_density_quantile", type=float, default=0.20)
    parser.add_argument("--residual_quantile", type=float, default=0.95)
    parser.add_argument("--max_reference", type=int, default=20000)
    parser.add_argument("--max_negative_fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    validate_candidates(parser.parse_args())
