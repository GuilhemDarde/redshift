import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from config import CONFIG
from data_loader import build_metadata, get_dataset_and_splits
from photometric_validation import (
    calibrate_zero_points,
    catalog_magnitudes_from_metadata,
    denormalize_images,
    images_to_fluxes,
    negative_flux_fraction,
)
from utils import set_global_seed


def band_fluxes(x: np.ndarray, measure: str) -> np.ndarray:
    if measure == "signed":
        return images_to_fluxes(x)
    if measure == "positive":
        linear = denormalize_images(x)
        return np.sum(np.clip(linear, 0.0, None), axis=(2, 3))
    raise ValueError(f"Mesure de flux inconnue: {measure}")


def linear_to_model_space(linear: np.ndarray) -> np.ndarray:
    if CONFIG.ASINH_NORM:
        return np.arcsinh(linear)
    return linear


def sample_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def catalog_flux_targets(
    dataset_x: np.ndarray,
    metadata: Dict[str, np.ndarray],
    split_indices: Dict[str, np.ndarray],
    source_index: np.ndarray,
    max_reference: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ref_indices = sample_indices(split_indices["train"], max_reference, seed)
    ref_mags = catalog_magnitudes_from_metadata(metadata, ref_indices)
    zero_points = calibrate_zero_points(dataset_x[ref_indices], ref_mags)
    source_mags = catalog_magnitudes_from_metadata(metadata, source_index)
    target_flux = 10.0 ** ((zero_points[None, :] - source_mags) / 2.5)
    target_flux[~np.isfinite(target_flux)] = np.nan
    return target_flux, zero_points


def source_flux_targets(dataset_x: np.ndarray, source_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return images_to_fluxes(dataset_x[source_index]), np.full(len(CONFIG.BAND_NAMES), np.nan, dtype=np.float64)


def renormalize_images(
    x_aug: np.ndarray,
    target_flux: np.ndarray,
    fallback_flux: np.ndarray,
    flux_measure: str,
    scale_min: float,
    scale_max: float,
    min_flux: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aug_flux = band_fluxes(x_aug, flux_measure)
    target = np.asarray(target_flux, dtype=np.float64).copy()
    fallback = np.asarray(fallback_flux, dtype=np.float64)
    bad_target = ~np.isfinite(target) | (target <= min_flux)
    target[bad_target] = fallback[bad_target]

    valid = np.isfinite(aug_flux) & np.isfinite(target) & (aug_flux > min_flux) & (target > min_flux)
    scale = np.ones_like(aug_flux, dtype=np.float64)
    scale[valid] = target[valid] / aug_flux[valid]
    scale = np.clip(scale, scale_min, scale_max)

    linear = denormalize_images(x_aug)
    renorm_linear = linear * scale[:, :, None, None]
    renorm = linear_to_model_space(renorm_linear).astype(np.float32)
    renorm_flux = band_fluxes(renorm, flux_measure)
    return renorm, scale, aug_flux, renorm_flux


def band_summary_rows(
    target_flux: np.ndarray,
    before_flux: np.ndarray,
    after_flux: np.ndarray,
    scale: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    before_ratio = before_flux / np.clip(target_flux, 1e-8, None)
    after_ratio = after_flux / np.clip(target_flux, 1e-8, None)
    for band_idx, band in enumerate(CONFIG.BAND_NAMES):
        rows.append({
            "band": band,
            "before_flux_ratio_median": float(np.nanmedian(before_ratio[:, band_idx])),
            "before_flux_ratio_p16": float(np.nanquantile(before_ratio[:, band_idx], 0.16)),
            "before_flux_ratio_p84": float(np.nanquantile(before_ratio[:, band_idx], 0.84)),
            "after_flux_ratio_median": float(np.nanmedian(after_ratio[:, band_idx])),
            "after_flux_ratio_p16": float(np.nanquantile(after_ratio[:, band_idx], 0.16)),
            "after_flux_ratio_p84": float(np.nanquantile(after_ratio[:, band_idx], 0.84)),
            "scale_median": float(np.nanmedian(scale[:, band_idx])),
            "scale_p16": float(np.nanquantile(scale[:, band_idx], 0.16)),
            "scale_p84": float(np.nanquantile(scale[:, band_idx], 0.84)),
        })
    return rows


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    candidates = np.load(args.augmentations, allow_pickle=False)
    if "x" not in candidates.files or "source_index" not in candidates.files:
        raise KeyError("Le fichier augmentations doit contenir x et source_index.")

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

    source_index = np.asarray(candidates["source_index"], dtype=np.int64)
    valid_source = (source_index >= 0) & (source_index < len(dataset.data["x"]))
    if not np.all(valid_source):
        raise ValueError(f"source_index invalide pour {int(np.sum(~valid_source))} augmentations.")

    fallback_flux = band_fluxes(dataset.data["x"][source_index], args.flux_measure)
    if args.target == "source_flux":
        target_flux = band_fluxes(dataset.data["x"][source_index], args.flux_measure)
        zero_points = np.full(len(CONFIG.BAND_NAMES), np.nan, dtype=np.float64)
    elif args.target == "catalog_flux":
        target_flux, zero_points = catalog_flux_targets(
            dataset.data["x"],
            metadata,
            split_indices,
            source_index,
            args.max_reference,
            args.seed,
        )
    else:
        raise ValueError(f"Target inconnu: {args.target}")

    renorm, scale, before_flux, after_flux = renormalize_images(
        candidates["x"],
        target_flux,
        fallback_flux,
        flux_measure=args.flux_measure,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )

    payload = {}
    for key in candidates.files:
        payload[key] = candidates[key]
    payload["x"] = renorm
    payload["flux_renormalization_scale"] = scale.astype(np.float32)
    payload["flux_renormalization_target"] = np.array(args.target)
    payload["flux_renormalization_scale_min"] = np.array(args.scale_min, dtype=np.float32)
    payload["flux_renormalization_scale_max"] = np.array(args.scale_max, dtype=np.float32)
    payload["flux_renormalization_measure"] = np.array(args.flux_measure)
    payload["flux_renormalization_zero_points"] = zero_points.astype(np.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.compressed:
        np.savez_compressed(args.output, **payload)
    else:
        np.savez(args.output, **payload)

    write_rows_csv(
        os.path.join(output_dir, "renormalization_band_summary.csv"),
        band_summary_rows(target_flux, before_flux, after_flux, scale),
    )
    write_rows_csv(os.path.join(output_dir, "renormalization_image_summary.csv"), [{
        "negative_flux_fraction_before_median": float(np.nanmedian(negative_flux_fraction(candidates["x"]))),
        "negative_flux_fraction_after_median": float(np.nanmedian(negative_flux_fraction(renorm))),
    }])
    write_rows_csv(os.path.join(output_dir, "renormalization_config.csv"), [{
        "augmentations": args.augmentations,
        "output": args.output,
        "target": args.target,
        "flux_measure": args.flux_measure,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "n": int(len(renorm)),
    }])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", choices=["source_flux", "catalog_flux"], default="source_flux")
    parser.add_argument("--flux_measure", choices=["signed", "positive"], default="signed")
    parser.add_argument("--scale_min", type=float, default=0.5)
    parser.add_argument("--scale_max", type=float, default=2.0)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="all")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--split_strategy", choices=["spatial", "marie_regular", "marie_strict"], default="spatial")
    parser.add_argument("--max_reference", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--compressed", action="store_true")
    run(parser.parse_args())
