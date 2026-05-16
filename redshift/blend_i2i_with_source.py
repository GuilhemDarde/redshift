import argparse
import os
from typing import Dict, List

import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from config import CONFIG
from data_loader import get_dataset_and_splits
from photometric_validation import denormalize_images, negative_flux_fraction
from renormalize_i2i_flux import band_fluxes, linear_to_model_space
from utils import set_global_seed


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def blend_linear(source_x: np.ndarray, aug_x: np.ndarray, alpha: float) -> np.ndarray:
    source_linear = denormalize_images(source_x)
    aug_linear = denormalize_images(aug_x)
    return source_linear + alpha * (aug_linear - source_linear)


def renormalize_to_source_flux(
    source_x: np.ndarray,
    blended_linear: np.ndarray,
    measure: str,
    scale_min: float,
    scale_max: float,
) -> tuple:
    blended_x = linear_to_model_space(blended_linear).astype(np.float32)
    if measure == "none":
        scale = np.ones(blended_linear.shape[:2], dtype=np.float64)
        return blended_x, scale
    target_flux = band_fluxes(source_x, measure)
    current_flux = band_fluxes(blended_x, measure)
    valid = np.isfinite(target_flux) & np.isfinite(current_flux) & (target_flux > 1e-8) & (current_flux > 1e-8)
    scale = np.ones_like(target_flux, dtype=np.float64)
    scale[valid] = target_flux[valid] / current_flux[valid]
    scale = np.clip(scale, scale_min, scale_max)
    renorm_linear = blended_linear * scale[:, :, None, None]
    return linear_to_model_space(renorm_linear).astype(np.float32), scale


def band_summary_rows(source_x: np.ndarray, before_x: np.ndarray, after_x: np.ndarray, scale: np.ndarray) -> List[Dict[str, object]]:
    source_linear = denormalize_images(source_x)
    before_linear = denormalize_images(before_x)
    after_linear = denormalize_images(after_x)
    source_flux = np.sum(np.clip(source_linear, 0.0, None), axis=(2, 3))
    before_flux = np.sum(np.clip(before_linear, 0.0, None), axis=(2, 3))
    after_flux = np.sum(np.clip(after_linear, 0.0, None), axis=(2, 3))
    source_l1_denom = np.sum(np.abs(source_linear), axis=(2, 3)) + 1e-12
    before_l1 = np.sum(np.abs(before_linear - source_linear), axis=(2, 3)) / source_l1_denom
    after_l1 = np.sum(np.abs(after_linear - source_linear), axis=(2, 3)) / source_l1_denom

    rows: List[Dict[str, object]] = []
    for band_idx, band in enumerate(CONFIG.BAND_NAMES[: source_linear.shape[1]]):
        before_corr = [_correlation(source_linear[i, band_idx], before_linear[i, band_idx]) for i in range(len(source_linear))]
        after_corr = [_correlation(source_linear[i, band_idx], after_linear[i, band_idx]) for i in range(len(source_linear))]
        rows.append({
            "band": band,
            "before_flux_ratio_median": float(np.nanmedian(before_flux[:, band_idx] / (source_flux[:, band_idx] + 1e-12))),
            "after_flux_ratio_median": float(np.nanmedian(after_flux[:, band_idx] / (source_flux[:, band_idx] + 1e-12))),
            "before_relative_l1_median": float(np.nanmedian(before_l1[:, band_idx])),
            "after_relative_l1_median": float(np.nanmedian(after_l1[:, band_idx])),
            "before_correlation_median": float(np.nanmedian(before_corr)),
            "after_correlation_median": float(np.nanmedian(after_corr)),
            "scale_median": float(np.nanmedian(scale[:, band_idx])),
        })
    return rows


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    candidates = np.load(args.augmentations, allow_pickle=False)
    if "x" not in candidates.files or "source_index" not in candidates.files:
        raise KeyError("Le fichier augmentations doit contenir x et source_index.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha doit etre dans [0, 1].")

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
    source_index = np.asarray(candidates["source_index"], dtype=np.int64)
    valid = (source_index >= 0) & (source_index < len(dataset.data["x"]))
    if not np.all(valid):
        raise ValueError(f"source_index invalide pour {int(np.sum(~valid))} augmentations.")

    source_x = np.asarray(dataset.data["x"][source_index], dtype=np.float32)
    aug_x = np.asarray(candidates["x"], dtype=np.float32)
    blended_linear = blend_linear(source_x, aug_x, args.alpha)
    blended_x, scale = renormalize_to_source_flux(
        source_x,
        blended_linear,
        measure=args.renormalize_flux,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )

    payload = {key: candidates[key] for key in candidates.files}
    payload["x"] = blended_x
    payload["blend_alpha"] = np.array(args.alpha, dtype=np.float32)
    payload["blend_renormalize_flux"] = np.array(args.renormalize_flux)
    payload["blend_source_flux_scale"] = scale.astype(np.float32)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.compressed:
        np.savez_compressed(args.output, **payload)
    else:
        np.savez(args.output, **payload)

    write_rows_csv(
        os.path.join(output_dir, "blend_band_summary.csv"),
        band_summary_rows(source_x, aug_x, blended_x, scale),
    )
    write_rows_csv(os.path.join(output_dir, "blend_image_summary.csv"), [{
        "alpha": args.alpha,
        "renormalize_flux": args.renormalize_flux,
        "n": int(len(blended_x)),
        "negative_flux_fraction_before_median": float(np.nanmedian(negative_flux_fraction(aug_x))),
        "negative_flux_fraction_after_median": float(np.nanmedian(negative_flux_fraction(blended_x))),
    }])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--renormalize_flux", choices=["none", "signed", "positive"], default="positive")
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
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--compressed", action="store_true")
    run(parser.parse_args())
