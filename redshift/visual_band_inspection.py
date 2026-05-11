import argparse
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from config import CONFIG
from data_loader import get_dataset_and_splits
from photometric_validation import denormalize_images, negative_flux_fraction
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _finite_quantile(values: np.ndarray, q: float, fallback: float) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    return float(np.quantile(values, q))


def _scale_gray(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    return np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)


def _rgb_preview_from_linear(linear: np.ndarray) -> np.ndarray:
    '''
    actions : Construit un apercu RGB stable a partir des bandes g/r/i.
    inputs : linear (np.ndarray C,H,W)
    appels : np.stack, np.arcsinh, np.quantile
    outputs : np.ndarray H,W,3
    '''
    channels = [3, 2, 1] if linear.shape[0] >= 4 else list(range(min(3, linear.shape[0])))[::-1]
    rgb = np.stack([np.clip(linear[idx], 0.0, None) for idx in channels], axis=-1)
    hi = _finite_quantile(rgb, 0.995, 1.0)
    if hi <= 0.0:
        return np.zeros_like(rgb)
    rgb = np.arcsinh(8.0 * rgb / hi) / np.arcsinh(8.0)
    return np.clip(rgb, 0.0, 1.0)


def _rgb_preview(image: np.ndarray) -> np.ndarray:
    return _rgb_preview_from_linear(denormalize_images(image[None])[0])


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= 1e-12 or b_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bandwise_visual_metrics(source_images: np.ndarray, augmented_images: np.ndarray) -> Dict[str, np.ndarray]:
    '''
    actions : Mesure les ecarts visuels par bande entre source reelle et augmentation.
    inputs : source_images, augmented_images (N,C,H,W)
    appels : denormalize_images, np.sum, _correlation
    outputs : Dict[str, np.ndarray]
    '''
    source_linear = denormalize_images(source_images)
    aug_linear = denormalize_images(augmented_images)
    eps = 1e-12
    metrics: Dict[str, List[np.ndarray]] = {
        "source_flux": [],
        "aug_flux": [],
        "flux_ratio": [],
        "relative_l1": [],
        "correlation": [],
    }

    source_pos = np.clip(source_linear, 0.0, None)
    aug_pos = np.clip(aug_linear, 0.0, None)
    source_flux = np.sum(source_pos, axis=(2, 3))
    aug_flux = np.sum(aug_pos, axis=(2, 3))
    metrics["source_flux"] = [source_flux]
    metrics["aug_flux"] = [aug_flux]
    metrics["flux_ratio"] = [aug_flux / (source_flux + eps)]
    metrics["relative_l1"] = [np.sum(np.abs(aug_linear - source_linear), axis=(2, 3)) / (np.sum(np.abs(source_linear), axis=(2, 3)) + eps)]

    corr = np.full(source_flux.shape, np.nan, dtype=np.float64)
    for sample_idx in range(source_linear.shape[0]):
        for band_idx in range(source_linear.shape[1]):
            corr[sample_idx, band_idx] = _correlation(source_linear[sample_idx, band_idx], aug_linear[sample_idx, band_idx])
    metrics["correlation"] = [corr]

    return {key: np.asarray(values[0], dtype=np.float64) for key, values in metrics.items()}


def select_visual_indices(
    candidate_indices: np.ndarray,
    median_relative_l1: np.ndarray,
    max_examples: int,
    seed: int,
    strategy: str = "mixed",
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if len(candidate_indices) <= max_examples:
        return candidate_indices
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return np.sort(rng.choice(candidate_indices, size=max_examples, replace=False))
    if strategy == "largest_change":
        order = np.argsort(median_relative_l1)[::-1]
        return np.sort(candidate_indices[order[:max_examples]])

    n_hard = max_examples // 2
    n_random = max_examples - n_hard
    hard = candidate_indices[np.argsort(median_relative_l1)[::-1][:n_hard]]
    remaining = np.setdiff1d(candidate_indices, hard, assume_unique=False)
    if len(remaining) > n_random:
        random = rng.choice(remaining, size=n_random, replace=False)
    else:
        random = remaining
    return np.sort(np.concatenate([hard, random]))


def _plot_contact_sheet(
    source_images: np.ndarray,
    aug_images: np.ndarray,
    candidate_indices: np.ndarray,
    source_indices: np.ndarray,
    output_path: str,
) -> None:
    n = len(candidate_indices)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.35 * n))
    if n == 1:
        axes = np.asarray([axes])
    for row in range(n):
        src_rgb = _rgb_preview(source_images[row])
        aug_rgb = _rgb_preview(aug_images[row])
        diff = np.sum(np.abs(denormalize_images(aug_images[row:row + 1])[0] - denormalize_images(source_images[row:row + 1])[0]), axis=0)
        vmax = _finite_quantile(diff, 0.995, 1.0)
        axes[row, 0].imshow(src_rgb, origin="lower")
        axes[row, 1].imshow(aug_rgb, origin="lower")
        axes[row, 2].imshow(_scale_gray(diff, 0.0, vmax), origin="lower", cmap="magma")
        axes[row, 0].set_ylabel(f"cand {int(candidate_indices[row])}\nsrc {int(source_indices[row])}", fontsize=8)
        if row == 0:
            axes[row, 0].set_title("Source RGB")
            axes[row, 1].set_title("Generated RGB")
            axes[row, 2].set_title("Diagnostic |generated-source|")
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_band_panel(
    source_image: np.ndarray,
    aug_image: np.ndarray,
    candidate_index: int,
    source_index: int,
    row: Dict[str, float],
    output_path: str,
) -> None:
    source_linear = denormalize_images(source_image[None])[0]
    aug_linear = denormalize_images(aug_image[None])[0]
    band_names = CONFIG.BAND_NAMES[: source_linear.shape[0]]
    n_rows = len(band_names) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(8.7, 2.0 * n_rows))

    axes[0, 0].imshow(_rgb_preview_from_linear(source_linear), origin="lower")
    axes[0, 1].imshow(_rgb_preview_from_linear(aug_linear), origin="lower")
    abs_delta = np.sum(np.abs(aug_linear - source_linear), axis=0)
    axes[0, 2].imshow(_scale_gray(abs_delta, 0.0, _finite_quantile(abs_delta, 0.995, 1.0)), origin="lower", cmap="magma")
    axes[0, 0].set_title("Source RGB")
    axes[0, 1].set_title("Generated RGB")
    axes[0, 2].set_title("Diagnostic |generated-source|")

    for band_idx, band in enumerate(band_names):
        r = band_idx + 1
        source_band = source_linear[band_idx]
        aug_band = aug_linear[band_idx]
        delta = aug_band - source_band
        display = np.concatenate([source_band.ravel(), aug_band.ravel()])
        vmin = _finite_quantile(display, 0.005, 0.0)
        vmax = _finite_quantile(display, 0.995, 1.0)
        dmax = _finite_quantile(np.abs(delta), 0.995, 1.0)

        axes[r, 0].imshow(_scale_gray(source_band, vmin, vmax), origin="lower", cmap="gray")
        axes[r, 1].imshow(_scale_gray(aug_band, vmin, vmax), origin="lower", cmap="gray")
        axes[r, 2].imshow(delta, origin="lower", cmap="coolwarm", vmin=-dmax, vmax=dmax)
        axes[r, 0].set_ylabel(band)
        axes[r, 1].set_title(
            f"ratio={row.get(f'flux_ratio_{band}', float('nan')):.3f} "
            f"L1={row.get(f'relative_l1_{band}', float('nan')):.3f}",
            fontsize=9,
        )
        axes[r, 2].set_title(f"corr={row.get(f'corr_{band}', float('nan')):.3f}", fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Generated candidate {candidate_index} from source {source_index} | "
        f"z={row.get('z', float('nan')):.3f} mag_i={row.get('mag_i', float('nan')):.3f}",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _metric_rows(
    candidate_indices: np.ndarray,
    source_indices: np.ndarray,
    cond: np.ndarray,
    modes: Optional[np.ndarray],
    strengths: Optional[np.ndarray],
    metrics: Dict[str, np.ndarray],
    neg_aug: np.ndarray,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    band_names = CONFIG.BAND_NAMES[: metrics["source_flux"].shape[1]]
    for pos, candidate_index in enumerate(candidate_indices):
        row: Dict[str, float] = {
            "candidate_index": int(candidate_index),
            "source_index": int(source_indices[pos]),
            "mode": str(modes[pos]) if modes is not None else "",
            "strength": float(strengths[pos]) if strengths is not None else float("nan"),
            "z": float(cond[pos, 0]) if cond.shape[1] > 0 else float("nan"),
            "mag_i": float(cond[pos, 1] * 2.0 + 22.0) if cond.shape[1] > 1 else float("nan"),
            "negative_flux_fraction_aug": float(neg_aug[pos]),
            "median_relative_l1": float(np.nanmedian(metrics["relative_l1"][pos])),
            "median_flux_ratio": float(np.nanmedian(metrics["flux_ratio"][pos])),
        }
        for band_idx, band in enumerate(band_names):
            row[f"source_flux_{band}"] = float(metrics["source_flux"][pos, band_idx])
            row[f"aug_flux_{band}"] = float(metrics["aug_flux"][pos, band_idx])
            row[f"flux_ratio_{band}"] = float(metrics["flux_ratio"][pos, band_idx])
            row[f"relative_l1_{band}"] = float(metrics["relative_l1"][pos, band_idx])
            row[f"corr_{band}"] = float(metrics["correlation"][pos, band_idx])
        rows.append(row)
    return rows


def _summary_rows(rows: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    rows = list(rows)
    if not rows:
        return []
    summary = []
    for band in CONFIG.BAND_NAMES:
        ratio_key = f"flux_ratio_{band}"
        l1_key = f"relative_l1_{band}"
        corr_key = f"corr_{band}"
        if ratio_key not in rows[0]:
            continue
        ratio = np.asarray([row[ratio_key] for row in rows], dtype=np.float64)
        l1 = np.asarray([row[l1_key] for row in rows], dtype=np.float64)
        corr = np.asarray([row[corr_key] for row in rows], dtype=np.float64)
        summary.append({
            "band": band,
            "median_flux_ratio": float(np.nanmedian(ratio)),
            "p16_flux_ratio": float(np.nanquantile(ratio, 0.16)),
            "p84_flux_ratio": float(np.nanquantile(ratio, 0.84)),
            "median_relative_l1": float(np.nanmedian(l1)),
            "median_correlation": float(np.nanmedian(corr)),
        })
    return summary


def _parse_bands(value: str) -> List[str]:
    bands = [band.strip() for band in value.split(",") if band.strip()]
    unknown = sorted(set(bands) - set(CONFIG.BAND_NAMES))
    if unknown:
        raise ValueError(f"Bandes inconnues: {unknown}")
    return bands


def visual_acceptance_mask(rows: List[Dict[str, float]], args: argparse.Namespace) -> np.ndarray:
    '''
    actions : Filtre les augmentations qui changent trop les flux par bande ou qui sont des quasi-copies.
    inputs : rows (List[Dict]), args
    appels : _parse_bands, np.isfinite
    outputs : np.ndarray bool
    '''
    core_bands = _parse_bands(args.core_bands)
    edge_bands = [band for band in CONFIG.BAND_NAMES if band not in core_bands]
    mask = np.ones(len(rows), dtype=bool)
    for idx, row in enumerate(rows):
        median_l1 = row.get("median_relative_l1", float("nan"))
        neg_frac = row.get("negative_flux_fraction_aug", float("nan"))
        keep = np.isfinite(median_l1) and np.isfinite(neg_frac)
        keep &= args.min_median_relative_l1 <= median_l1 <= args.max_median_relative_l1
        keep &= neg_frac <= args.max_negative_fraction

        for band in core_bands:
            ratio = row.get(f"flux_ratio_{band}", float("nan"))
            l1 = row.get(f"relative_l1_{band}", float("nan"))
            corr = row.get(f"corr_{band}", float("nan"))
            keep &= np.isfinite(ratio) and args.core_flux_ratio_min <= ratio <= args.core_flux_ratio_max
            keep &= np.isfinite(l1) and l1 <= args.core_relative_l1_max
            keep &= np.isfinite(corr) and corr >= args.core_corr_min

        for band in edge_bands:
            ratio = row.get(f"flux_ratio_{band}", float("nan"))
            l1 = row.get(f"relative_l1_{band}", float("nan"))
            corr = row.get(f"corr_{band}", float("nan"))
            keep &= np.isfinite(ratio) and args.edge_flux_ratio_min <= ratio <= args.edge_flux_ratio_max
            keep &= np.isfinite(l1) and l1 <= args.edge_relative_l1_max
            keep &= np.isfinite(corr) and corr >= args.edge_corr_min
        mask[idx] = bool(keep)
    return mask


def _write_filtered_augmentations(
    candidates: np.lib.npyio.NpzFile,
    output_path: str,
    original_candidate_indices: np.ndarray,
    visual_mask: np.ndarray,
    metric_rows: List[Dict[str, float]],
    n_total_candidates: int,
) -> None:
    selected = np.asarray(original_candidate_indices[visual_mask], dtype=np.int64)
    selected_mask = np.zeros(n_total_candidates, dtype=bool)
    selected_mask[selected] = True
    payload = {}
    for key in candidates.files:
        arr = candidates[key]
        if np.asarray(arr).shape[:1] == (n_total_candidates,):
            payload[key] = arr[selected_mask]
        else:
            payload[key] = arr
    payload["visual_acceptance_mask_candidates"] = selected_mask
    payload["visual_candidate_index"] = selected
    for key in ["median_relative_l1", "median_flux_ratio", "negative_flux_fraction_aug"]:
        payload[f"visual_{key}"] = np.asarray([row[key] for row in metric_rows], dtype=np.float64)[visual_mask]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **payload)


def _plot_band_metric_summary(summary_rows: List[Dict[str, float]], output_dir: str) -> None:
    if not summary_rows:
        return
    bands = [row["band"] for row in summary_rows]
    median = np.asarray([row["median_flux_ratio"] for row in summary_rows])
    lo = np.asarray([row["p16_flux_ratio"] for row in summary_rows])
    hi = np.asarray([row["p84_flux_ratio"] for row in summary_rows])
    l1 = np.asarray([row["median_relative_l1"] for row in summary_rows])
    corr = np.asarray([row["median_correlation"] for row in summary_rows])
    x = np.arange(len(bands))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].errorbar(x, median, yerr=[median - lo, hi - median], fmt="o", capsize=4)
    axes[0].axhline(1.0, color="black", linewidth=1, alpha=0.5)
    axes[0].set_title("Flux ratio generated/source")
    axes[0].set_ylabel("ratio")
    axes[1].bar(x, l1)
    axes[1].set_title("Relative L1")
    axes[1].set_ylabel("median")
    axes[2].bar(x, corr)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Pixel correlation")
    axes[2].set_ylabel("median")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "band_metric_summary.png"), dpi=150)
    plt.close()


def _write_visual_report(output_dir: str, n_available: int, metric_rows: List[Dict[str, float]], summary_rows: List[Dict[str, float]], args: argparse.Namespace) -> None:
    visual_mask = visual_acceptance_mask(metric_rows, args)
    lines = [
        "# Visual Band Inspection Report",
        "",
        f"- Augmentation file: `{args.augmentations}`",
        f"- Valid source-linked candidates: `{n_available}`",
        f"- Metric sample size: `{len(metric_rows)}`",
        f"- Accepted by visual filter: `{int(np.sum(visual_mask))}` (`{100.0 * np.sum(visual_mask) / max(len(visual_mask), 1):.2f}%`)",
        f"- Mode filter: `{args.mode_filter or 'none'}`",
        f"- Selection strategy: `{args.selection}`",
        f"- Filtered output: `{args.output_filtered or 'none'}`",
        "",
        "## Band Summary",
        "",
        "| band | median flux ratio | p16-p84 flux ratio | median relative L1 | median corr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['band']} | {row['median_flux_ratio']:.3f} | "
            f"{row['p16_flux_ratio']:.3f}-{row['p84_flux_ratio']:.3f} | "
            f"{row['median_relative_l1']:.3f} | {row['median_correlation']:.3f} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "- `rgb_contact_sheet.png`: source, generated image, and absolute difference diagnostic.",
        "- `band_panel_*.png`: source/generated image/difference diagnostic for every band.",
        "- `band_metric_summary.png`: band-level flux ratio, L1 and pixel correlation.",
        "",
        "Important interpretation:",
        "",
        "- The CFM output is the generated image. Difference maps are computed after generation for quality control only.",
        "- Difference maps must not be presented as the diffusion output or as a noise image.",
        "",
        "Manual checks to report:",
        "",
        "- The object should remain in the same location and not duplicate or erase nearby sources.",
        "- No single band should show a strong artifact absent from the others.",
        "- Residual maps should look like mild local perturbations, not global background shifts.",
        "- Flux ratios should stay close to one unless the photometric condition justifies a change.",
    ])
    with open(os.path.join(output_dir, "visual_band_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    candidates = np.load(args.augmentations, allow_pickle=False)
    x = candidates["x"]
    cond = candidates["cond"]
    source_index = candidates["source_index"] if "source_index" in candidates.files else np.full(len(x), -1, dtype=np.int64)
    modes = candidates["mode"] if "mode" in candidates.files else None
    strengths = candidates["strength"] if "strength" in candidates.files else None

    dataset, _ = get_dataset_and_splits(
        region=args.region,
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
    )

    valid = (source_index >= 0) & (source_index < len(dataset.data["x"]))
    if args.mode_filter is not None and modes is not None:
        valid &= modes == args.mode_filter
    candidate_indices = np.where(valid)[0]
    if len(candidate_indices) == 0:
        raise RuntimeError("Aucun candidat avec source_index valide pour l'inspection visuelle.")

    metric_indices = candidate_indices
    if args.max_metric_samples is not None and len(metric_indices) > args.max_metric_samples:
        rng = np.random.default_rng(args.seed)
        metric_indices = np.sort(rng.choice(metric_indices, size=args.max_metric_samples, replace=False))

    source_images = dataset.data["x"][source_index[metric_indices]]
    aug_images = x[metric_indices]
    metrics = bandwise_visual_metrics(source_images, aug_images)
    neg_aug = negative_flux_fraction(aug_images)
    metric_modes = modes[metric_indices] if modes is not None else None
    metric_strengths = strengths[metric_indices] if strengths is not None else None
    rows = _metric_rows(metric_indices, source_index[metric_indices], cond[metric_indices], metric_modes, metric_strengths, metrics, neg_aug)
    write_rows_csv(os.path.join(output_dir, "visual_band_metrics.csv"), rows)

    summary = _summary_rows(rows)
    write_rows_csv(os.path.join(output_dir, "visual_band_summary.csv"), summary)
    _plot_band_metric_summary(summary, output_dir)

    visual_mask = visual_acceptance_mask(rows, args)
    write_rows_csv(os.path.join(output_dir, "visual_filter_summary.csv"), [{
        "available_source_linked_candidates": int(len(candidate_indices)),
        "evaluated_candidates": int(len(metric_indices)),
        "accepted_by_visual_filter": int(np.sum(visual_mask)),
        "acceptance_rate_pct": float(100.0 * np.sum(visual_mask) / max(len(visual_mask), 1)),
        "core_bands": args.core_bands,
        "core_flux_ratio_min": args.core_flux_ratio_min,
        "core_flux_ratio_max": args.core_flux_ratio_max,
        "edge_flux_ratio_min": args.edge_flux_ratio_min,
        "edge_flux_ratio_max": args.edge_flux_ratio_max,
        "core_relative_l1_max": args.core_relative_l1_max,
        "edge_relative_l1_max": args.edge_relative_l1_max,
        "core_corr_min": args.core_corr_min,
        "edge_corr_min": args.edge_corr_min,
        "min_median_relative_l1": args.min_median_relative_l1,
        "max_median_relative_l1": args.max_median_relative_l1,
    }])

    if args.output_filtered is not None:
        if len(metric_indices) != len(candidate_indices):
            logger.warning(
                "Le fichier filtre ne couvrira que %s/%s candidats car --max_metric_samples est actif.",
                len(metric_indices),
                len(candidate_indices),
            )
        _write_filtered_augmentations(candidates, args.output_filtered, metric_indices, visual_mask, rows, len(x))
        logger.info(
            "Augmentations filtrees visuellement sauvegardees: %s (%s/%s)",
            args.output_filtered,
            int(np.sum(visual_mask)),
            len(visual_mask),
        )

    median_l1 = np.asarray([row["median_relative_l1"] for row in rows], dtype=np.float64)
    panel_indices = select_visual_indices(metric_indices, median_l1, args.max_examples, args.seed, args.selection)
    panel_pos = np.asarray([int(np.where(metric_indices == idx)[0][0]) for idx in panel_indices], dtype=np.int64)
    panel_sources = source_images[panel_pos]
    panel_aug = aug_images[panel_pos]
    panel_source_indices = source_index[panel_indices]

    _plot_contact_sheet(
        panel_sources[: args.max_contact_examples],
        panel_aug[: args.max_contact_examples],
        panel_indices[: args.max_contact_examples],
        panel_source_indices[: args.max_contact_examples],
        os.path.join(output_dir, "rgb_contact_sheet.png"),
    )

    row_by_candidate = {int(row["candidate_index"]): row for row in rows}
    for candidate_idx, src_idx, src_img, aug_img in zip(panel_indices, panel_source_indices, panel_sources, panel_aug):
        _plot_band_panel(
            src_img,
            aug_img,
            int(candidate_idx),
            int(src_idx),
            row_by_candidate[int(candidate_idx)],
            os.path.join(output_dir, f"band_panel_candidate_{int(candidate_idx):06d}_source_{int(src_idx):06d}.png"),
        )

    _write_visual_report(output_dir, len(candidate_indices), rows, summary, args)
    logger.info("Inspection visuelle sauvegardee: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmentations", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=CONFIG.exp_path("visual_band_inspection"))
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="all")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--mode_filter", type=str, default=None)
    parser.add_argument("--max_metric_samples", type=int, default=8000)
    parser.add_argument("--max_examples", type=int, default=12)
    parser.add_argument("--max_contact_examples", type=int, default=12)
    parser.add_argument("--selection", choices=["mixed", "random", "largest_change"], default="mixed")
    parser.add_argument("--output_filtered", type=str, default=None)
    parser.add_argument("--core_bands", type=str, default="g,r,i,z")
    parser.add_argument("--core_flux_ratio_min", type=float, default=0.90)
    parser.add_argument("--core_flux_ratio_max", type=float, default=1.08)
    parser.add_argument("--edge_flux_ratio_min", type=float, default=0.75)
    parser.add_argument("--edge_flux_ratio_max", type=float, default=1.25)
    parser.add_argument("--core_relative_l1_max", type=float, default=0.30)
    parser.add_argument("--edge_relative_l1_max", type=float, default=0.50)
    parser.add_argument("--core_corr_min", type=float, default=0.97)
    parser.add_argument("--edge_corr_min", type=float, default=0.90)
    parser.add_argument("--min_median_relative_l1", type=float, default=0.03)
    parser.add_argument("--max_median_relative_l1", type=float, default=0.30)
    parser.add_argument("--max_negative_fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    run(parser.parse_args())
