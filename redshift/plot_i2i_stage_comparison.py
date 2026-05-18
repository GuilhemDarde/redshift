import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import ensure_dir, write_rows_csv
from config import CONFIG
from data_loader import get_dataset_and_splits
from photometric_validation import denormalize_images
from renormalize_i2i_flux import linear_to_model_space
from utils import set_global_seed
from visual_band_inspection import _rgb_preview, bandwise_visual_metrics, select_visual_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_npz(path: str) -> np.lib.npyio.NpzFile:
    data = np.load(path, allow_pickle=False)
    if "x" not in data.files or "source_index" not in data.files:
        raise KeyError(f"{path} doit contenir les cles x et source_index.")
    return data


def _map_final_to_raw_indices(raw: np.lib.npyio.NpzFile, final: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Retourne, pour chaque ligne du fichier final, l'indice correspondant dans le fichier i2i brut.

    Cas privilegie:
    - les fichiers visual-filtered contiennent visual_candidate_index, qui pointe vers le fichier
      d'entree du filtre visuel. Pour un blend construit depuis le fichier i2i brut accepte, cet
      indice correspond directement a la ligne du fichier i2i brut.

    Fallback:
    - si les deux fichiers ont meme longueur et memes source_index, on suppose l'ordre conserve.
    - sinon on matche par source_index en consommant les occurrences une par une.
    """
    raw_source = np.asarray(raw["source_index"], dtype=np.int64)
    final_source = np.asarray(final["source_index"], dtype=np.int64)

    if "visual_candidate_index" in final.files:
        candidate_index = np.asarray(final["visual_candidate_index"], dtype=np.int64)
        if candidate_index.ndim == 1 and len(candidate_index) == len(final_source):
            if np.all((candidate_index >= 0) & (candidate_index < len(raw_source))):
                if np.all(raw_source[candidate_index] == final_source):
                    return candidate_index
                logger.warning(
                    "visual_candidate_index existe mais les source_index ne correspondent pas tous; fallback par source_index."
                )

    if len(raw_source) == len(final_source) and np.all(raw_source == final_source):
        return np.arange(len(final_source), dtype=np.int64)

    by_source: Dict[int, List[int]] = {}
    for raw_idx, src_idx in enumerate(raw_source):
        by_source.setdefault(int(src_idx), []).append(raw_idx)

    mapped = np.full(len(final_source), -1, dtype=np.int64)
    used_per_source: Dict[int, int] = {}
    for final_idx, src_idx in enumerate(final_source):
        src = int(src_idx)
        offset = used_per_source.get(src, 0)
        options = by_source.get(src, [])
        if offset < len(options):
            mapped[final_idx] = options[offset]
            used_per_source[src] = offset + 1

    if np.any(mapped < 0):
        missing = int(np.sum(mapped < 0))
        raise RuntimeError(
            f"Impossible d'aligner {missing} images finales avec le fichier i2i brut. "
            "Utiliser un fichier final derive du meme fichier brut, idealement avec visual_candidate_index."
        )
    return mapped


def _raw_indices_from_final_metadata(final: np.lib.npyio.NpzFile) -> np.ndarray:
    if "visual_candidate_index" in final.files:
        candidate_index = np.asarray(final["visual_candidate_index"], dtype=np.int64)
        if candidate_index.ndim == 1 and len(candidate_index) == len(final["source_index"]):
            return candidate_index
    return np.full(len(final["source_index"]), -1, dtype=np.int64)


def _reconstruct_raw_i2i_from_final_blend(source_x: np.ndarray, final: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Reconstruit l'image i2i brute si le fichier final est issu de blend_i2i_with_source.py.

    Le blend sauvegarde:
      final = renorm(source + alpha * (raw_i2i - source))

    En inversant la renormalisation par bande puis le melange residuel, on recupere une
    approximation de raw_i2i suffisante pour la planche de reunion source/raw/fusion.
    """
    required = {"blend_alpha", "blend_source_flux_scale"}
    missing = sorted(required - set(final.files))
    if missing:
        raise KeyError(
            "Impossible de reconstruire l'i2i brut sans fichier raw_augmentations. "
            f"Cles manquantes dans le fichier final: {missing}"
        )
    alpha = float(np.asarray(final["blend_alpha"]).reshape(-1)[0])
    if alpha <= 0.0:
        raise ValueError("blend_alpha doit etre strictement positif pour reconstruire l'i2i brut.")

    scale = np.asarray(final["blend_source_flux_scale"], dtype=np.float64)
    if scale.ndim != 2 or scale.shape[0] != len(final["x"]):
        raise ValueError("blend_source_flux_scale doit avoir la forme (N, C).")
    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-12), scale, 1.0)

    source_linear = denormalize_images(source_x)
    final_linear = denormalize_images(np.asarray(final["x"], dtype=np.float32))
    pre_renorm_blend = final_linear / scale[:, :, None, None]
    raw_linear = source_linear + (pre_renorm_blend - source_linear) / alpha
    return linear_to_model_space(raw_linear).astype(np.float32)


def _select_final_rows(
    source_images: np.ndarray,
    final_images: np.ndarray,
    max_examples: int,
    seed: int,
    selection: str,
) -> np.ndarray:
    candidate_indices = np.arange(len(final_images), dtype=np.int64)
    if len(candidate_indices) <= max_examples:
        return candidate_indices
    metrics = bandwise_visual_metrics(source_images, final_images)
    median_l1 = np.nanmedian(metrics["relative_l1"], axis=1)
    return select_visual_indices(candidate_indices, median_l1, max_examples, seed, strategy=selection)


def _plot_stage_contact_sheet(
    source_images: np.ndarray,
    raw_images: np.ndarray,
    final_images: np.ndarray,
    source_indices: np.ndarray,
    raw_indices: np.ndarray,
    final_indices: np.ndarray,
    output_path: str,
    title: str,
) -> None:
    n = len(final_indices)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(7.8, 2.35 * n))
    if n == 1:
        axes = np.asarray([axes])
    for row in range(n):
        axes[row, 0].imshow(_rgb_preview(source_images[row]), origin="lower")
        axes[row, 1].imshow(_rgb_preview(raw_images[row]), origin="lower")
        axes[row, 2].imshow(_rgb_preview(final_images[row]), origin="lower")
        axes[row, 0].set_ylabel(
            f"src {int(source_indices[row])}\nraw {int(raw_indices[row])}\nfinal {int(final_indices[row])}",
            fontsize=7,
        )
        if row == 0:
            axes[row, 0].set_title("Image source reelle")
            axes[row, 1].set_title("Sortie i2i brute")
            axes[row, 2].set_title("Image finale apres fusion")
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(title, y=0.995, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _example_rows(
    source_indices: np.ndarray,
    raw_indices: np.ndarray,
    final_indices: np.ndarray,
    raw_path: Optional[str],
    final_path: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pos, (src_idx, raw_idx, final_idx) in enumerate(zip(source_indices, raw_indices, final_indices)):
        rows.append({
            "example_row": int(pos),
            "source_index": int(src_idx),
            "raw_i2i_index": int(raw_idx),
            "final_fused_index": int(final_idx),
            "raw_i2i_file": raw_path or "reconstructed_from_final_blend_metadata",
            "final_fused_file": final_path,
        })
    return rows


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    final = _load_npz(args.final_augmentations)
    raw = _load_npz(args.raw_augmentations) if args.raw_augmentations else None

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

    final_source = np.asarray(final["source_index"], dtype=np.int64)
    valid = (final_source >= 0) & (final_source < len(dataset.data["x"]))
    if not np.all(valid):
        raise ValueError(f"source_index invalide pour {int(np.sum(~valid))} images finales.")

    all_source_images = np.asarray(dataset.data["x"][final_source], dtype=np.float32)
    if raw is not None:
        raw_to_final = _map_final_to_raw_indices(raw, final)
        all_raw_images = np.asarray(raw["x"][raw_to_final], dtype=np.float32)
    else:
        raw_to_final = _raw_indices_from_final_metadata(final)
        all_raw_images = _reconstruct_raw_i2i_from_final_blend(all_source_images, final)
    all_final_images = np.asarray(final["x"], dtype=np.float32)

    if args.example_indices:
        final_rows = np.asarray([int(value) for value in args.example_indices.split(",") if value.strip()], dtype=np.int64)
        if np.any((final_rows < 0) | (final_rows >= len(all_final_images))):
            raise ValueError("--example_indices contient un indice hors bornes.")
    else:
        final_rows = _select_final_rows(
            all_source_images,
            all_final_images,
            max_examples=args.max_examples,
            seed=args.seed,
            selection=args.selection,
        )

    source_images = all_source_images[final_rows]
    raw_images = all_raw_images[final_rows]
    final_images = all_final_images[final_rows]
    source_indices = final_source[final_rows]
    raw_indices = raw_to_final[final_rows]

    contact_path = os.path.join(output_dir, args.output_name)
    _plot_stage_contact_sheet(
        source_images,
        raw_images,
        final_images,
        source_indices,
        raw_indices,
        final_rows,
        contact_path,
        title=args.title,
    )
    write_rows_csv(
        os.path.join(output_dir, "i2i_stage_examples.csv"),
        _example_rows(source_indices, raw_indices, final_rows, args.raw_augmentations, args.final_augmentations),
    )

    with open(os.path.join(output_dir, "i2i_stage_comparison_report.md"), "w") as f:
        f.write(
            "# I2I Stage Comparison\n\n"
            f"- Raw i2i file: `{args.raw_augmentations}`\n"
            f"- Final fused file: `{args.final_augmentations}`\n"
            f"- Figure: `{args.output_name}`\n\n"
            "Columns:\n\n"
            "1. Source real image.\n"
            "2. Raw i2i output generated by the CFM, loaded from raw_augmentations or reconstructed from blend metadata.\n"
            "3. Final fused augmentation actually used after residual blending/post-processing.\n\n"
            "Important: the third column is not a delta map. It is an image in the same space as the source and generated image.\n"
        )
    logger.info("Comparaison source/i2i/fusion sauvegardee: %s", contact_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_augmentations", default=None)
    parser.add_argument("--final_augmentations", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="rgb_i2i_stage_contact_sheet.png")
    parser.add_argument("--title", default="Source reelle | sortie i2i brute | image finale apres fusion")
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="all")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--split_strategy", choices=["spatial", "marie_regular", "marie_strict"], default="spatial")
    parser.add_argument("--max_examples", type=int, default=8)
    parser.add_argument("--selection", choices=["mixed", "random", "largest_change"], default="mixed")
    parser.add_argument("--example_indices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    run(parser.parse_args())
