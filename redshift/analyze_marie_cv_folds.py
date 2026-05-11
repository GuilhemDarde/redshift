import argparse
import io
import logging
import os
import tarfile
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analysis_utils import compute_regression_metrics, ensure_dir, save_metadata_npz, write_rows_csv
from analyze_treyer_figure7 import make_figure7_report, magnitude_support_rows
from config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_METADATA_KEYS = (
    "ra",
    "dec",
    "field",
    "id",
    "tract",
    "patch",
    "mask",
    "flag_field",
    "u",
    "us",
    "g",
    "r",
    "i",
    "z",
    "y",
    "j",
    "h",
    "ks",
    "zphot",
    "zspec",
    "zflag",
    "label",
    "survey",
    "class",
    "obj_type",
    "compact",
    "star_forming",
    "clean",
)


def _np_load_dict_from_bytes(data: bytes, label: str) -> Dict[str, np.ndarray]:
    obj = np.load(io.BytesIO(data), allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"{label} doit contenir un dict numpy sauvegarde avec allow_pickle=True.")
    return obj


def _np_load_dict_from_file(path: str) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"{path} doit contenir un dict numpy sauvegarde avec allow_pickle=True.")
    return obj


def _load_array_from_file(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def _member_suffix(member_name: str) -> str:
    parts = PurePosixPath(member_name).parts
    for i, part in enumerate(parts):
        if part.isdigit():
            return "/".join(parts[i:])
    return member_name


def _tar_member_by_suffix(tf: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [member for member in tf.getmembers() if member.isfile() and _member_suffix(member.name) == suffix]
    if not matches:
        raise FileNotFoundError(f"Membre introuvable dans le tar: {suffix}")
    if len(matches) > 1:
        names = ", ".join(member.name for member in matches[:3])
        raise ValueError(f"Suffixe ambigu dans le tar ({suffix}): {names}")
    return matches[0]


def _load_fold_dict_from_tar(tf: tarfile.TarFile, fold: int, result_file: str) -> Tuple[Dict[str, np.ndarray], int]:
    suffix = f"{fold}/{result_file}"
    member = _tar_member_by_suffix(tf, suffix)
    extracted = tf.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"Impossible de lire {member.name} dans le tar.")
    return _np_load_dict_from_bytes(extracted.read(), member.name), int(member.size)


def _load_fold_array_from_tar(tf: tarfile.TarFile, fold: int, filename: str) -> Tuple[Optional[np.ndarray], int]:
    try:
        member = _tar_member_by_suffix(tf, f"{fold}/{filename}")
    except FileNotFoundError:
        return None, 0
    extracted = tf.extractfile(member)
    if extracted is None:
        return None, int(member.size)
    return np.load(io.BytesIO(extracted.read()), allow_pickle=True), int(member.size)


def _as_vector(data: Dict[str, np.ndarray], key: str, fold: int) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Cle absente dans fold {fold}: {key}")
    array = np.asarray(data[key])
    if array.ndim == 0:
        raise ValueError(f"Cle {key} du fold {fold} est scalaire, vecteur attendu.")
    return array


def _safe_1d_array(value: object, n: int) -> Optional[np.ndarray]:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != n:
        return None
    if array.dtype.kind == "O":
        try:
            array = array.astype(str)
        except (TypeError, ValueError):
            return None
    return array


def _concat_rows(rows: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.concatenate(values, axis=0) for key, values in rows.items() if values}


def _fold_metrics_row(fold: int, z_true: np.ndarray, z_pred: np.ndarray, mag_i: np.ndarray, result_bytes: int, val_indices: Optional[np.ndarray]) -> Dict[str, object]:
    row = compute_regression_metrics(z_true, z_pred)
    row.update(
        {
            "fold": fold,
            "result_bytes": result_bytes,
            "n_val_indices": int(len(val_indices)) if val_indices is not None else "",
            "z_min": float(np.nanmin(z_true)) if len(z_true) else float("nan"),
            "z_max": float(np.nanmax(z_true)) if len(z_true) else float("nan"),
            "mag_i_min": float(np.nanmin(mag_i)) if len(mag_i) else float("nan"),
            "mag_i_max": float(np.nanmax(mag_i)) if len(mag_i) else float("nan"),
        }
    )
    return row


def _iter_fold_payloads(
    folds: Sequence[int],
    result_file: str,
    folds_dir: Optional[str] = None,
    folds_tar: Optional[str] = None,
) -> Iterable[Tuple[int, Dict[str, np.ndarray], Optional[np.ndarray], int]]:
    if (folds_dir is None) == (folds_tar is None):
        raise ValueError("Fournir exactement un des deux arguments: folds_dir ou folds_tar.")

    if folds_dir is not None:
        for fold in folds:
            fold_dir = os.path.join(folds_dir, str(fold))
            result_path = os.path.join(fold_dir, result_file)
            val_indices_path = os.path.join(fold_dir, "val_indices.npy")
            if not os.path.exists(result_path):
                raise FileNotFoundError(result_path)
            result_bytes = os.path.getsize(result_path)
            val_indices = _load_array_from_file(val_indices_path) if os.path.exists(val_indices_path) else None
            yield fold, _np_load_dict_from_file(result_path), val_indices, result_bytes
        return

    assert folds_tar is not None
    with tarfile.open(folds_tar, "r") as tf:
        for fold in folds:
            data, result_bytes = _load_fold_dict_from_tar(tf, fold, result_file)
            val_indices, _ = _load_fold_array_from_tar(tf, fold, "val_indices.npy")
            yield fold, data, val_indices, result_bytes


def build_marie_cv_concat(
    output_dir: str,
    folds: Sequence[int] = (0, 1, 2, 3, 4),
    result_file: str = "epoch_50_val_results.npy",
    folds_dir: Optional[str] = None,
    folds_tar: Optional[str] = None,
    metadata_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
    strict_unique_indices: bool = False,
) -> Tuple[str, str, List[Dict[str, object]]]:
    ensure_dir(output_dir)
    prediction_parts: Dict[str, List[np.ndarray]] = {"z_true": [], "z_pred": []}
    metadata_parts: Dict[str, List[np.ndarray]] = {"z_true": [], "mag_i": [], "fold_id": [], "fold_row": []}
    fold_rows: List[Dict[str, object]] = []
    all_val_indices: List[np.ndarray] = []

    for fold, data, val_indices, result_bytes in _iter_fold_payloads(
        folds=folds,
        result_file=result_file,
        folds_dir=folds_dir,
        folds_tar=folds_tar,
    ):
        z_true = _as_vector(data, "z_true", fold).astype(np.float64)
        z_pred = _as_vector(data, "z_pred", fold).astype(np.float64)
        mag_source_key = "mag_i" if "mag_i" in data else "i"
        mag_i = _as_vector(data, mag_source_key, fold).astype(np.float64)
        n = len(z_true)
        if len(z_pred) != n or len(mag_i) != n:
            raise ValueError(f"Longueurs incoherentes dans fold {fold}: z_true={n}, z_pred={len(z_pred)}, mag_i={len(mag_i)}")
        if val_indices is not None and len(val_indices) != n:
            raise ValueError(f"val_indices incoherent dans fold {fold}: {len(val_indices)} vs resultats {n}")

        prediction_parts["z_true"].append(z_true)
        prediction_parts["z_pred"].append(z_pred)
        metadata_parts["z_true"].append(z_true)
        metadata_parts["mag_i"].append(mag_i)
        metadata_parts["fold_id"].append(np.full(n, fold, dtype=np.int16))
        metadata_parts["fold_row"].append(np.arange(n, dtype=np.int64))
        if val_indices is not None:
            val_indices = np.asarray(val_indices).astype(np.int64)
            all_val_indices.append(val_indices)
            metadata_parts.setdefault("val_index", []).append(val_indices)
            prediction_parts.setdefault("val_index", []).append(val_indices)
        if "index" in data:
            marie_index = _safe_1d_array(data["index"], n)
            if marie_index is not None:
                metadata_parts.setdefault("marie_index", []).append(marie_index)

        for key in metadata_keys:
            if key in {"z_true", "mag_i"} or key not in data:
                continue
            array = _safe_1d_array(data[key], n)
            if array is not None:
                metadata_parts.setdefault(key, []).append(array)

        fold_rows.append(_fold_metrics_row(fold, z_true, z_pred, mag_i, result_bytes, val_indices))
        logger.info("Fold %s charge: n=%s | result=%.1f MiB", fold, n, result_bytes / 1024**2)

    predictions = _concat_rows(prediction_parts)
    metadata = _concat_rows(metadata_parts)
    n_total = len(predictions["z_true"])
    if len(predictions["z_pred"]) != n_total or len(metadata["mag_i"]) != n_total:
        raise ValueError("Concat finale incoherente.")
    predictions = {key: value for key, value in predictions.items() if len(value) == n_total}
    metadata = {key: value for key, value in metadata.items() if len(value) == n_total}

    if all_val_indices:
        concat_idx = np.concatenate(all_val_indices)
        unique_count = len(np.unique(concat_idx))
        if strict_unique_indices and unique_count != len(concat_idx):
            raise ValueError(f"val_indices non uniques: {unique_count}/{len(concat_idx)}")
        fold_rows.append(
            {
                "fold": "concat",
                "n": n_total,
                "unique_val_indices": int(unique_count),
                "duplicate_val_indices": int(len(concat_idx) - unique_count),
            }
        )

    predictions_path = os.path.join(output_dir, "predictions_marie_cv_concat.npz")
    metadata_path = os.path.join(output_dir, "metadata_marie_cv_concat.npz")
    np.savez(predictions_path, **predictions)
    save_metadata_npz(metadata_path, metadata)
    write_rows_csv(os.path.join(output_dir, "metrics_by_fold.csv"), fold_rows)
    logger.info("Concat Marie CV sauvegardee: n=%s -> %s", n_total, output_dir)
    return predictions_path, metadata_path, fold_rows


def run(args: argparse.Namespace) -> None:
    folds = tuple(args.folds)
    predictions_path, metadata_path, _ = build_marie_cv_concat(
        output_dir=args.output_dir,
        folds=folds,
        result_file=args.result_file,
        folds_dir=args.folds_dir,
        folds_tar=args.folds_tar,
        strict_unique_indices=args.strict_unique_indices,
    )

    make_figure7_report(
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        output_dir=args.output_dir,
        z_bins=args.z_bins,
        mag_i_min=args.mag_i_min,
        mag_i_max=args.mag_i_max,
        mag_i_bins=args.mag_i_bins,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    metadata = np.load(metadata_path, allow_pickle=False)
    predictions = np.load(predictions_path, allow_pickle=False)
    support_rows = magnitude_support_rows(
        metadata["mag_i"],
        metadata["mag_i"],
        predictions["z_true"],
        predictions["z_pred"],
        np.linspace(args.mag_i_min, args.mag_i_max, args.mag_i_bins + 1),
        quantile=args.mag_support_quantile,
    )
    for row in support_rows:
        row["support_reference"] = "all_cv_val_mag_i"
    write_rows_csv(os.path.join(args.output_dir, "metrics_by_mag_support_cv.csv"), support_rows)
    logger.info("Analyse Marie CV terminee: %s", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatene les 5 folds Marie/Treyer COSMOS UD et reproduit une analyse Figure 7-like.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folds_dir", type=str, help="Dossier extrait contenant les sous-dossiers 0..4.")
    source.add_argument("--folds_tar", type=str, help="Archive tar contenant les sous-dossiers 0..4, analysee sans extraction.")
    parser.add_argument("--output_dir", type=str, default=CONFIG.exp_path("marie_cv_concat"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--result_file", type=str, default="epoch_50_val_results.npy")
    parser.add_argument("--z_bins", type=int, default=20)
    parser.add_argument("--mag_i_min", type=float, default=CONFIG.I_MIN)
    parser.add_argument("--mag_i_max", type=float, default=CONFIG.I_MAX)
    parser.add_argument("--mag_i_bins", type=int, default=14)
    parser.add_argument("--mag_support_quantile", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--strict_unique_indices", action="store_true")
    run(parser.parse_args())
