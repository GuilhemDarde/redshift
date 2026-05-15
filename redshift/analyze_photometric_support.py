import argparse
import os
from typing import Dict, List

import numpy as np

from analysis_utils import compute_regression_metrics, ensure_dir, load_metadata, write_rows_csv
from density_utils import low_support_by_radius_mask, standardized_knn_radius


def _features_from_metadata(metadata: Dict[str, np.ndarray]) -> np.ndarray:
    required = ["mag_i", "mag_g", "mag_r", "mag_z"]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"Metadata incomplet pour le support photometrique. Cles manquantes: {missing}")
    mag_i = np.asarray(metadata["mag_i"], dtype=np.float64)
    g_r = np.asarray(metadata["mag_g"], dtype=np.float64) - np.asarray(metadata["mag_r"], dtype=np.float64)
    r_i = np.asarray(metadata["mag_r"], dtype=np.float64) - mag_i
    i_z = mag_i - np.asarray(metadata["mag_z"], dtype=np.float64)
    return np.column_stack([mag_i, g_r, r_i, i_z])


def _split_indices(metadata: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "split" not in metadata:
        raise KeyError("Metadata sans colonne split. Impossible de retrouver train/test sans fuite.")
    split = np.asarray(metadata["split"]).astype(str)
    return {
        "train": np.where(split == "train")[0].astype(np.int64),
        "val": np.where(split == "val")[0].astype(np.int64),
        "test": np.where(split == "test")[0].astype(np.int64),
    }


def _prediction_indices(pred: Dict[str, np.ndarray], split_indices: Dict[str, np.ndarray]) -> np.ndarray:
    if "test_indices" in pred:
        return np.asarray(pred["test_indices"], dtype=np.int64)[: len(pred["z_true"])]
    return split_indices["test"][: len(pred["z_true"])]


def analyze(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    pred_npz = np.load(args.predictions, allow_pickle=False)
    pred = {key: pred_npz[key] for key in pred_npz.files}
    metadata = load_metadata(args.metadata)
    split_indices = _split_indices(metadata)
    eval_indices = _prediction_indices(pred, split_indices)

    features = _features_from_metadata(metadata)
    radius = standardized_knn_radius(features, split_indices["train"], k=args.k)
    low_all, threshold = low_support_by_radius_mask(radius, split_indices["train"], fraction=args.low_fraction)
    low_eval = low_all[eval_indices]
    valid_eval = np.isfinite(radius[eval_indices])
    mag_i_eval = np.asarray(metadata["mag_i"], dtype=np.float64)[eval_indices]
    faint = mag_i_eval >= args.faint_mag_threshold

    rows: List[Dict[str, object]] = []
    for subset, mask in [
        ("global", np.ones(len(pred["z_true"]), dtype=bool)),
        ("low_photometric_support", low_eval),
        ("normal_photometric_support", valid_eval & ~low_eval),
        ("faint_and_low_photometric_support", faint & low_eval),
        ("faint_mag", faint),
        ("normal_faint_mag", ~faint),
    ]:
        row = compute_regression_metrics(pred["z_true"][mask], pred["z_pred"][mask])
        row.update({"subset": subset})
        rows.append(row)
    write_rows_csv(os.path.join(output_dir, "metrics_photometric_support.csv"), rows)
    write_rows_csv(os.path.join(output_dir, "photometric_support_definition.csv"), [{
        "feature_space": "mag_i,g-r,r-i,i-z",
        "k": args.k,
        "low_support_fraction": args.low_fraction,
        "radius_threshold": threshold,
        "train_low_support_n": int(np.sum(low_all[split_indices["train"]])),
        "eval_low_support_n": int(np.sum(low_eval)),
        "eval_n": int(len(eval_indices)),
    }])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--low_fraction", type=float, default=0.20)
    parser.add_argument("--faint_mag_threshold", type=float, default=23.5)
    analyze(parser.parse_args())
