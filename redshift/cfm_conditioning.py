from typing import Dict, Optional

import numpy as np


MARIE_BAND_NAMES = ("us", "g", "r", "i", "z", "y")
CFM_CONDITION_DIMS = {
    "legacy7": 7,
    "marie_mags": 10,
}


def condition_dim(schema: str) -> int:
    if schema not in CFM_CONDITION_DIMS:
        raise ValueError(f"Schema de conditionnement CFM inconnu: {schema}")
    return CFM_CONDITION_DIMS[schema]


def condition_choices():
    return tuple(CFM_CONDITION_DIMS)


def normalized_mag(mag: np.ndarray) -> np.ndarray:
    return (np.asarray(mag, dtype=np.float32) - 22.0) / 2.0


def denormalized_mag(norm_mag: np.ndarray) -> np.ndarray:
    return np.asarray(norm_mag, dtype=np.float32) * 2.0 + 22.0


def _mags_marie(data: Dict[str, np.ndarray]) -> np.ndarray:
    if "mags_marie" in data:
        mags = np.asarray(data["mags_marie"], dtype=np.float32)
    else:
        mags = np.asarray(data["mags"], dtype=np.float32)
    if mags.ndim != 2 or mags.shape[1] < len(MARIE_BAND_NAMES):
        raise ValueError("mags_marie/mags doit avoir au moins 6 colonnes.")
    return np.nan_to_num(mags[:, : len(MARIE_BAND_NAMES)], nan=0.0, posinf=0.0, neginf=0.0)


def build_cfm_condition(data: Dict[str, np.ndarray], schema: str = "legacy7") -> np.ndarray:
    if schema == "legacy7":
        cond = np.asarray(data["cond"], dtype=np.float32)
        if cond.shape[1] != CFM_CONDITION_DIMS["legacy7"]:
            raise ValueError(f"Conditionnement legacy7 attendu Dim=7, reçu Dim={cond.shape[1]}.")
        return cond

    if schema == "marie_mags":
        n = len(data["z_true"])
        z_true = np.asarray(data["z_true"], dtype=np.float32).reshape(n, 1)
        mags_norm = normalized_mag(_mags_marie(data))
        ebv = np.asarray(data.get("ebv", np.zeros(n, dtype=np.float32)), dtype=np.float32).reshape(n, 1)
        re_norm = np.asarray(data["re_norm"], dtype=np.float32).reshape(n, 1)
        n_norm = np.asarray(data["n_norm"], dtype=np.float32).reshape(n, 1)
        return np.concatenate([z_true, mags_norm, ebv, re_norm, n_norm], axis=1).astype(np.float32)

    raise ValueError(f"Schema de conditionnement CFM inconnu: {schema}")


def build_cfm_photo_targets(data: Dict[str, np.ndarray], schema: str = "legacy7") -> Optional[np.ndarray]:
    if schema == "legacy7":
        return None
    if schema == "marie_mags":
        return _mags_marie(data).astype(np.float32)
    raise ValueError(f"Schema de conditionnement CFM inconnu: {schema}")


def photometric_features_from_cfm_condition(cond: np.ndarray, schema: str = "legacy7") -> Dict[str, np.ndarray]:
    cond = np.asarray(cond, dtype=np.float32)
    if schema == "legacy7":
        return {
            "mag_i": denormalized_mag(cond[:, 1]),
            "g_r": cond[:, 2],
            "r_i": cond[:, 3],
            "i_z": cond[:, 4],
        }
    if schema == "marie_mags":
        mags = denormalized_mag(cond[:, 1:7])
        return {
            "mag_i": mags[:, 3],
            "g_r": mags[:, 1] - mags[:, 2],
            "r_i": mags[:, 2] - mags[:, 3],
            "i_z": mags[:, 3] - mags[:, 4],
        }
    raise ValueError(f"Schema de conditionnement CFM inconnu: {schema}")
