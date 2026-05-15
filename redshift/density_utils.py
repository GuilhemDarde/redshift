from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback utile sur environnements minimaux
    cKDTree = None


def projected_radec_coordinates(ra: np.ndarray, dec: np.ndarray, ra_ref: Optional[float] = None, dec_ref: Optional[float] = None) -> np.ndarray:
    '''
    actions : Projette RA/DEC en coordonnées locales planes robustes au wrap-around en RA.
    inputs : ra (np.ndarray), dec (np.ndarray), ra_ref (float), dec_ref (float)
    appels : np.asarray, np.nanmedian, np.mod, np.cos, np.deg2rad, np.column_stack
    outputs : np.ndarray
    '''
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if ra_ref is None:
        ra_ref = float(np.nanmedian(ra))
    if dec_ref is None:
        dec_ref = float(np.nanmedian(dec))

    dra = (np.mod(ra - ra_ref + 180.0, 360.0) - 180.0) * np.cos(np.deg2rad(dec_ref))
    ddec = dec - dec_ref
    return np.column_stack([dra, ddec])


def _query_knn_distances(train_coords: np.ndarray, query_coords: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if cKDTree is not None:
        tree = cKDTree(train_coords)
        dist, idx = tree.query(query_coords, k=k)
        return np.asarray(dist), np.asarray(idx)

    diff = query_coords[:, None, :] - train_coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    order = np.argsort(dist_matrix, axis=1)[:, :k]
    dist = np.take_along_axis(dist_matrix, order, axis=1)
    return dist, order


def compute_train_knn_density(
    ra: np.ndarray,
    dec: np.ndarray,
    train_indices: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    actions : Calcule une densité locale en deg^-2 pour tous les objets, référencée aux voisins du split train.
    inputs : ra (np.ndarray), dec (np.ndarray), train_indices (np.ndarray), k (int)
    appels : projected_radec_coordinates, _query_knn_distances
    outputs : Tuple[np.ndarray, np.ndarray]
    '''
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.size < 2:
        n = len(ra)
        return np.zeros(n, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

    ra_train = np.asarray(ra, dtype=np.float64)[train_indices]
    dec_train = np.asarray(dec, dtype=np.float64)[train_indices]
    ra_ref = float(np.nanmedian(ra_train))
    dec_ref = float(np.nanmedian(dec_train))
    coords = projected_radec_coordinates(ra, dec, ra_ref=ra_ref, dec_ref=dec_ref)
    train_coords = coords[train_indices]

    k_eff = int(max(1, min(k, train_indices.size - 1)))
    dist, idx = _query_knn_distances(train_coords, coords, k=min(k_eff + 1, train_indices.size))
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    source_pos = np.full(len(ra), -1, dtype=np.int64)
    source_pos[train_indices] = np.arange(train_indices.size)
    kth_radius = np.empty(len(ra), dtype=np.float64)

    for row in range(len(ra)):
        row_dist = dist[row]
        row_idx = idx[row]
        pos = source_pos[row]
        if pos >= 0:
            row_dist = row_dist[row_idx != pos]
        if row_dist.size == 0:
            kth_radius[row] = np.nan
        else:
            kth_radius[row] = row_dist[min(k_eff - 1, row_dist.size - 1)]

    safe_radius = np.clip(kth_radius, 1e-12, None)
    density = k_eff / (np.pi * safe_radius * safe_radius)
    density[~np.isfinite(kth_radius)] = 0.0
    return density, kth_radius


def compute_catalog_knn_density(
    ra: np.ndarray,
    dec: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    actions : Calcule une densité locale leave-one-out dans le catalogue complet.
    inputs : ra (np.ndarray), dec (np.ndarray), k (int)
    appels : projected_radec_coordinates, _query_knn_distances
    outputs : Tuple[np.ndarray, np.ndarray]
    '''
    n = len(ra)
    if n < 2:
        return np.zeros(n, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

    coords = projected_radec_coordinates(ra, dec)
    k_eff = int(max(1, min(k, n - 1)))
    dist, idx = _query_knn_distances(coords, coords, k=min(k_eff + 1, n))
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    kth_radius = np.empty(n, dtype=np.float64)
    for row in range(n):
        row_dist = dist[row][idx[row] != row]
        if row_dist.size == 0:
            kth_radius[row] = np.nan
        else:
            kth_radius[row] = row_dist[min(k_eff - 1, row_dist.size - 1)]

    safe_radius = np.clip(kth_radius, 1e-12, None)
    density = k_eff / (np.pi * safe_radius * safe_radius)
    density[~np.isfinite(kth_radius)] = 0.0
    return density, kth_radius


def standardized_knn_radius(
    features: np.ndarray,
    reference_indices: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    '''
    actions : Calcule la distance au k-ieme voisin train dans un espace de features standardise sur le train.
    inputs : features (np.ndarray), reference_indices (np.ndarray), k (int)
    appels : np.nanmean, np.nanstd, _query_knn_distances
    outputs : np.ndarray
    '''
    features = np.asarray(features, dtype=np.float64)
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError("features doit etre une matrice 2D.")
    if reference_indices.size < 2:
        return np.full(features.shape[0], np.nan, dtype=np.float64)

    ref = features[reference_indices]
    mean = np.nanmean(ref, axis=0)
    std = np.nanstd(ref, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    normalized = (features - mean[None, :]) / std[None, :]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    ref_normalized = normalized[reference_indices]

    k_eff = int(max(1, min(k, reference_indices.size - 1)))
    dist, idx = _query_knn_distances(ref_normalized, normalized, k=min(k_eff + 1, reference_indices.size))
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    source_pos = np.full(features.shape[0], -1, dtype=np.int64)
    source_pos[reference_indices] = np.arange(reference_indices.size)
    radius = np.empty(features.shape[0], dtype=np.float64)
    for row in range(features.shape[0]):
        row_dist = dist[row]
        row_idx = idx[row]
        pos = source_pos[row]
        if pos >= 0:
            row_dist = row_dist[row_idx != pos]
        radius[row] = row_dist[min(k_eff - 1, row_dist.size - 1)] if row_dist.size else np.nan
    return radius


def low_support_by_radius_mask(radius: np.ndarray, reference_indices: np.ndarray, fraction: float = 0.20) -> Tuple[np.ndarray, float]:
    '''
    actions : Selectionne les objets les plus eloignes du support train selon une fraction haute de rayon kNN.
    inputs : radius (np.ndarray), reference_indices (np.ndarray), fraction (float)
    appels : np.quantile, np.isfinite
    outputs : Tuple[np.ndarray, float]
    '''
    radius = np.asarray(radius, dtype=np.float64)
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction doit etre dans ]0, 1[.")

    ref_values = radius[reference_indices]
    ref_values = ref_values[np.isfinite(ref_values)]
    if ref_values.size == 0:
        threshold = float("nan")
        return np.zeros_like(radius, dtype=bool), threshold

    threshold = float(np.quantile(ref_values, 1.0 - fraction))
    return np.isfinite(radius) & (radius >= threshold), threshold


def low_density_mask(density: np.ndarray, reference_indices: np.ndarray, quantile: float = 0.20) -> Tuple[np.ndarray, float]:
    '''
    actions : Sélectionne les objets sous un quantile de densité mesuré sur une population de référence.
    inputs : density (np.ndarray), reference_indices (np.ndarray), quantile (float)
    appels : np.asarray, np.quantile, np.isfinite
    outputs : Tuple[np.ndarray, float]
    '''
    density = np.asarray(density, dtype=np.float64)
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile doit etre dans ]0, 1[.")

    ref_values = density[reference_indices]
    ref_values = ref_values[np.isfinite(ref_values)]
    if ref_values.size == 0:
        threshold = float("nan")
        return np.zeros_like(density, dtype=bool), threshold

    threshold = float(np.quantile(ref_values, quantile))
    return np.isfinite(density) & (density <= threshold), threshold
