import glob
import logging
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from torch.utils.data import DataLoader, Dataset, Subset

from analysis_utils import (
    apply_region_mask,
    assert_split_integrity,
    compute_marie_regular_cv_indices,
    compute_marie_strict_cv_indices,
    compute_split_indices,
    save_metadata_csv,
    save_metadata_npz,
    split_labels,
)
from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _decode_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O", "U"}:
        return np.asarray([
            v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
            for v in arr
        ])
    return arr


def normalize_field_label(value: object) -> str:
    '''
    actions : Normalise les libelles de champ pour reconnaitre COSMOS Ultra Deep/UDF.
    inputs : value (object)
    appels : str, re.sub
    outputs : str
    '''
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        return "unknown"
    if "cosmos" in text and ("ud" in text or "udf" in text or "ultra" in text or "deep" in text):
        return "cosmos_ud"
    if text in {"ud", "udf", "ultradeep", "ultra_deep", "deep"}:
        return "cosmos_ud"
    return text


def field_label_from_filename(file_path: str) -> str:
    '''
    actions : Déduit la couche COSMOS D/UD depuis le nom du fichier source.
    inputs : file_path (str)
    appels : os.path.basename, str.lower
    outputs : str
    '''
    name = os.path.basename(file_path).lower()
    stem = name[:-4] if name.endswith(".npz") else name
    if stem.endswith("_ud") or "_ud_" in stem:
        return "cosmos_ud"
    if stem.endswith("_d") or "_d_" in stem:
        return "cosmos_deep"
    return "unknown"


def field_labels_from_source_file(source_file: np.ndarray) -> np.ndarray:
    return np.asarray([field_label_from_filename(str(path)) for path in source_file], dtype="<U32")


def infer_field_labels(info: np.ndarray, names: Tuple[str, ...], file_path: str) -> np.ndarray:
    '''
    actions : Tente de deduire le champ observationnel depuis les colonnes info ou le nom de fichier.
    inputs : info (np.ndarray), names (Tuple[str]), file_path (str)
    appels : normalize_field_label, _decode_array
    outputs : np.ndarray[str]
    '''
    n = len(info)
    file_label = field_label_from_filename(file_path)
    if file_label != "unknown":
        return np.full(n, file_label, dtype="<U32")

    bool_candidates = ("is_cosmos_ud", "cosmos_ud", "is_ud", "ud", "is_udf", "udf", "ultradeep", "ultra_deep")
    for key in bool_candidates:
        if key in names:
            values = np.asarray(info[key])
            if values.dtype.kind in {"b", "i", "u", "f"}:
                return np.where(values.astype(float) > 0, "cosmos_ud", "other").astype("<U32")

    text_candidates = ("field", "field_name", "survey", "area", "region", "layer", "deepfield", "deep_field")
    for key in text_candidates:
        if key in names:
            values = _decode_array(info[key])
            if values.dtype.kind in {"S", "O", "U"}:
                return np.asarray([normalize_field_label(v) for v in values], dtype="<U32")

    return np.full(n, "unknown", dtype="<U32")


def _filter_first_axis(data: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(mask)
    filtered = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            filtered[key] = arr[mask]
        else:
            filtered[key] = value
    return filtered


class CosmosDataset(Dataset):
    '''
    actions : Charge les cubes spectraux COSMOS, croise les coordonnées spatiales avec le catalogue morphologique FITS, normalise les données et construit le tenseur de conditionnement (Dim=7).
    inputs : data_dir (str), morpho_path (str)
    appels : glob.glob, os.path.join, self._load_morpho_catalog, self._load_and_process
    outputs : Instance de CosmosDataset
    '''
    def __init__(
        self,
        data_dir: str = CONFIG.DATA_PATH,
        morpho_path: str = CONFIG.MORPHO_PATH,
        region: str = "all",
        field: str = "all",
        sample_filter: str = "spec",
        max_files: Optional[int] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_files is not None:
            self.files = self.files[:max_files]
        self.data_dir = data_dir
        self.morpho_path = morpho_path
        self.region = region
        self.field = field
        self.sample_filter = sample_filter
        self.cache_path = cache_path if cache_path is not None else CONFIG.PROCESSED_DATASET_PATH

        if self.cache_path and max_files is None and os.path.exists(self.cache_path):
            self.data = self._apply_dataset_filters(self._load_processed_cache(self.cache_path))
            return

        self.morpho_data = self._load_morpho_catalog()
        processed = self._load_and_process()
        filtered = self._apply_dataset_filters(processed)
        self.data = filtered
        if self.cache_path and max_files is None:
            self._save_processed_cache(self.cache_path)

    def _load_processed_cache(self, cache_path: str) -> Dict[str, np.ndarray]:
        logger.info("Chargement du dataset pretraite depuis le cache : %s", cache_path)
        with np.load(cache_path, allow_pickle=False) as cached:
            data = {key: cached[key] for key in cached.files if not key.startswith("__")}

            cache_region = str(cached["__region"].item()) if "__region" in cached.files else "unknown"
            if cache_region != self.region:
                raise ValueError(
                    f"Cache dataset construit pour region={cache_region}, mais region={self.region}. "
                    "Utilisez un autre COSMOS_PROCESSED_DATASET_PATH ou supprimez le cache."
                )

        required = {"x", "cond", "ra", "dec", "z_true", "mag_i", "mags", "flags", "re_norm", "n_norm"}
        missing = sorted(required - set(data))
        if missing:
            raise KeyError(f"Cache dataset incomplet {cache_path}. Cles manquantes: {missing}")
        if "label_type" not in data:
            data["label_type"] = np.full(len(data["z_true"]), "spec", dtype="<U16")
        if "ebv" not in data:
            data["ebv"] = np.zeros(len(data["z_true"]), dtype=np.float32)
        if "mags_marie" not in data:
            data["mags_marie"] = np.asarray(data["mags"], dtype=np.float32)
        logger.info("Dataset charge depuis cache : %s objets. Conditionnement Dim=%s.", len(data["cond"]), data["cond"].shape[1])
        return data

    def _apply_dataset_filters(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n = len(data["z_true"])
        mask = np.ones(n, dtype=bool)

        if self.sample_filter not in {"all", "spec"}:
            raise ValueError("--sample_filter doit valoir all ou spec.")
        if self.sample_filter == "spec":
            labels = np.asarray(data.get("label_type", np.full(n, "spec", dtype="<U16"))).astype(str)
            mask &= labels == "spec"

        if self.field not in {"all", "", None}:
            if "source_file" in data:
                labels = field_labels_from_source_file(data["source_file"])
            elif "field" in data:
                labels = np.asarray([normalize_field_label(v) for v in data["field"]], dtype="<U32")
            else:
                raise KeyError(
                    "Le cache ne contient ni source_file ni field. Pour utiliser --field cosmos_ud, "
                    "reconstruisez un cache depuis les fichiers NPZ sources contenant l'information de champ."
                )
            wanted = normalize_field_label(self.field)
            mask &= labels == wanted
            if not np.any(mask):
                available = sorted(set(labels.tolist()))
                raise ValueError(f"Aucun objet pour field={wanted}. Champs disponibles: {available[:20]}")

        filtered = _filter_first_axis(data, mask)
        logger.info(
            "Filtres dataset: sample_filter=%s field=%s -> %s/%s objets.",
            self.sample_filter,
            self.field,
            len(filtered["z_true"]),
            n,
        )
        return filtered

    def _save_processed_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        logger.info("Sauvegarde du dataset pretraite dans le cache : %s", cache_path)
        np.savez(
            cache_path,
            **self.data,
            __region=np.array(self.region),
            __data_dir=np.array(self.data_dir),
            __morpho_path=np.array(self.morpho_path),
            __n_files=np.array(len(self.files)),
        )

    def _load_morpho_catalog(self) -> Dict[str, np.ndarray]:
        '''
        actions : Charge le catalogue FITS, auto-détecte les alias de colonnes (souvent altérés par l'export VizieR) et initialise l'objet astropy SkyCoord.
        inputs : Aucun
        appels : Table.read, SkyCoord
        outputs : Dictionnaire contenant les coordonnées astrométriques et les variables morphologiques (Dict[str, np.ndarray])
        '''
        if not os.path.exists(self.morpho_path):
            raise FileNotFoundError(f"Catalogue morphologique introuvable : {self.morpho_path}")
        
        logger.info(f"Chargement du catalogue morphologique FITS : {self.morpho_path}")
        cat = Table.read(self.morpho_path, format='fits')
        
        names = cat.colnames
        
        ra_key = next((k for k in names if k.upper() in ['RAJ2000', 'RA', 'RA_J2000']), None)
        dec_key = next((k for k in names if k.upper() in ['DEJ2000', 'DEC', 'DEC_J2000']), None)
        re_key = next((k for k in names if k.upper() in ['RE.G1', 'RE_G1', 'RE_GALFIT_HI', 'RE', 'RADIUS']), None)
        n_key = next((k for k in names if k.upper() in ['N.G1', 'N_G1', 'N_GALFIT_HI', 'N_SERSIC', 'N']), None)
        
        if not all([ra_key, dec_key, re_key, n_key]):
            raise KeyError(f"Colonnes morphologiques introuvables. Colonnes disponibles dans le FITS : {names}")
            
        logger.info(f"Mapping des colonnes -> RA: {ra_key}, DEC: {dec_key}, Re: {re_key}, n: {n_key}")
        
        ra = cat[ra_key].data
        dec = cat[dec_key].data
        re = cat[re_key].data
        n_sersic = cat[n_key].data
        
        valid_mask = (re > 0) & (n_sersic > 0) & (~np.isnan(re)) & (~np.isnan(n_sersic))
        
        coords = SkyCoord(ra=ra[valid_mask]*u.deg, dec=dec[valid_mask]*u.deg)
        
        return {
            'coords': coords,
            're': re[valid_mask],
            'n': n_sersic[valid_mask]
        }
        
    def _load_and_process(self) -> Dict[str, np.ndarray]:
        '''
        actions : Filtre les fichiers .npz, effectue un cross-match par plus proche voisin avec le catalogue morphologique, applique les normalisations et concatène les vecteurs.
        inputs : Aucun
        appels : np.load, np.arcsinh, SkyCoord, SkyCoord.match_to_catalog_sky, np.stack, np.concatenate, np.transpose, np.log10, np.mean, np.std, np.clip
        outputs : Dictionnaire contenant les tenseurs 'x', 'cond' et 'ra' (Dict[str, np.ndarray])
        '''
        if not self.files:
            raise FileNotFoundError(f"Aucun fichier .npz trouvé dans {CONFIG.DATA_PATH}")

        logger.info(f"Traitement et Cross-Matching de {len(self.files)} fichiers...")

        all_x, all_cond_base, all_re, all_n, all_ra, all_dec, all_flags, all_mags = [], [], [], [], [], [], [], []
        all_mags_marie, all_ebv = [], []
        all_field, all_label_type, all_source_file = [], [], []
        
        morpho_coords = self.morpho_data['coords']
        morpho_re = self.morpho_data['re']
        morpho_n = self.morpho_data['n']

        for f in self.files:
            try:
                with np.load(f) as raw:
                    if 'info' not in raw or 'cube' not in raw:
                        continue
                    
                    info = raw['info']
                    cube = raw['cube']

                    if info.dtype.names:
                        info.dtype.names = tuple([name.lower() for name in info.dtype.names])
                    else:
                        logger.warning("Champ info non structure dans %s, fichier ignore.", f)
                        continue
                    names = info.dtype.names

                    z_key = 'z_spec' if 'z_spec' in names else 'zspec'
                    ra_key = 'ra' if 'ra' in names else 'ra_zs'
                    dec_key = 'dec' if 'dec' in names else 'dec_zs'
                    
                    if 'i' not in names or z_key not in names or ra_key not in names or dec_key not in names:
                        continue

                    mask_base = (info['i'] >= CONFIG.I_MIN) & (info['i'] <= CONFIG.I_MAX) & \
                                (info[z_key] > 0.001) & (info[z_key] <= CONFIG.Z_MAX)

                    flags_all = None
                    if 'flag' in raw.files:
                        flags_all = raw['flag'][:, CONFIG.CHANNELS]
                        mask_base = mask_base & (np.sum(flags_all, axis=1) == 0)

                    if np.sum(mask_base) == 0:
                        continue

                    cube_valid = cube[mask_base][:, :, :, CONFIG.CHANNELS]
                    info_valid = info[mask_base]
                    field_valid = infer_field_labels(info_valid, names, f)
                    label_type_valid = np.full(len(info_valid), "spec", dtype="<U16")
                    source_file_valid = np.full(len(info_valid), os.path.basename(f), dtype="<U256")
                    flags_valid = flags_all[mask_base] if flags_all is not None else np.full((len(info_valid), len(CONFIG.CHANNELS)), np.nan)
                    
                    batch_coords = SkyCoord(ra=info_valid[ra_key]*u.deg, dec=info_valid[dec_key]*u.deg)
                    idx, d2d, _ = batch_coords.match_to_catalog_sky(morpho_coords)
                    
                    match_mask = d2d < (1.0 * u.arcsec)
                    
                    if np.sum(match_mask) == 0:
                        continue

                    cube_matched = cube_valid[match_mask]
                    if CONFIG.ASINH_NORM:
                        cube_matched = np.arcsinh(cube_matched)

                    info_matched = info_valid[match_mask]
                    flags_matched = flags_valid[match_mask]
                    field_matched = field_valid[match_mask]
                    label_type_matched = label_type_valid[match_mask]
                    source_file_matched = source_file_valid[match_mask]
                    matched_idx = idx[match_mask]

                    z_spec = info_matched[z_key]
                    mag_i = info_matched['i']
                    ra_matched = info_matched[ra_key]
                    dec_matched = info_matched[dec_key]

                    g = info_matched['g'] if 'g' in names else mag_i
                    r = info_matched['r'] if 'r' in names else mag_i
                    z_phot = info_matched['z'] if 'z' in names else mag_i
                    mags = np.stack(
                        [info_matched[b] if b in names else np.full_like(mag_i, np.nan, dtype=np.float64) for b in CONFIG.BAND_NAMES],
                        axis=1,
                    )
                    marie_band_names = ["us", "g", "r", "i", "z", "y"]
                    mags_marie = np.stack(
                        [
                            info_matched["u"] if b == "us" and "us" not in names and "u" in names
                            else info_matched[b] if b in names
                            else np.full_like(mag_i, np.nan, dtype=np.float64)
                            for b in marie_band_names
                        ],
                        axis=1,
                    )
                    ebv = info_matched["ebv"] if "ebv" in names else np.zeros_like(mag_i, dtype=np.float64)

                    c1 = g - r
                    c2 = r - mag_i
                    c3 = mag_i - z_phot

                    cond_base = np.stack([
                        z_spec,
                        (mag_i - 22.0) / 2.0,
                        c1, c2, c3
                    ], axis=1)

                    metadata_for_region = {
                        "ra": np.asarray(ra_matched),
                        "dec": np.asarray(dec_matched),
                        "z_true": np.asarray(z_spec),
                    }
                    region_mask = apply_region_mask(metadata_for_region, self.region)
                    if np.sum(region_mask) == 0:
                        continue

                    all_x.append(cube_matched[region_mask])
                    all_cond_base.append(cond_base[region_mask])
                    all_re.append(morpho_re[matched_idx][region_mask])
                    all_n.append(morpho_n[matched_idx][region_mask])
                    all_ra.append(ra_matched[region_mask])
                    all_dec.append(dec_matched[region_mask])
                    all_flags.append(flags_matched[region_mask])
                    all_mags.append(mags[region_mask])
                    all_mags_marie.append(mags_marie[region_mask])
                    all_ebv.append(ebv[region_mask])
                    all_field.append(field_matched[region_mask])
                    all_label_type.append(label_type_matched[region_mask])
                    all_source_file.append(source_file_matched[region_mask])

            except Exception as e:
                logger.warning(f"Erreur de traitement sur le fichier {f} : {e}")
                continue

        if not all_x:
            raise ValueError("Le cross-matching a échoué pour tous les fichiers. Vérifiez la tolérance spatiale (1 arcsec) et les catalogues.")

        x_concat = np.concatenate(all_x, axis=0)
        cond_base_concat = np.concatenate(all_cond_base, axis=0)
        re_concat = np.concatenate(all_re, axis=0)
        n_concat = np.concatenate(all_n, axis=0)
        ra_concat = np.concatenate(all_ra, axis=0)
        dec_concat = np.concatenate(all_dec, axis=0)
        flags_concat = np.concatenate(all_flags, axis=0)
        mags_concat = np.concatenate(all_mags, axis=0)
        mags_marie_concat = np.concatenate(all_mags_marie, axis=0)
        ebv_concat = np.concatenate(all_ebv, axis=0)
        field_concat = np.concatenate(all_field, axis=0)
        label_type_concat = np.concatenate(all_label_type, axis=0)
        source_file_concat = np.concatenate(all_source_file, axis=0)

        log_re = np.log10(re_concat)
        mu_re = np.mean(log_re)
        sigma_re = np.std(log_re)
        r_norm = (log_re - mu_re) / (sigma_re + 1e-8)

        log_n = np.log10(n_concat)
        log_n_min = np.log10(0.5)
        log_n_max = np.log10(8.0)
        n_norm = 2.0 * (log_n - log_n_min) / (log_n_max - log_n_min) - 1.0
        n_norm = np.clip(n_norm, -1.0, 1.0)

        morpho_vec = np.stack([r_norm, n_norm], axis=1)
        cond_final = np.concatenate([cond_base_concat, morpho_vec], axis=1)

        x_concat = np.transpose(x_concat, (0, 3, 1, 2))
        
        data = {
            'x': x_concat,
            'cond': cond_final,
            'ra': ra_concat,
            'dec': dec_concat,
            'z_true': cond_base_concat[:, 0],
            'mag_i': cond_base_concat[:, 1] * 2.0 + 22.0,
            'mags': mags_concat,
            'mags_marie': mags_marie_concat,
            'ebv': ebv_concat,
            'flags': flags_concat,
            'field': field_concat,
            'label_type': label_type_concat,
            'source_file': source_file_concat,
            're_norm': r_norm,
            'n_norm': n_norm,
        }

        logger.info(f"Dataset chargé et matché : {len(data['cond'])} objets restants. Conditionnement Dim={cond_final.shape[1]}.")
        
        return data

    def __len__(self) -> int:
        '''
        actions : Retourne le nombre total d'échantillons validés dans le dataset après le cross-matching.
        inputs : Aucun
        appels : len
        outputs : Nombre d'échantillons (int)
        '''
        return len(self.data['cond'])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        actions : Récupère le tenseur d'image et le vecteur de conditionnement étendu (photométrie + morphologie).
        inputs : idx (int)
        appels : torch.tensor
        outputs : Tuple contenant le tenseur d'image et le tenseur de conditionnement (Tuple[torch.Tensor, torch.Tensor])
        '''
        x = torch.tensor(self.data['x'][idx], dtype=torch.float32)
        cond = torch.tensor(self.data['cond'][idx], dtype=torch.float32)
        return x, cond

def build_metadata(dataset: CosmosDataset, split_indices: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    metadata = {
        "ra": dataset.data["ra"],
        "dec": dataset.data["dec"],
        "z_true": dataset.data["z_true"],
        "mag_i": dataset.data["mag_i"],
        "re_norm": dataset.data["re_norm"],
        "n_norm": dataset.data["n_norm"],
        "ebv": dataset.data.get("ebv", np.zeros(len(dataset), dtype=np.float32)),
        "field": dataset.data.get("field", np.full(len(dataset), "unknown", dtype="<U32")),
        "label_type": dataset.data.get("label_type", np.full(len(dataset), "spec", dtype="<U16")),
    }
    mags = dataset.data["mags"]
    flags = dataset.data["flags"]
    for idx, band in enumerate(CONFIG.BAND_NAMES):
        metadata[f"mag_{band}"] = mags[:, idx]
        metadata[f"flag_{band}"] = flags[:, idx]
    if "mags_marie" in dataset.data:
        for idx, band in enumerate(["us", "g", "r", "i", "z", "y"]):
            metadata[f"mag_marie_{band}"] = dataset.data["mags_marie"][:, idx]
    if split_indices is not None:
        metadata["split"] = split_labels(len(dataset), split_indices)
    return metadata


def export_metadata(
    dataset: CosmosDataset,
    output_path: str,
    split_indices: Optional[Dict[str, np.ndarray]] = None,
    csv_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    metadata = build_metadata(dataset, split_indices=split_indices)
    save_metadata_npz(output_path, metadata)
    if csv_path is not None:
        save_metadata_csv(csv_path, metadata)
    return metadata


def get_dataset_and_splits(
    region: str = "all",
    field: str = "all",
    sample_filter: str = "spec",
    max_files: Optional[int] = None,
    n_folds: Optional[int] = None,
    fold_id: Optional[int] = None,
    cache_path: Optional[str] = None,
    split_strategy: str = "spatial",
) -> Tuple[CosmosDataset, Dict[str, np.ndarray]]:
    dataset = CosmosDataset(
        CONFIG.DATA_PATH,
        morpho_path=CONFIG.MORPHO_PATH,
        region=region,
        field=field,
        sample_filter=sample_filter,
        max_files=max_files,
        cache_path=cache_path,
    )
    if split_strategy == "spatial":
        split_indices = compute_split_indices(dataset.data["ra"], n_folds=n_folds, fold_id=fold_id)
        logger.info("Split strategy: spatial RA")
    elif split_strategy == "marie_regular":
        if fold_id is None:
            raise ValueError("split_strategy=marie_regular requiert --fold_id.")
        split_indices = compute_marie_regular_cv_indices(len(dataset), n_folds=n_folds or CONFIG.N_FOLDS, fold_id=fold_id, seed=42)
        logger.info(
            "Split strategy: Marie regular CV | fold=%s/%s | seed=42",
            fold_id,
            n_folds or CONFIG.N_FOLDS,
        )
    elif split_strategy == "marie_strict":
        if fold_id is None:
            raise ValueError("split_strategy=marie_strict requiert --fold_id.")
        split_indices = compute_marie_strict_cv_indices(len(dataset), n_folds=n_folds or CONFIG.N_FOLDS, fold_id=fold_id, seed=42)
        logger.info(
            "Split strategy: Marie strict CV | fold=%s/%s | val fold=%s | seed=42",
            fold_id,
            n_folds or CONFIG.N_FOLDS,
            (fold_id + 1) % (n_folds or CONFIG.N_FOLDS),
        )
    else:
        raise ValueError("split_strategy doit valoir spatial, marie_regular ou marie_strict.")
    assert_split_integrity(
        len(dataset),
        split_indices,
        allow_val_test_overlap=split_strategy == "marie_regular",
    )
    return dataset, split_indices


def get_dataloaders(
    batch_size: int = CONFIG.BATCH_SIZE,
    num_workers: int = CONFIG.NUM_WORKERS,
    region: str = "all",
    field: str = "all",
    sample_filter: str = "spec",
    max_files: Optional[int] = None,
    n_folds: Optional[int] = None,
    fold_id: Optional[int] = None,
    cache_path: Optional[str] = None,
    split_strategy: str = "spatial",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    actions : Construit les DataLoaders en appliquant le partitionnement demande.
    inputs : Aucun
    appels : CosmosDataset, np.argsort, Subset, DataLoader
    outputs : Tuple contenant les DataLoaders de Train, Val et Test (Tuple[DataLoader, DataLoader, DataLoader])
    '''
    full_ds, split_indices = get_dataset_and_splits(
        region=region,
        field=field,
        sample_filter=sample_filter,
        max_files=max_files,
        n_folds=n_folds,
        fold_id=fold_id,
        cache_path=cache_path,
        split_strategy=split_strategy,
    )
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    logger.info(
        "Split %s : Train=%s, Val=%s, Test=%s",
        split_strategy,
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
