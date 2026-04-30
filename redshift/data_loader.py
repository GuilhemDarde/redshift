import glob
import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from torch.utils.data import DataLoader, Dataset, Subset

from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CosmosDataset(Dataset):
    '''
    actions : Charge les cubes spectraux COSMOS, croise les coordonnées spatiales avec le catalogue morphologique FITS, normalise les données et construit le tenseur de conditionnement (Dim=7).
    inputs : data_dir (str), morpho_path (str)
    appels : glob.glob, os.path.join, self._load_morpho_catalog, self._load_and_process
    outputs : Instance de CosmosDataset
    '''
    def __init__(self, data_dir: str = CONFIG.DATA_PATH, morpho_path: str = CONFIG.MORPHO_PATH) -> None:
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.morpho_path = morpho_path
        self.morpho_data = self._load_morpho_catalog()
        self.data = self._load_and_process()

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

        all_x, all_cond_base, all_re, all_n, all_ra = [], [], [], [], []
        
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
                    names = info.dtype.names

                    z_key = 'z_spec' if 'z_spec' in names else 'zspec'
                    ra_key = 'ra' if 'ra' in names else 'ra_zs'
                    dec_key = 'dec' if 'dec' in names else 'dec_zs'
                    
                    if 'i' not in names or z_key not in names or ra_key not in names or dec_key not in names:
                        continue

                    mask_base = (info['i'] >= CONFIG.I_MIN) & (info['i'] <= CONFIG.I_MAX) & \
                                (info[z_key] > 0.001) & (info[z_key] <= CONFIG.Z_MAX)

                    if 'flag' in raw.files:
                        flags = raw['flag'][:, CONFIG.CHANNELS]
                        mask_base = mask_base & (np.sum(flags, axis=1) == 0)

                    if np.sum(mask_base) == 0:
                        continue

                    cube_valid = cube[mask_base][:, :, :, CONFIG.CHANNELS]
                    info_valid = info[mask_base]
                    
                    batch_coords = SkyCoord(ra=info_valid[ra_key]*u.deg, dec=info_valid[dec_key]*u.deg)
                    idx, d2d, _ = batch_coords.match_to_catalog_sky(morpho_coords)
                    
                    match_mask = d2d < (1.0 * u.arcsec)
                    
                    if np.sum(match_mask) == 0:
                        continue

                    cube_matched = cube_valid[match_mask]
                    if CONFIG.ASINH_NORM:
                        cube_matched = np.arcsinh(cube_matched)

                    info_matched = info_valid[match_mask]
                    matched_idx = idx[match_mask]

                    z_spec = info_matched[z_key]
                    mag_i = info_matched['i']

                    g = info_matched['g'] if 'g' in names else mag_i
                    r = info_matched['r'] if 'r' in names else mag_i
                    z_phot = info_matched['z'] if 'z' in names else mag_i

                    c1 = g - r
                    c2 = r - mag_i
                    c3 = mag_i - z_phot

                    cond_base = np.stack([
                        z_spec,
                        (mag_i - 22.0) / 2.0,
                        c1, c2, c3
                    ], axis=1)

                    all_x.append(cube_matched)
                    all_cond_base.append(cond_base)
                    all_re.append(morpho_re[matched_idx])
                    all_n.append(morpho_n[matched_idx])
                    all_ra.append(info_matched[ra_key])

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
            'ra': ra_concat
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

def get_dataloaders(batch_size: int = CONFIG.BATCH_SIZE, num_workers: int = CONFIG.NUM_WORKERS) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    actions : Construit les DataLoaders en appliquant un partitionnement spatial strict sur les données cross-matchées.
    inputs : Aucun
    appels : CosmosDataset, np.argsort, Subset, DataLoader
    outputs : Tuple contenant les DataLoaders de Train, Val et Test (Tuple[DataLoader, DataLoader, DataLoader])
    '''
    full_ds = CosmosDataset(CONFIG.DATA_PATH, morpho_path=CONFIG.MORPHO_PATH)
    ra_values = full_ds.data['ra']
    sorted_indices = np.argsort(ra_values)

    total = len(full_ds)
    train_size = int(0.80 * total)
    val_size = int(0.10 * total)

    train_idx = sorted_indices[:train_size]
    val_idx = sorted_indices[train_size : train_size + val_size]
    test_idx = sorted_indices[train_size + val_size :]

    logger.info(f"Split Spatial RA : Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
