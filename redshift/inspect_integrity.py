import glob
import logging
import os
from typing import Dict

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_matching_statistics(data_dir: str, morpho_path: str) -> Dict[str, int]:
    '''
    actions : Calcule les statistiques de filtrage et de cross-matching entre les données NPZ et le catalogue FITS.
    inputs : data_dir (str), morpho_path (str)
    appels : Table.read, SkyCoord, glob.glob, np.load, SkyCoord.match_to_catalog_sky
    outputs : Dictionnaire contenant les compteurs d'objets (Dict[str, int])
    '''
    if not os.path.exists(morpho_path):
        raise FileNotFoundError(f"Catalogue introuvable : {morpho_path}")

    logger.info("Chargement du catalogue morphologique...")
    cat = Table.read(morpho_path, format='fits')
    
    ra_cat = cat['RAJ2000'].data
    dec_cat = cat['DEJ2000'].data
    
    # Filtrage des NaNs dans le catalogue de référence
    valid_cat_mask = ~np.isnan(ra_cat) & ~np.isnan(dec_cat)
    morpho_coords = SkyCoord(
        ra=ra_cat[valid_cat_mask] * u.deg, 
        dec=dec_cat[valid_cat_mask] * u.deg
    )

    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    stats = {
        "total_raw": 0,
        "after_physics_filter": 0,
        "successfully_matched": 0
    }

    logger.info(f"Analyse de {len(files)} fichiers NPZ...")

    for f in files:
        try:
            with np.load(f) as raw:
                if 'info' not in raw:
                    continue
                
                info = raw['info']
                if info.dtype.names:
                    info.dtype.names = tuple([n.lower() for n in info.dtype.names])
                names = info.dtype.names

                # Identification des clés
                z_key = 'z_spec' if 'z_spec' in names else 'zspec'
                ra_key = 'ra' if 'ra' in names else 'ra_zs'
                dec_key = 'dec' if 'dec' in names else 'dec_zs'
                
                if 'i' not in names or z_key not in names or ra_key not in names or dec_key not in names:
                    continue

                stats["total_raw"] += len(info)

                # Application des filtres config.py
                mask = (info['i'] >= CONFIG.I_MIN) & \
                       (info['i'] <= CONFIG.I_MAX) & \
                       (info[z_key] > 0.001) & \
                       (info[z_key] <= CONFIG.Z_MAX)
                
                info_filtered = info[mask]
                stats["after_physics_filter"] += len(info_filtered)

                if len(info_filtered) == 0:
                    continue

                # Cross-matching
                batch_coords = SkyCoord(
                    ra=info_filtered[ra_key] * u.deg, 
                    dec=info_filtered[dec_key] * u.deg
                )
                
                _, d2d, _ = batch_coords.match_to_catalog_sky(morpho_coords)
                
                # Tolérance de 1 arcseconde
                matches = np.sum(d2d < (1.0 * u.arcsec))
                stats["successfully_matched"] += int(matches)

        except Exception as e:
            logger.warning(f"Erreur sur le fichier {f} : {e}")

    return stats

if __name__ == "__main__":
    try:
        results = check_matching_statistics(CONFIG.DATA_PATH, CONFIG.MORPHO_PATH)
        
        logger.info("=== BILAN DU MATCHING ===")
        logger.info(f"Objets totaux dans les NPZ        : {results['total_raw']}")
        logger.info(f"Objets après filtres physiques    : {results['after_physics_filter']}")
        logger.info(f"Objets matchés avec morphologie   : {results['successfully_matched']}")
        
        if results['after_physics_filter'] > 0:
            efficiency = (results['successfully_matched'] / results['after_physics_filter']) * 100
            logger.info(f"Efficacité du cross-match         : {efficiency:.2f}%")
        else:
            logger.warning("Aucun objet n'a passé les filtres physiques initiaux.")
            
    except Exception as e:
        logger.error(f"Échec de l'inspection : {e}")
