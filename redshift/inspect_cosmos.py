import os
import glob
import logging
import numpy as np
from typing import Optional
from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_cosmos_npz(data_dir: str = CONFIG.DATA_PATH) -> Optional[str]:
    '''
    actions : Inspecte et journalise le contenu du premier fichier .npz COSMOS trouvé.
    inputs :
        - data_dir (str) : Chemin du répertoire contenant les fichiers .npz.
    appels : np.load, glob.glob
    outputs :
        - str : Chemin du fichier inspecté (ou None si aucun fichier trouvé).
    '''
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not files:
        logger.error("Aucun fichier .npz trouvé dans le répertoire spécifié.")
        return None
        
    target_file = files[0]
    logger.info(f"Inspection du fichier : {os.path.basename(target_file)}")
    
    try:
        with np.load(target_file) as data:
            logger.info(f"Clés principales (tableaux dans l'archive) : {data.files}")
            
            if 'info' in data:
                info_array = data['info']
                logger.info(f"Champs disponibles dans 'info' (métadonnées) : {info_array.dtype.names}")
                
            if 'cube' in data:
                cube_array = data['cube']
                logger.info(f"Forme du tenseur 'cube' (images) : {cube_array.shape}")
                
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier : {e}")
            
    return target_file

if __name__ == "__main__":
    inspect_cosmos_npz()