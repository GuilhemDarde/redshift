import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    '''
    actions : Configuration globale autonome (Standard Library Only).
    '''
    # Chemins
    DATA_PATH: str = os.getenv("COSMOS_DATA_PATH", "/home/barrage/SPECT_COSMOS_US/us/")
    EXP_FOLDER: str = os.getenv("COSMOS_EXP_FOLDER", "/home/hugo/experiments/cfm_v1/")
    MORPHO_PATH: str = os.getenv("COSMOS_MORPHO_PATH", "/home/hugo/catalogue_morpho_cosmos.fits")

    # Paramètres Données
    I_MIN: float = 18.0
    I_MAX: float = 25.0
    Z_MAX: float = 6.0
    
    # On utilise field() pour initialiser une liste vide
    CHANNELS: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    
    IMG_SIZE: int = 64
    ASINH_NORM: bool = True

    # Statistiques (Constantes physiques)
    MAG_MEAN: float = 23.563542
    MAG_STD: float = 1.276101

    # Paramètres CFM
    TIMESTEPS: int = 100
    COND_EMB_DIM: int = 256

    # Entraînement
    BATCH_SIZE: int = 64
    SYNTH_BATCH_SIZE: int = 128
    GENERATION_BATCH_SIZE: int = 256
    NUM_WORKERS: int = int(os.getenv("COSMOS_NUM_WORKERS", "2"))
    SYNTH_NUM_WORKERS: int = int(os.getenv("COSMOS_SYNTH_NUM_WORKERS", "4"))
    SEED: int = int(os.getenv("COSMOS_SEED", "42"))
    LR: float = 1e-4
    N_EPOCHS: int = 100

    # Artefacts canoniques
    CFM_CHECKPOINT: str = "cfm_model_physics.pt"
    SYNTHETIC_50K: str = "synthetic_cosmos_50k_v3.npz"
    SYNTHETIC_100K: str = "synthetic_cosmos_100k_v3.npz"
    SOTA_RESULTS: str = "results_sota_ensemble.npz"

    DEVICE: str = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"

    def exp_path(self, filename: str) -> str:
        return os.path.join(self.EXP_FOLDER, filename)

    def __post_init__(self):
        '''Vérifications au lancement'''
        # Création du dossier d'expériences
        os.makedirs(self.EXP_FOLDER, exist_ok=True)
        
        # Vérification si les données sont accessibles
        if not os.path.exists(self.DATA_PATH):
            print(f"Dossier introuvable : {self.DATA_PATH}")
        if not os.path.exists(self.MORPHO_PATH):
            print(f"Catalogue morphologique introuvable : {self.MORPHO_PATH}")

# Instance globale
CONFIG = Config()
