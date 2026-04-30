import os
from dataclasses import dataclass, field
from typing import List, Tuple

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)

@dataclass
class Config:
    '''
    actions : Configuration globale autonome (Standard Library Only).
    '''
    # Chemins
    DATA_PATH: str = os.getenv("COSMOS_DATA_PATH", os.path.join(PROJECT_ROOT, "data", "cosmos_npz"))
    EXP_FOLDER: str = os.getenv("COSMOS_EXP_FOLDER", os.path.join(PROJECT_ROOT, "experiments"))
    MORPHO_PATH: str = os.getenv("COSMOS_MORPHO_PATH", os.path.join(PROJECT_ROOT, "catalogs", "catalogue_morpho_cosmos.fits"))
    METADATA_PATH: str = os.getenv("COSMOS_METADATA_PATH", "")

    # Paramètres Données
    I_MIN: float = 18.0
    I_MAX: float = 25.0
    Z_MAX: float = 6.0
    BAND_NAMES: List[str] = field(default_factory=lambda: ["u", "g", "r", "i", "z", "y"])
    
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
    N_FOLDS: int = int(os.getenv("COSMOS_N_FOLDS", "5"))

    # Analyses
    I_ANALYSIS_RANGE: Tuple[float, float] = (18.0, 20.0)
    I_ANALYSIS_BINS: int = 20
    Z_BIN_EDGES: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.8, 4.0, 6.0])
    STRIPE82_RA_MIN: float = float(os.getenv("COSMOS_STRIPE82_RA_MIN", "300.0"))
    STRIPE82_RA_MAX: float = float(os.getenv("COSMOS_STRIPE82_RA_MAX", "60.0"))
    STRIPE82_DEC_MIN: float = float(os.getenv("COSMOS_STRIPE82_DEC_MIN", "-1.25"))
    STRIPE82_DEC_MAX: float = float(os.getenv("COSMOS_STRIPE82_DEC_MAX", "1.25"))

    # Artefacts canoniques
    CFM_CHECKPOINT: str = "cfm_model_physics.pt"
    SYNTHETIC_50K: str = "synthetic_cosmos_50k_v3.npz"
    SYNTHETIC_100K: str = "synthetic_cosmos_100k_v3.npz"
    SOTA_RESULTS: str = "results_sota_ensemble.npz"
    DATASET_METADATA: str = "dataset_metadata.npz"

    DEVICE: str = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"

    def exp_path(self, filename: str) -> str:
        return os.path.join(self.EXP_FOLDER, filename)

    def __post_init__(self):
        '''Vérifications au lancement'''
        # Création du dossier d'expériences
        os.makedirs(self.EXP_FOLDER, exist_ok=True)
        
        # Verification des chemins explicites sans imposer de donnees locales aux tests.
        if "COSMOS_DATA_PATH" in os.environ and not os.path.exists(self.DATA_PATH):
            print(f"Dossier introuvable : {self.DATA_PATH}")
        if "COSMOS_MORPHO_PATH" in os.environ and not os.path.exists(self.MORPHO_PATH):
            print(f"Catalogue morphologique introuvable : {self.MORPHO_PATH}")

# Instance globale
CONFIG = Config()
