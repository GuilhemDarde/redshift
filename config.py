import os

class Config:
    """
    Configuration globale pour le projet COSMOS Flow Matching.
    """

    DATA_PATH = "/home/barrage/SPECT_COSMOS_US/us/" 
    EXP_FOLDER = "/home/hugo/experiments/cfm_v1/"    
    # Paramètres données
    I_MIN = 18.0
    I_MAX = 25.0
    Z_MAX = 6.0
    
    # 0 : u, 1 : g, 2 : r, 3 : i, 4 : z, 5 : y
    CHANNELS = [0, 1, 2, 3, 4, 5] 
    
    IMG_SIZE = 64
    ASINH_NORM = True
    
    # Statistiques
    MAG_MEAN = 23.563542
    MAG_STD  = 1.276101

    # Paramètres CFM
    TIMESTEPS = 100 
    COND_EMB_DIM = 256
    
    # Entraînement
    BATCH_SIZE = 64 
    LR = 1e-4
    N_EPOCHS = 100
    DEVICE = "cuda"

    def __init__(self):
        os.makedirs(self.EXP_FOLDER, exist_ok=True)

CONFIG = Config()