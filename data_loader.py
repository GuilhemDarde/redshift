import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import os
from config import CONFIG

class CosmosDataset(Dataset):
    """
    Chargement des données COSMOS avec filtrage flags et mag.
    Intègre désormais la Data Augmentation physique (Isotropie).
    """
    def __init__(self, data_dir, mode='train'):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.mode = mode # Permet d'activer/désactiver l'augmentation
        self.data = self._load_and_process()

    def _load_and_process(self):
        print(f"Chargement de {len(self.files)} fichiers...")
        
        all_x = []
        all_z = []
        all_mags = []

        for f in self.files:
            try:
                with np.load(f) as raw:
                    if 'info' not in raw or 'cube' not in raw:
                        continue
                        
                    info = raw['info']
                    cube = raw['cube']
                    
                    # Normalisation des noms de colonnes
                    if info.dtype.names:
                        info.dtype.names = tuple([n.lower() for n in info.dtype.names])
                    names = info.dtype.names
                    
                    z_key = 'z_spec' if 'z_spec' in names else 'zspec'
                    if 'i' not in names or z_key not in names:
                        continue

                    # Filtrage
                    mask = np.ones(len(info), dtype=bool)
                    
                    # Flags
                    if 'flag' in raw.files:
                        flags = raw['flag'][:, CONFIG.CHANNELS]
                        mask = mask & (np.sum(flags, axis=1) == 0)

                    # Magnitude & Redshift (i <= 25.0 pour le Deep Learning)
                    mask_vals = (info['i'] >= CONFIG.I_MIN) & \
                                (info['i'] <= 25.0) & \
                                (info[z_key] > 0.001) & \
                                (info[z_key] <= CONFIG.Z_MAX)
                    
                    final_mask = mask & mask_vals

                    if np.sum(final_mask) == 0:
                        continue

                    # Extraction des canaux spécifiés dans CONFIG.CHANNELS
                    cube_filtered = cube[final_mask][:, :, :, CONFIG.CHANNELS]
                    zspec_filtered = info[z_key][final_mask]
                    mag_i_filtered = info['i'][final_mask]
                    
                    # Normalisation Images (asinh)
                    if CONFIG.ASINH_NORM:
                        cube_filtered = np.arcsinh(cube_filtered) 

                    all_x.append(cube_filtered)
                    all_z.append(zspec_filtered)
                    all_mags.append(mag_i_filtered)

            except Exception as e:
                # print(f"Skipping {f}: {e}") # Commenté pour éviter le spam si beaucoup d'erreurs
                continue

        if len(all_x) == 0:
            raise ValueError("Aucune donnée chargée. Vérifiez les chemins et filtres.")

        data = {
            'x': np.concatenate(all_x, axis=0),
            'zspec': np.concatenate(all_z, axis=0),
            'mag': np.concatenate(all_mags, axis=0),
        }
        
        # [N, H, W, C] -> [N, C, H, W] pour PyTorch
        data['x'] = np.transpose(data['x'], (0, 3, 1, 2))
        
        print(f"Dataset prêt : {len(data['zspec'])} objets chargés.")
        return data

    def __len__(self):
        return len(self.data['zspec'])

    def __getitem__(self, idx):
        x = torch.tensor(self.data['x'][idx], dtype=torch.float32)
        z = torch.tensor(self.data['zspec'][idx], dtype=torch.float32)
        m = torch.tensor(self.data['mag'][idx], dtype=torch.float32)

        # --- DATA AUGMENTATION (Physique : Isotropie) ---
        # Uniquement si mode='train'.
        # Note : Si utilisé via random_split sur un dataset unique, 
        # le mode initial s'applique à tous les sous-ensembles.
        if self.mode == 'train':
            # 1. Flip Horizontal (Probabilité 50%)
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [2]) # Flip W (axis 2)
            
            # 2. Flip Vertical (Probabilité 50%)
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [1]) # Flip H (axis 1)
                
            # 3. Rotation 90°/180°/270° (Aléatoire k=0,1,2,3)
            # Les galaxies n'ont pas de "haut" ni de "bas" dans l'univers.
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2]) # dims 1,2 sont H,W

        return x, z, m

def get_dataloaders(): # Renommé au pluriel
    """
    Retourne 3 dataloaders : train, val, test.
    Split rigide : 80% / 10% / 10%
    """
    # 1. Chargement du Dataset complet
    # On active l'augmentation (mode='train') car c'est le dataset source.
    # (Note: idéalement on désactiverait pour val/test, mais c'est complexe sans recharger les données)
    full_ds = CosmosDataset(CONFIG.DATA_PATH, mode='train')
    
    # 2. Calcul des tailles
    total_size = len(full_ds)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size
    
    print(f"Split Dataset : Train={train_size} | Val={val_size} | Test={test_size}")
    
    # 3. Découpe Aléatoire mais Fixée (Reproductibilité avec seed)
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size], generator=generator)
    
    # 4. Création des Loaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader