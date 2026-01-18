import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from config import CONFIG

class CosmosDataset(Dataset):
    """
    Chargement des données COSMOS avec filtrage flags et mag.
    """
    def __init__(self, data_dir, mode='train'):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
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

                    # filtrage
                    mask = np.ones(len(info), dtype=bool)
                    
                    # flags
                    # On rejette si la somme des flags sur les canaux utilisés > 0
                    if 'flag' in raw.files:
                        flags = raw['flag'][:, CONFIG.CHANNELS]
                        mask = mask & (np.sum(flags, axis=1) == 0)

                    # magnitude & redshift
                    # i <= 25.0 pour le ud
                    mask_vals = (info['i'] >= CONFIG.I_MIN) & \
                                (info['i'] <= 25.0) & \
                                (info[z_key] > 0.001) & \
                                (info[z_key] <= CONFIG.Z_MAX)
                    
                    final_mask = mask & mask_vals

                    if np.sum(final_mask) == 0:
                        continue

                    # Extraction
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
                print(f"Skipping {f}: {e}")
                continue

        if len(all_x) == 0:
            raise ValueError("Aucune donnée chargée. Vérifiez les chemins et filtres.")

        data = {
            'x': np.concatenate(all_x, axis=0),
            'zspec': np.concatenate(all_z, axis=0),
            'mag': np.concatenate(all_mags, axis=0),
        }
        
        # [N, H, W, C] -> [N, C, H, W]
        data['x'] = np.transpose(data['x'], (0, 3, 1, 2))
        
        print(f"Dataset prêt : {len(data['zspec'])} objets chargés.")
        return data

    def __len__(self):
        return len(self.data['zspec'])

    def __getitem__(self, idx):
        x = torch.tensor(self.data['x'][idx], dtype=torch.float32)
        z = torch.tensor(self.data['zspec'][idx], dtype=torch.float32)
        m = torch.tensor(self.data['mag'][idx], dtype=torch.float32)
        return x, z, m

def get_dataloader():
    ds = CosmosDataset(CONFIG.DATA_PATH)
    return DataLoader(ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)