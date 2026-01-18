import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import torch
from config import CONFIG

def plot_ra_dec(data_dir=CONFIG.DATA_PATH):
    '''
    Role fonction : Visualiser la couverture spatiale (RA vs DEC) des objets sélectionnés.
    Inputs : data_dir (str)
    Actions :
        - Parcourt les fichiers .npz
        - Extrait RA/DEC des objets valides (selon les critères de config)
        - Affiche un scatter plot et le sauvegarde
    '''
    print("Génération du plot RA/DEC (lecture rapide)...")
    ra_list = []
    dec_list = []
    
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    for f in files:
        try:
            with np.load(f) as raw:
                if 'info' not in raw: continue
                info = raw['info']
                
                # Normalisation des noms de colonnes
                if info.dtype.names:
                    info.dtype.names = tuple([n.lower() for n in info.dtype.names])
                
                names = info.dtype.names
                z_key = 'z_spec' if 'z_spec' in names else 'zspec'
                
                if z_key not in names or 'i' not in names:
                    continue

                # Réplication du masque de sélection
                mask = (info['i'] >= CONFIG.I_MIN) & \
                       (info['i'] <= 25.0) & \
                       (info[z_key] > 0.001) & \
                       (info[z_key] <= CONFIG.Z_MAX)
                
                if 'flag' in raw.files:
                    flags = raw['flag'][:, CONFIG.CHANNELS]
                    mask = mask & (np.sum(flags, axis=1) == 0)

                if np.sum(mask) > 0:
                    ra_key = 'ra' if 'ra' in names else 'RA'
                    dec_key = 'dec' if 'dec' in names else 'DEC'
                    
                    ra_list.append(info[ra_key][mask])
                    dec_list.append(info[dec_key][mask])
        except Exception as e:
            continue

    if len(ra_list) == 0:
        print("Aucune donnée trouvée pour le plot.")
        return

    ra = np.concatenate(ra_list)
    dec = np.concatenate(dec_list)

    plt.figure(figsize=(8, 8))
    plt.scatter(ra, dec, s=0.5, alpha=0.5, c='blue')
    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    plt.title(f'Couverture Spatiale ({len(ra)} objets)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "coverage_ra_dec.png")
    plt.savefig(save_path)
    print(f"Plot RA/DEC sauvegardé sous '{save_path}'.")

def visualize_batch(images, z, mags, title="Batch Preview"):
    '''
    Role fonction : Afficher un batch d'images avec leurs redshifts.
    Inputs : images tensor [B, C, H, W], z [B], mags [B]
    Actions : Affiche une grille d'images (somme des canaux).
    '''
    if hasattr(images, 'cpu'): images = images.detach().cpu()
    if hasattr(z, 'cpu'): z = z.detach().cpu()
    if hasattr(mags, 'cpu'): mags = mags.detach().cpu()
    
    if hasattr(images, 'numpy'): images = images.numpy()
    if hasattr(z, 'numpy'): z = z.numpy()
    if hasattr(mags, 'numpy'): mags = mags.numpy()
    
    B = images.shape[0]
    n_show = min(B, 5)
    
    fig, axes = plt.subplots(1, n_show, figsize=(15, 3))
    if n_show == 1: axes = [axes]
    
    for i in range(n_show):
        # images[i] est [C, H, W] -> Somme sur C (axis 0)
        im_disp = np.sum(images[i], axis=0)
        
        # Affichage
        axes[i].imshow(im_disp, cmap='magma', origin='lower')
        
        # Gestion robuste des scalaires vs tableaux 1D
        z_val = z[i].item() if z[i].ndim == 0 else z[i][0]
        mag_val = mags[i].item() if mags[i].ndim == 0 else mags[i][0]
        
        axes[i].set_title(f"z={z_val:.2f} | mag={mag_val:.1f}")
        axes[i].axis('off')
        
    plt.suptitle(title)
    plt.tight_layout()
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "batch_preview.png")
    plt.savefig(save_path)
    print(f"Batch preview sauvegardé sous '{save_path}'.")