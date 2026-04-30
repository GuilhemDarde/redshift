import glob
import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_global_seed(seed: int = CONFIG.SEED) -> None:
    '''
    actions : Fixe les graines pseudo-aléatoires pour rendre les runs comparables.
    inputs : seed (int)
    appels : np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all
    outputs : None
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_ra_dec(data_dir: str = CONFIG.DATA_PATH) -> None:
    '''
    actions : Visualise et sauvegarde la couverture spatiale (RA/DEC) des objets du jeu de données.
    inputs : data_dir (str)
    appels : glob.glob, np.load, np.concatenate, plt.subplots, os.path.join, plt.savefig
    outputs : None
    '''
    logger.info("Génération du plot RA/DEC...")
    ra_list = []
    dec_list = []
    
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    for f in files:
        try:
            with np.load(f) as raw:
                if 'info' not in raw:
                    continue
                info = raw['info']
                
                if info.dtype.names:
                    info.dtype.names = tuple([n.lower() for n in info.dtype.names])
                names = info.dtype.names
                
                z_key = 'z_spec' if 'z_spec' in names else 'zspec'
                if z_key not in names or 'i' not in names:
                    continue

                mask = (info['i'] >= CONFIG.I_MIN) & \
                       (info['i'] <= CONFIG.I_MAX) & \
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
            logger.warning(f"Erreur lors de la lecture de {f} : {e}")
            continue

    if not ra_list:
        logger.warning("Aucune donnée disponible pour le plot RA/DEC.")
        return

    ra = np.concatenate(ra_list)
    dec = np.concatenate(dec_list)

    plt.figure(figsize=(8, 8))
    plt.scatter(ra, dec, s=0.5, alpha=0.5, c='blue')
    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    plt.title(f'Couverture Spatiale ({len(ra)} objets)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "coverage_ra_dec.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Plot RA/DEC sauvegardé : {save_path}")

def visualize_batch(images: torch.Tensor, z: torch.Tensor, mags: torch.Tensor, title: str = "Batch Preview") -> None:
    '''
    actions : Génère et sauvegarde une grille de visualisation pour un lot d'images avec leurs attributs physiques.
    inputs : images (torch.Tensor), z (torch.Tensor), mags (torch.Tensor), title (str)
    appels : torch.Tensor.detach, torch.Tensor.cpu, torch.Tensor.numpy, np.sum, plt.subplots, plt.savefig
    outputs : None
    '''
    images_np = images.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    mags_np = mags.detach().cpu().numpy()
    
    batch_size = images_np.shape[0]
    n_show = min(batch_size, 5)
    
    fig, axes = plt.subplots(1, n_show, figsize=(15, 3))
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        im_disp = np.sum(images_np[i], axis=0)
        
        z_val = z_np[i].item() if z_np[i].ndim == 0 else z_np[i][0]
        mag_val = mags_np[i].item() if mags_np[i].ndim == 0 else mags_np[i][0]
        
        axes[i].imshow(im_disp, cmap='magma', origin='lower')
        axes[i].set_title(f"z={z_val:.2f} | m={mag_val:.1f}")
        axes[i].axis('off')
        
    plt.suptitle(title)
    plt.tight_layout()
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "batch_preview.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Visualisation du batch sauvegardée : {save_path}")

def statistical_report(z_true: np.ndarray, z_pred: np.ndarray, mags: np.ndarray, title: str = "Analysis") -> None:
    '''
    actions : Calcule les métriques de performance globales (Biais, NMAD, Outliers) et génère les graphiques d'analyse des résidus.
    inputs : z_true (np.ndarray), z_pred (np.ndarray), mags (np.ndarray), title (str)
    appels : np.mean, np.median, np.abs, np.sum, plt.subplots, np.linspace, np.exp, plt.savefig
    outputs : None
    '''
    residuals = (z_pred - z_true) / (1.0 + z_true)
    
    bias = np.mean(residuals)
    sigma_nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    outlier_mask = np.abs(residuals) > 0.15
    eta = np.sum(outlier_mask) / len(z_true) * 100.0
    
    logger.info(f"--- RAPPORT STATISTIQUE : {title} ---")
    logger.info(f"Biais       : {bias:.5f}")
    logger.info(f"NMAD        : {sigma_nmad:.5f}")
    logger.info(f"Outliers    : {eta:.2f}%")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].scatter(z_true, residuals, s=1, alpha=0.3, c='k')
    axes[0].axhline(0, color='r', linestyle='--')
    axes[0].set_xlabel("True Redshift")
    axes[0].set_ylabel("$\\Delta z / (1+z)$")
    axes[0].set_title("Biais vs Redshift")
    axes[0].set_ylim(-0.3, 0.3)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(mags, residuals, s=1, alpha=0.3, c='blue')
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Magnitude i")
    axes[1].set_title("Robustesse vs Magnitude")
    axes[1].set_ylim(-0.3, 0.3)
    axes[1].grid(True, alpha=0.3)

    bins = np.linspace(-0.2, 0.2, 100)
    axes[2].hist(residuals, bins=bins, density=True, alpha=0.6, color='gray', label='Data')
    
    x = np.linspace(-0.2, 0.2, 200)
    pdf = (1.0 / (sigma_nmad * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - bias) / sigma_nmad)**2)
    axes[2].plot(x, pdf, 'r-', label=f'Gauss NMAD={sigma_nmad:.3f}')
    axes[2].legend()
    axes[2].set_title("Distribution des Erreurs")
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG.EXP_FOLDER, f"residuals_{title}.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Rapport sauvegardé : {save_path}")

def compute_pit(z_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    '''
    actions : Calcule la transformée intégrale de probabilité (PIT) pour vérifier la calibration de l'incertitude prédictive.
    inputs : z_true (np.ndarray), mu (np.ndarray), sigma (np.ndarray)
    appels : stats.norm.cdf
    outputs : Vecteur des valeurs PIT (np.ndarray)
    '''
    return stats.norm.cdf(z_true, loc=mu, scale=sigma)

def extract_photometry(images: torch.Tensor) -> torch.Tensor:
    '''
    actions : Extrait un proxy photométrique (flux total) via un pooling spatial global par somme sur les canaux.
    inputs : images (torch.Tensor)
    appels : torch.sum
    outputs : Tenseur des flux extraits (torch.Tensor)
    '''
    return torch.sum(images, dim=(2, 3))
