import argparse
import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from data_loader import get_dataloaders
from backbone import GalaxyEquivariantMDN, MDNLoss
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_synthetic_loader(path: str, batch_size: int = CONFIG.SYNTH_BATCH_SIZE) -> DataLoader:
    '''
    actions : Charge le jeu de données synthétiques avec prise en charge de l'encodage V2.
    inputs : path (str), batch_size (int)
    appels : os.path.exists, np.load, torch.tensor, TensorDataset, DataLoader
    outputs : DataLoader PyTorch
    '''
    if not os.path.exists(path):
        candidates = sorted([f for f in os.listdir(CONFIG.EXP_FOLDER) if f.startswith("synthetic") and f.endswith(".npz")])
        if candidates:
            path = os.path.join(CONFIG.EXP_FOLDER, candidates[-1])
            logger.info(f"Fichier non spécifié, utilisation automatique de : {path}")
        else:
            raise FileNotFoundError(f"Aucun dataset synthétique trouvé dans {CONFIG.EXP_FOLDER}")
    
    logger.info(f"Chargement données synthétiques : {path}")
    data = np.load(path)
    
    x = torch.tensor(data['x'], dtype=torch.float32)
    
    if 'cond' in data:
        z = torch.tensor(data['cond'][:, 0], dtype=torch.float32)
    else:
        z = torch.tensor(data['z'], dtype=torch.float32)
        
    ds = TensorDataset(x, z)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=CONFIG.NUM_WORKERS)

def compute_pit_mixture(z_true: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    '''
    actions : Calcule la transformée intégrale de probabilité (PIT) pour une distribution Mixture Density Network (somme pondérée des CDF).
    inputs : z_true (np.ndarray), pi (np.ndarray), mu (np.ndarray), sigma (np.ndarray)
    appels : np.zeros_like, stats.norm.cdf
    outputs : Vecteur des valeurs PIT (np.ndarray)
    '''
    pit_values = np.zeros_like(z_true, dtype=np.float64)
    num_gaussians = pi.shape[1]
    
    for k in range(num_gaussians):
        pit_values += pi[:, k] * stats.norm.cdf(z_true, loc=mu[:, k], scale=sigma[:, k])
        
    return pit_values

def get_predictions(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    actions : Extrait les prédictions paramétriques complètes du G-CNN MDN sur un jeu de données complet.
    inputs : loader (DataLoader), model (torch.nn.Module), device (torch.device)
    appels : torch.no_grad, np.array
    outputs : Tuple contenant z_true, pi, mu, sigma (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray])
    '''
    pis, mus, sigmas, z_trues = [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, target = batch
                if target.dim() > 1 and target.shape[1] >= 5:
                    z = target[:, 0]
                else:
                    z = target
            else: 
                x, z, _ = batch

            x = x.to(device)
            pi, mu, sigma = model(x)
            
            pis.extend(pi.cpu().numpy())
            mus.extend(mu.cpu().numpy())
            sigmas.extend(sigma.cpu().numpy())
            z_trues.extend(z.numpy())
            
    return np.array(z_trues), np.array(pis), np.array(mus), np.array(sigmas)

def run_uncertainty_analysis(
    epochs: int = 15,
    num_gaussians: int = 3,
    batch_size: int = CONFIG.SYNTH_BATCH_SIZE,
    seed: int = CONFIG.SEED,
) -> None:
    '''
    actions : Exécute le pipeline d'entraînement et génère les diagrammes PIT Sim2Sim et Sim2Real pour valider la calibration des incertitudes du modèle équivariant.
    inputs : epochs (int), num_gaussians (int)
    appels : load_synthetic_loader, get_dataloaders, GalaxyEquivariantMDN, optim.Adam, MDNLoss, get_predictions, compute_pit_mixture, plt.subplots, plt.savefig
    outputs : None
    '''
    set_global_seed(seed)
    device = torch.device(CONFIG.DEVICE)
    
    syn_loader = load_synthetic_loader(CONFIG.exp_path(CONFIG.SYNTHETIC_50K), batch_size=batch_size)
    _, _, test_real = get_dataloaders(batch_size=batch_size, num_workers=CONFIG.NUM_WORKERS)
    
    model = GalaxyEquivariantMDN(num_gaussians=num_gaussians, use_meta=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = MDNLoss()
    
    logger.info(">>> Training G-CNN MDN Backbone (sur Synthétique)...")
    model.train()
    for epoch in range(epochs):
        losses = []
        for x, z in syn_loader:
            x, z = x.to(device), z.to(device)
            optimizer.zero_grad()
            
            pi, mu, sigma = model(x)
            loss = criterion(pi, mu, sigma, z)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logger.info(f"Epoch {epoch+1}/{epochs} | NLL Loss: {np.mean(losses):.4f}")

    model.eval()
    
    logger.info(">>> Calcul PIT sur Données Réelles (Sim2Real)...")
    z_r, pi_r, mu_r, sigma_r = get_predictions(test_real, model, device)
    pit_real = compute_pit_mixture(z_r, pi_r, mu_r, sigma_r)
    
    logger.info(">>> Calcul PIT sur Données Synthétiques (Sim2Sim)...")
    z_s, pi_s, mu_s, sigma_s = get_predictions(syn_loader, model, device)
    pit_sim = compute_pit_mixture(z_s, pi_s, mu_s, sigma_s)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(pit_sim, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    axes[0].axhline(1.0, color='r', linestyle='--', linewidth=2, label="Idéal (Uniforme)")
    axes[0].set_title("PIT Sim2Sim (Intrinsèque)\nDevrait être plat")
    axes[0].legend()

    axes[1].hist(pit_real, bins=30, density=True, alpha=0.6, color='salmon', edgecolor='black')
    axes[1].axhline(1.0, color='r', linestyle='--', linewidth=2, label="Idéal")
    axes[1].set_title("PIT Sim2Real (Transfert)\nDiagnostic du Gap")
    axes[1].legend()
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "pit_comparison_gcnn_mdn.png")
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Graphique sauvegardé : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num_gaussians", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=CONFIG.SYNTH_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    args = parser.parse_args()
    
    run_uncertainty_analysis(
        epochs=args.epochs,
        num_gaussians=args.num_gaussians,
        batch_size=args.batch_size,
        seed=args.seed,
    )
