import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import logging
from config import CONFIG
from model import ConditionalFlowMatching, get_timestep_embedding
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def build_condition_vector(
    z_value: float,
    mag_value: float,
    device: torch.device,
    g_r: float = 0.0,
    r_i: float = 0.0,
    i_z: float = 0.0,
    re_norm: float = 0.0,
    n_norm: float = 0.0,
) -> torch.Tensor:
    """Construit le vecteur de conditionnement 7D attendu par le CFM actuel."""
    mag_norm = (mag_value - 22.0) / 2.0
    return torch.tensor(
        [[z_value, mag_norm, g_r, r_i, i_z, re_norm, n_norm]],
        device=device,
        dtype=torch.float32,
    )

@torch.no_grad()
def generate_with_fixed_noise(model: ConditionalFlowMatching, x_0: torch.Tensor, cond_vector: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
    '''
    actions : Génère une image depuis un bruit initial x_0 FIXE pour comparer l'effet du conditionnement.
    inputs : x_0 [B, 6, H, W]
    outputs : x_1 [B, 6, H, W]
    '''
    B = cond_vector.shape[0]
    device = cond_vector.device
    cond_emb = model.condition_encoder(cond_vector)
    
    x = x_0.clone().to(device)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_val = i / num_steps
        t_batch = torch.full((B,), t_val * 1000, device=device)
        t_emb = get_timestep_embedding(t_batch, 128)
        
        v_pred = model.denoiser(x, t_emb, cond_emb)
        x = x + v_pred * dt
        
    return x

def plot_evolution_grid(checkpoint_path: str, fixed_mag: float = 22.0, num_seeds: int = 4):
    '''
    actions : Crée une matrice (Seeds x Redshifts) pour visualiser l'impact de Z sur la morphologie.
    '''
    device = torch.device(CONFIG.DEVICE)
    set_global_seed(CONFIG.SEED)
    logger.info(f"Chargement modèle : {checkpoint_path}")
    
    model = ConditionalFlowMatching(num_timesteps=CONFIG.TIMESTEPS).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    z_values = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    n_cols = len(z_values)
    n_rows = num_seeds
    
    logger.info(f"Génération grille {n_rows}x{n_cols}...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    
    for row in range(n_rows):
        # Bruit unique pour cette ligne (Identité intrinsèque)
        seed_noise = torch.randn(1, 6, 64, 64, device=device)
        
        for col, z_val in enumerate(z_values):
            cond_vector = build_condition_vector(z_val, fixed_mag, device)
            
            img_tensor = generate_with_fixed_noise(model, seed_noise, cond_vector)
            
            if CONFIG.ASINH_NORM:
                img_tensor = torch.sinh(img_tensor)
            
            img_np = img_tensor.cpu().numpy()[0]
            img_composite = np.sum(img_np[1:4], axis=0) # Somme g,r,i
            
            # Gestion axes (cas n_rows=1 ou n_cols=1)
            if n_rows > 1: ax = axes[row, col]
            else: ax = axes[col]
            
            ax.imshow(img_composite, cmap='magma', origin='lower')
            
            if row == 0: ax.set_title(f"z = {z_val}", fontsize=12, fontweight='bold')
            if col == 0: ax.set_ylabel(f"Seed {row+1}", fontsize=12)
            ax.axis('off')
            
    plt.suptitle(f"Morphologie vs Redshift (Mag={fixed_mag})", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "evolution_grid.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Grille sauvegardée : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mag", type=float, default=22.5, help="Fixed Magnitude")
    parser.add_argument("--seeds", type=int, default=5, help="Number of galaxy seeds")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a CFM checkpoint")
    args = parser.parse_args()
    
    ckpt = args.checkpoint or CONFIG.exp_path(CONFIG.CFM_CHECKPOINT)
    plot_evolution_grid(ckpt, args.mag, args.seeds)
