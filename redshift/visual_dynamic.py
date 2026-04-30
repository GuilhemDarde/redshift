import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import logging
from config import CONFIG
from model import ConditionalFlowMatching, get_timestep_embedding
from utils import set_global_seed

# Setup Logging
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

def visualize_trajectory(checkpoint_path: str, target_z: float, target_mag: float, num_steps: int = 20):
    '''
    actions : Génère une planche contact montrant l'évolution du bruit vers la galaxie (t=0 -> t=1).
    inputs :
        - checkpoint_path (str)
        - target_z (float)
        - target_mag (float)
    outputs : Sauvegarde 'dynamic_evolution.png'
    '''
    device = torch.device(CONFIG.DEVICE)
    set_global_seed(CONFIG.SEED)
    logger.info(f"Chargement du modèle : {checkpoint_path}")
    
    # Chargement Modèle
    model = ConditionalFlowMatching(num_timesteps=CONFIG.TIMESTEPS).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    cond_vector = build_condition_vector(target_z, target_mag, device)
    cond_emb = model.condition_encoder(cond_vector)
    
    x = torch.randn(1, 6, 64, 64, device=device)
    
    capture_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    capture_indices = {round(t * num_steps): t for t in capture_times}
    snapshots = [(0.0, x.clone())]
    
    # Intégration Manuelle (Euler) pour capture
    dt = 1.0 / num_steps
    current_t = 0.0
    
    logger.info(f"Intégration ODE ({num_steps} steps)...")
    
    with torch.no_grad():
        for i in range(num_steps):
            next_t = (i + 1) / num_steps
            
            # Prédiction Champ de Vecteurs
            t_batch = torch.full((1,), current_t * 1000, device=device)
            t_emb = get_timestep_embedding(t_batch, 128)
            v_pred = model.denoiser(x, t_emb, cond_emb)
            
            # Euler Step
            x = x + v_pred * dt
            current_t = next_t
            
            if i + 1 in capture_indices and capture_indices[i + 1] > 0:
                snapshots.append((capture_indices[i + 1], x.clone()))
                    
    # Rendu Graphique
    logger.info(f"Génération image composite ({len(snapshots)} snapshots)...")
    fig, axes = plt.subplots(1, len(snapshots), figsize=(18, 3))
    
    for i, (time_value, img_tensor) in enumerate(snapshots):
        if CONFIG.ASINH_NORM:
            img_tensor = torch.sinh(img_tensor)
        
        img = img_tensor.cpu().numpy()[0] # [6, 64, 64]
        # Composite RGB (g,r,i -> indices 1,2,3)
        rgb = np.sum(img[1:4], axis=0)
        
        axes[i].imshow(rgb, cmap='magma', origin='lower')
        axes[i].set_title(f"t = {time_value:.1f}", fontsize=14)
        axes[i].axis('off')

    plt.suptitle(f"Dynamique de Génération (Flow Matching)\nz={target_z}, mag={target_mag}", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "dynamic_evolution.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Sauvegardé : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z", type=float, default=1.0, help="Target Redshift")
    parser.add_argument("--mag", type=float, default=22.0, help="Target Magnitude")
    parser.add_argument("--steps", type=int, default=20, help="Integration steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a CFM checkpoint")
    args = parser.parse_args()

    ckpt = args.checkpoint or CONFIG.exp_path(CONFIG.CFM_CHECKPOINT)
    visualize_trajectory(ckpt, args.z, args.mag, args.steps)
