import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config import CONFIG
from model import ConditionalFlowMatching, get_timestep_embedding

# Fonction de génération "custom" pour contrôler le bruit initial x_0
# (On ne peut pas utiliser model.generate() directement car elle recrée du bruit aléatoire)
@torch.no_grad()
def generate_with_fixed_noise(model, x_0, z_tensor, m_tensor, num_steps=50):
    """
    Génère une image en partant d'un bruit x_0 FIXE.
    Cela permet de voir comment CE bruit spécifique évolue selon z.
    """
    B = z_tensor.shape[0]
    device = z_tensor.device
    
    # Condition
    cond_emb = model.condition_encoder(z_tensor, m_tensor)
    
    # On part du bruit fourni
    x = x_0.clone().to(device)
    dt = 1.0 / num_steps
    
    # Résolution ODE
    for i in range(num_steps):
        t_val = i / num_steps
        t_batch = torch.full((B,), t_val * 1000, device=device)
        t_emb = get_timestep_embedding(t_batch, 128)
        
        v_pred = model.denoiser(x, t_emb, cond_emb)
        x = x + v_pred * dt
        
    return x

def plot_evolution_grid(checkpoint_path, fixed_mag=22.0, num_seeds=4):
    device = torch.device(CONFIG.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Chargement du modèle : {checkpoint_path}")
    
    # Chargement Modèle
    model = ConditionalFlowMatching(num_timesteps=100).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Paramètres de la grille
    z_values = [0.5, 1.0, 2.0, 3.0, 4.0]
    n_cols = len(z_values)
    n_rows = num_seeds
    
    # Normalisation de la magnitude (fixe pour toute la grille pour isoler l'effet de z)
    m_norm_val = (fixed_mag - CONFIG.MAG_MEAN) / CONFIG.MAG_STD
    
    print(f"Génération de la grille : {n_rows} galaxies x {n_cols} redshifts...")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    
    # Pour chaque ligne (chaque seed de galaxie)
    for row in range(n_rows):
        # On crée un bruit unique pour cette ligne (identité latente de la galaxie)
        # Shape [1, 6, 64, 64]
        seed_noise = torch.randn(1, 6, 64, 64, device=device)
        
        for col, z_val in enumerate(z_values):
            # Préparation des tenseurs pour ce z spécifique
            z_tensor = torch.tensor([[z_val]], device=device).float()
            m_tensor = torch.tensor([[m_norm_val]], device=device).float()
            
            # Génération avec le bruit forcé
            img_tensor = generate_with_fixed_noise(model, seed_noise, z_tensor, m_tensor)
            
            # Post-traitement visuel (sinh inverse + somme canaux g,r,i)
            if CONFIG.ASINH_NORM:
                img_tensor = torch.sinh(img_tensor)
            
            img_np = img_tensor.cpu().numpy()[0] # [6, 64, 64]
            img_composite = np.sum(img_np[1:4], axis=0) # Somme g,r,i
            
            # Affichage
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.imshow(img_composite, cmap='magma', origin='lower')
            
            if row == 0:
                ax.set_title(f"z = {z_val}", fontsize=14, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"Seed {row+1}", fontsize=12, fontweight='bold')
                
            ax.axis('off')
            
    plt.suptitle(f"Évolution Morphologique (Mag i={fixed_mag})", fontsize=16)
    plt.tight_layout()
    
    filename = "evolution_grid.png"
    plt.savefig(filename)
    print(f"Grille sauvegardée sous : {filename}")

if __name__ == "__main__":
    ckpt = os.path.join(CONFIG.EXP_FOLDER, "cfm_model_epoch_100.pt")
    
    if os.path.exists(ckpt):
        plot_evolution_grid(ckpt, fixed_mag=22.5, num_seeds=5)
    else:
        print(f"Checkpoint introuvable : {ckpt}")