'''
actions : Visualiser la trajectoire de génération (ODE Trajectory).
          Génère une image composite montrant l'évolution de t=0 (Bruit) à t=1 (Galaxie).
inputs : Checkpoint CFM.
outputs : dynamic_evolution.png
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config import CONFIG
from model import ConditionalFlowMatching, get_timestep_embedding

def visualize_trajectory(checkpoint_path, target_z=1.0, target_mag=22.0, num_steps=20):
    device = torch.device(CONFIG.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Chargement du modèle : {checkpoint_path}")
    
    # Chargement
    model = ConditionalFlowMatching(num_timesteps=100).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Paramètres
    z_tensor = torch.tensor([[target_z]], device=device).float()
    m_norm = (target_mag - CONFIG.MAG_MEAN) / CONFIG.MAG_STD
    m_tensor = torch.tensor([[m_norm]], device=device).float()
    cond_emb = model.condition_encoder(z_tensor, m_tensor)
    
    # On fixe le bruit initial pour la reproductibilité
    torch.manual_seed(42)
    x = torch.randn(1, 6, 64, 64, device=device)
    
    # On veut capturer l'état à ces moments précis du temps t
    capture_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    snapshots = []
    
    # Boucle d'intégration manuelle (Euler) pour capturer les étapes
    dt = 1.0 / num_steps
    current_t = 0.0
    
    print(f"Intégration ODE ({num_steps} steps)...")
    
    # Capture t=0
    snapshots.append(x.clone())
    
    with torch.no_grad():
        for i in range(num_steps):
            # Prochain temps
            next_t = (i + 1) / num_steps
            
            # Prédiction Vitesse
            t_batch = torch.full((1,), current_t * 1000, device=device)
            t_emb = get_timestep_embedding(t_batch, 128)
            v_pred = model.denoiser(x, t_emb, cond_emb)
            
            # Euler Step
            x = x + v_pred * dt
            current_t = next_t
            
            # Vérification si on doit capturer (si on est proche d'un point de capture)
            # On capture simplement si on vient de dépasser un seuil
            for target_t in capture_times:
                if abs(current_t - target_t) < 1e-4 and target_t > 0:
                    snapshots.append(x.clone())
    
    # Visualisation
    print(f"Génération de la planche contact ({len(snapshots)} images)...")
    fig, axes = plt.subplots(1, len(snapshots), figsize=(18, 3))
    
    for i, img_tensor in enumerate(snapshots):
        # Post-process
        if CONFIG.ASINH_NORM:
            img_tensor = torch.sinh(img_tensor)
        
        img = img_tensor.cpu().numpy()[0] # [6, 64, 64]
        # Composite g,r,i
        rgb = np.sum(img[1:4], axis=0)
        
        axes[i].imshow(rgb, cmap='magma', origin='lower')
        axes[i].set_title(f"t = {capture_times[i]:.1f}", fontsize=14)
        axes[i].axis('off')
        
        # Ajout d'une flèche entre les plots (cosmétique)
        if i < len(snapshots) - 1:
            # Positionnement approximatif au milieu
            pass 

    plt.suptitle(f"Dynamique de Génération (Flow Matching)\nz={target_z}, mag={target_mag}", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(CONFIG.EXP_FOLDER, "dynamic_evolution.png")
    plt.savefig(save_path)
    print(f"Image sauvegardée : {save_path}")

if __name__ == "__main__":
    ckpt = os.path.join(CONFIG.EXP_FOLDER, "cfm_model_epoch_100.pt")
    if os.path.exists(ckpt):
        visualize_trajectory(ckpt)
    else:
        print("Modèle introuvable.")