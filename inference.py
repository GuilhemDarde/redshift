import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config import CONFIG
from model import ConditionalFlowMatching

def visualize_generated_samples(checkpoint_path, target_z, target_mag):
    '''
    Role : Générer des images via CFM pour vérification visuelle.
    Inputs : checkpoint_path (str), target_z (float), target_mag (float)
    '''
    device = torch.device(CONFIG.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Chargement du modèle depuis {checkpoint_path}...")
    
    # 1. Initialiser le modèle CFM
    # Note: num_timesteps ici définit la discrétisation pour l'entraînement, 
    # mais pour l'inférence on choisira le nombre de pas d'Euler (ex: 50).
    model = ConditionalFlowMatching(num_timesteps=100).to(device)
    
    # Charger les poids
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except RuntimeError as e:
        print(f"Erreur de chargement des clés : {e}")
        print("Assurez-vous que le modèle sauvegardé correspond bien à la classe ConditionalFlowMatching.")
        return

    model.eval()
    
    # 2. Préparation de la condition
    # On crée un batch de 4 exemples identiques pour voir la variabilité du bruit
    n_samples = 4
    z_tensor = torch.tensor([[target_z]] * n_samples, device=device).float()
    
    # Normalisation mag identique au training (Centrée Réduite)
    # Rappel: Dans train.py on fait (m - MEAN) / STD
    m_norm_val = (target_mag - CONFIG.MAG_MEAN) / CONFIG.MAG_STD
    m_tensor = torch.tensor([[m_norm_val]] * n_samples, device=device).float()
    
    print(f"Génération pour z={target_z}, mag_i={target_mag} (norm={m_norm_val:.2f})...")
    
    # 3. Génération (Inférence ODE)
    with torch.no_grad():
        # generate renvoie [B, 6, 64, 64]
        # On utilise 50 pas d'Euler pour une bonne qualité
        generated = model.generate(z_tensor, m_tensor, num_steps=50)
        
    # 4. Visualisation
    # Les images sont normalisées avec asinh lors du chargement.
    # Pour visualiser, on peut faire l'inverse : sinh(x)
    if CONFIG.ASINH_NORM:
        generated = torch.sinh(generated)
        
    generated = generated.cpu().numpy() # [4, 6, 64, 64]
    
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))
    
    for i in range(n_samples):
        # On somme les canaux g,r,i (indices 1,2,3) pour faire une "pseudo-image" visible
        # Ou simplement la somme de tout si on veut juste du signal
        img = generated[i, 1:4] # g, r, i
        img_composite = np.sum(img, axis=0)
        
        axes[i].imshow(img_composite, cmap='magma', origin='lower')
        axes[i].set_title(f"Sample {i+1}\nz={target_z}")
        axes[i].axis('off')
    
    plt.tight_layout()
    output_file = f"sample_z{target_z}_m{target_mag}.png"
    plt.savefig(output_file)
    print(f"Image sauvegardée : {output_file}")
    # plt.show() # Décommenter si vous avez un affichage graphique

if __name__ == "__main__":
    ckpt = os.path.join(CONFIG.EXP_FOLDER, "cfm_model_epoch_100.pt")
    
    if os.path.exists(ckpt):
        visualize_generated_samples(ckpt, target_z=1.5, target_mag=22.0)
    else:
        print(f"En attente du fichier {ckpt}...")
        print("Laissez l'entraînement tourner encore quelques minutes !")