import torch
import numpy as np
import os
from tqdm import tqdm
from config import CONFIG
from model import ConditionalFlowMatching

def generate_mass(n_samples=100000, batch_size=200):
    """
    Génère un dataset synthétique massif et le sauvegarde sur disque.
    """
    save_path = os.path.join(CONFIG.EXP_FOLDER, f"synthetic_cosmos_{n_samples//1000}k.npz")
    
    # Vérification si déjà existant
    if os.path.exists(save_path):
        print(f"⚠️ Le fichier {save_path} existe déjà.")
        choice = input("Voulez-vous l'écraser ? (y/n) : ")
        if choice.lower() != 'y':
            return

    device = torch.device(CONFIG.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"--- Démarrage de la génération de {n_samples} galaxies (Device: {device}) ---")

    # 1. Chargement du Modèle CFM
    cfm = ConditionalFlowMatching(num_timesteps=100).to(device)
    ckpt_path = os.path.join(CONFIG.EXP_FOLDER, "cfm_model_epoch_100.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Modèle introuvable : {ckpt_path}")
        
    cfm.load_state_dict(torch.load(ckpt_path, map_location=device))
    cfm.eval()

    # 2. Boucle de génération
    all_x = []
    all_z = []
    all_mags = []
    
    n_batches = int(np.ceil(n_samples / batch_size))
    
    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Génération"):
            current_batch = min(batch_size, n_samples - len(all_z))
            if current_batch <= 0: break

            # Echantillonnage des conditions (Prieur Uniforme pour bien couvrir l'espace)
            # Z entre 0.0 et CONFIG.Z_MAX
            z_batch = torch.rand(current_batch, device=device) * CONFIG.Z_MAX
            
            # Mag entre 18 et 25
            mag_unnorm = torch.rand(current_batch, device=device) * (25 - 18) + 18
            mag_norm = (mag_unnorm - CONFIG.MAG_MEAN) / CONFIG.MAG_STD
            
            # Mise en forme
            z_input = z_batch.unsqueeze(1)
            m_input = mag_norm.unsqueeze(1)
            
            # Génération (ODE Solve)
            # steps=30 est suffisant pour apprendre un backbone
            imgs = cfm.generate(z_input, m_input, num_steps=30) 
            
            # Stockage CPU pour économiser la VRAM
            all_x.append(imgs.cpu().numpy()) # On garde en float32
            all_z.append(z_batch.cpu().numpy())
            all_mags.append(mag_unnorm.cpu().numpy())

    # 3. Assemblage et Sauvegarde
    print("Assemblage des tableaux numpy...")
    x_final = np.concatenate(all_x, axis=0)
    z_final = np.concatenate(all_z, axis=0)
    m_final = np.concatenate(all_mags, axis=0)

    print(f"Sauvegarde sous {save_path}...")
    np.savez_compressed(save_path, x=x_final, z=z_final, mag=m_final)
    
    print(f"✅ Terminé. Taille du fichier : ~{os.path.getsize(save_path)/1e9:.2f} GB")

if __name__ == "__main__":
    generate_mass(n_samples=100000, batch_size=200)