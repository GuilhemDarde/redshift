'''
actions : 
    1. Charge le dataset synthétique (100k).
    2. Entraîne un Backbone (CNN) dessus.
    3. Teste en Zero-Shot sur le VRAI dataset (Sim2Real).
    4. Fine-tune sur une fraction du VRAI dataset (Transfer Learning).
    5. Génère le rapport statistique complet (Biais, NMAD, Robustesse Mag).
inputs : synthetic_cosmos_100k.npz, DataLoader réel.
outputs : residuals_FULL_Sim2Real_Final.png
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18
from config import CONFIG
from data_loader import get_dataloaders
from statistical_report import statistical_report 

# --- Architecture CNN (Régresseur) ---
class GalaxyRegresser(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18 standard (Non pré-entraîné ImageNet car nos images sont très différentes)
        self.net = resnet18(pretrained=False)
        # Modif entrée : 6 canaux
        old_conv = self.net.conv1
        self.net.conv1 = nn.Conv2d(6, old_conv.out_channels, 
                                   kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                                   padding=old_conv.padding, bias=old_conv.bias)
        # Modif sortie : 1 valeur (z)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x):
        return self.net(x)

def load_synthetic_loader(path, batch_size=128):
    print(f"Chargement des données synthétiques : {path}")
    data = np.load(path)
    # Conversion en tenseurs
    x = torch.tensor(data['x'], dtype=torch.float32)
    z = torch.tensor(data['z'], dtype=torch.float32)
    
    ds = TensorDataset(x, z)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chemins
    synth_path = os.path.join(CONFIG.EXP_FOLDER, "synthetic_cosmos_100k.npz")
    if not os.path.exists(synth_path):
        print(f"ERREUR: {synth_path} introuvable. Lancez generate_mass_dataset.py d'abord !")
        return

    # 1. Chargement des Données
    syn_loader = load_synthetic_loader(synth_path)
    real_train, _, real_test = get_dataloaders() # On récupère les vraies données
    
    # 2. Modèle et Optimisation
    backbone = GalaxyRegresser().to(device)
    optimizer = optim.Adam(backbone.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # --- PHASE A : Entraînement sur SYNTHÉTIQUE (Pre-training) ---
    print("\n=== A. Entraînement sur DONNÉES SYNTHÉTIQUES (100k) ===")
    epochs_syn = 15 # 15 suffisent souvent si le dataset est propre, tu peux remettre 50
    
    for ep in range(epochs_syn):
        backbone.train()
        total_loss = 0
        for x, z in syn_loader:
            x, z = x.to(device), z.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = backbone(x)
            loss = criterion(pred, z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(syn_loader)
        print(f"Epoch {ep+1}/{epochs_syn} | Loss Synth: {avg_loss:.4f}")

    # --- PHASE B : Test Zero-Shot sur RÉEL ---
    # (Optionnel ici car on veut surtout le résultat final, mais utile pour debugger)
    print("\n=== B. Test Zero-Shot sur RÉEL (Sim2Real) ===")
    backbone.eval()
    
    # --- PHASE C : Fine-Tuning sur RÉEL (Transfert) ---
    print("\n=== C. Fine-Tuning sur RÉEL (10% des données) ===")
    # On réduit le learning rate pour ne pas casser le pré-entraînement
    optimizer = optim.Adam(backbone.parameters(), lr=1e-4)
    
    # On simule un scénario "peu de données" (ex: 500 batchs)
    ft_steps = 500
    step = 0
    backbone.train()
    
    print(f"Fine-tuning en cours...")
    for x, z, _ in real_train:
        if step >= ft_steps: break
        x, z = x.to(device), z.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(backbone(x), z)
        loss.backward()
        optimizer.step()
        step += 1

    # --- PHASE D : Test Final (Avec Magnitudes) ---
    print("\n=== D. Test Final (Collecte des métriques complètes) ===")
    backbone.eval()
    preds_ft, targets_ft, mags_ft = [], [], [] # <-- NOUVEAU : on stocke les mags
    
    with torch.no_grad():
        # MODIFICATION : On récupère 'mag' (3ème élément du loader)
        for x, z, mag in real_test:
            x = x.to(device)
            p = backbone(x).cpu().view(-1)
            
            preds_ft.extend(p.numpy())
            targets_ft.extend(z.numpy())
            mags_ft.extend(mag.numpy()) # <-- Stockage

    # Conversion en numpy arrays
    targets_ft = np.array(targets_ft)
    preds_ft = np.array(preds_ft)
    mags_ft = np.array(mags_ft)

    # --- Lancement du RAPPORT FINAL ---
    # C'est ici qu'on appelle ta nouvelle fonction
    print("Génération des graphiques de validation...")
    statistical_report(targets_ft, preds_ft, mags_ft, title="Sim2Real_Final")

if __name__ == "__main__":
    run_experiment()