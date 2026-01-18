import torch
import numpy as np
import os
from config import CONFIG
from data_loader import get_dataloader
from model import ConditionalFlowMatching 
import utils

def train():
    os.makedirs(CONFIG.EXP_FOLDER, exist_ok=True)
    device = torch.device(CONFIG.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    dataloader = get_dataloader()
    
    # Batch de vérification
    first_batch = next(iter(dataloader))
    utils.visualize_batch(first_batch[0], first_batch[1], first_batch[2], title="Training Batch Preview")

    # Initialisation du modèle CFM
    print("Initialisation du modèle Conditional Flow Matching...")
    model = ConditionalFlowMatching(num_timesteps=100).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG.N_EPOCHS * len(dataloader)
    )

    print("Début de l'entraînement...")
    
    for epoch in range(CONFIG.N_EPOCHS):
        model.train()
        epoch_loss = []
        
        for batch_idx, (images, z, mags) in enumerate(dataloader):
            images = images.to(device)
            z = z.to(device).unsqueeze(1)
            mags = mags.to(device).unsqueeze(1)
            
            # Normalisation magnitudes (globale)
            mags_norm = (mags - CONFIG.MAG_MEAN) / CONFIG.MAG_STD

            optimizer.zero_grad()
            
            # Forward pass du CFM (calcule la Loss sur le champ de vecteurs)
            loss = model(images, z, mags_norm)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss.append(loss.item())
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Step {batch_idx} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = np.mean(epoch_loss)
        print(f"=== Epoch {epoch+1}/{CONFIG.N_EPOCHS} Finished | Avg Loss: {avg_loss:.5f} ===")
        
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CONFIG.EXP_FOLDER, f"cfm_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Modèle sauvegardé : {save_path}")

if __name__ == "__main__":
    train()