import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import CONFIG
from data_loader import get_dataloaders # Import de la nouvelle fonction
from model import ConditionalFlowMatching 
import utils

def train():
    os.makedirs(CONFIG.EXP_FOLDER, exist_ok=True)
    device = torch.device(CONFIG.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # 1. Récupération des 3 loaders
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Batch de vérification (optionnel, sur le train loader)
    first_batch = next(iter(train_loader))
    utils.visualize_batch(first_batch[0], first_batch[1], first_batch[2], title="Training Batch Preview (Augmented)")

    print("Initialisation du modèle Conditional Flow Matching...")
    model = ConditionalFlowMatching(num_timesteps=100).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR, weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG.N_EPOCHS * len(train_loader)
    )

    # --- Historique pour les plots ---
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    print("Début de l'entraînement...")
    
    for epoch in range(CONFIG.N_EPOCHS):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        batch_losses = []
        
        for batch_idx, (images, z, mags) in enumerate(train_loader):
            images = images.to(device)
            z = z.to(device).unsqueeze(1)
            # Normalisation magnitudes (globale)
            mags_norm = (mags.to(device).unsqueeze(1) - CONFIG.MAG_MEAN) / CONFIG.MAG_STD

            optimizer.zero_grad()
            loss = model(images, z, mags_norm)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            batch_losses.append(loss.item())
            
        # Moyenne de l'époque train
        avg_train_loss = np.mean(batch_losses)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(scheduler.get_last_lr()[0])

        # --- PHASE DE VALIDATION (Sans gradient) ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, z, mags in val_loader:
                images = images.to(device)
                z = z.to(device).unsqueeze(1)
                mags_norm = (mags.to(device).unsqueeze(1) - CONFIG.MAG_MEAN) / CONFIG.MAG_STD
                
                loss = model(images, z, mags_norm)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)

        # --- LOGGING ---
        print(f"Epoch {epoch+1}/{CONFIG.N_EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        # Sauvegarde régulière (Checkpoint)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG.EXP_FOLDER, f"cfm_model_epoch_{epoch+1}.pt"))

    # --- FIN : SAUVEGARDE & PLOT ---
    print("Entraînement terminé. Génération des courbes...")
    
    # 1. Sauvegarde des données brutes
    np.savez(os.path.join(CONFIG.EXP_FOLDER, 'training_history.npz'), 
             train_loss=history['train_loss'], 
             val_loss=history['val_loss'],
             lr=history['lr'])

    # 2. Plot de la Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Flow Matching Loss (MSE)')
    plt.title('Convergence de l\'entraînement (Train vs Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot_path = os.path.join(CONFIG.EXP_FOLDER, 'loss_curve.png')
    plt.savefig(save_plot_path)
    print(f"Courbe de Loss sauvegardée : {save_plot_path}")

if __name__ == "__main__":
    train()