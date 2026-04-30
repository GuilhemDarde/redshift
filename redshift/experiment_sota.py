import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import amp

from config import CONFIG
from data_loader import get_dataloaders
from backbone import GalaxyEquivariantMDN, MDNLoss
from utils import set_global_seed

import torch.multiprocessing as mp

# Force la méthode 'spawn' pour éviter les deadlocks CUDA avec les DataLoaders
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

def load_synthetic_loader(path: str, batch_size: int = CONFIG.SYNTH_BATCH_SIZE) -> DataLoader:
    '''
    actions : Charge le dataset synthétique en optimisant le débit binaire via le pin_memory et le multiprocessing.
    inputs : path (str), batch_size (int)
    appels : os.path.exists, np.load, torch.tensor, TensorDataset, DataLoader
    outputs : DataLoader
    '''
    if not os.path.exists(path):
        candidates = sorted([f for f in os.listdir(CONFIG.EXP_FOLDER) if f.startswith("synthetic") and f.endswith(".npz")])
        path = os.path.join(CONFIG.EXP_FOLDER, candidates[-1]) if candidates else path

    logger.info(f"Chargement synthétique optimisé : {path}")
    data = np.load(path)
    x = torch.tensor(data['x'], dtype=torch.float32)
    z = torch.tensor(data['cond'][:, 0] if 'cond' in data else data['z'], dtype=torch.float32)

    return DataLoader(
        TensorDataset(x, z), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=CONFIG.SYNTH_NUM_WORKERS, 
        pin_memory=True
    )

def train_one_model(model_idx: int, args: argparse.Namespace, train_real: DataLoader, syn_loader: DataLoader, device: torch.device) -> torch.nn.Module:
    '''
    actions : Orchestre l'apprentissage d'un modèle unitaire avec sauvegarde de l'état (checkpointing) pour la tolérance aux pannes.
    inputs : model_idx (int), args (argparse.Namespace), train_real (DataLoader), syn_loader (DataLoader), device (torch.device)
    appels : os.path.join, os.path.exists, torch.load, GalaxyEquivariantMDN, optim.Adam, MDNLoss, amp.GradScaler, amp.autocast, torch.save
    outputs : torch.nn.Module
    '''
    model_path = CONFIG.exp_path(f"sota_model_gcnn_mdn_{model_idx}.pt")
    model = GalaxyEquivariantMDN(num_gaussians=args.num_gaussians, use_meta=False).to(device)
    
    if os.path.exists(model_path):
        logger.info(f"--- [Modèle {model_idx+1}/{args.n_models}] Restauration depuis {model_path} ---")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    logger.info(f"--- [Modèle {model_idx+1}/{args.n_models}] Entraînement Initial ---")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = MDNLoss()
    scaler = amp.GradScaler()

    phases = [
        ("Synthétique", syn_loader, args.epochs_syn, 1.0),
        ("Fine-Tuning", train_real, args.ft_epochs, 0.1)
    ]

    for name, loader, epochs, lr_mult in phases:
        if epochs <= 0: continue
        for pg in optimizer.param_groups: pg['lr'] = args.lr * lr_mult
        
        model.train()
        for ep in range(epochs):
            losses = []
            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                z = (batch[1][:, 0] if batch[1].dim() > 1 else batch[1]).to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast():
                    pi, mu, sigma = model(x)
                    loss = criterion(pi, mu, sigma, z)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            
            if (ep+1) % 5 == 0:
                logger.info(f"   {name} | Ep {ep+1}/{epochs} | Loss: {np.mean(losses):.4f}")

    torch.save(model.state_dict(), model_path)
    logger.info(f"   Modèle sauvegardé : {model_path}")
    return model

def predict_ensemble(models: List[torch.nn.Module], loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    actions : Exécute une inférence parallélisée sur l'ensemble de modèles pour quantifier les incertitudes totales.
    inputs : models (List[torch.nn.Module]), loader (DataLoader), device (torch.device)
    appels : amp.autocast, torch.no_grad, torch.sum, np.stack, np.mean, np.var
    outputs : Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    mus_all, vars_all, z_trues = [], [], []
    for m in models: m.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            z = batch[1][:, 0] if batch[1].dim() > 1 else batch[1]
            z_trues.append(z.numpy())

            b_mus, b_vars = [], []
            with amp.autocast():
                for model in models:
                    pi, mu, sigma = model(x)
                    e_mu = torch.sum(pi * mu, dim=1)
                    e_var = torch.sum(pi * (sigma**2 + mu**2), dim=1) - e_mu**2
                    b_mus.append(e_mu.cpu().float().numpy())
                    b_vars.append(e_var.cpu().float().numpy())

            mus_all.append(np.stack(b_mus))
            vars_all.append(np.stack(b_vars))

    z_t = np.concatenate(z_trues, axis=0)
    m_s = np.concatenate(mus_all, axis=1)
    v_s = np.concatenate(vars_all, axis=1)
    
    return z_t, np.mean(m_s, axis=0), np.sqrt(np.mean(v_s, axis=0) + np.var(m_s, axis=0))

def run_sota_experiment(args: argparse.Namespace) -> None:
    '''
    actions : Pilote le workflow complet de l'expérience SOTA avec agrégation des métriques finales.
    inputs : args (argparse.Namespace)
    appels : load_synthetic_loader, get_dataloaders, train_one_model, predict_ensemble, np.savez
    outputs : None
    '''
    set_global_seed(args.seed)
    device = torch.device(CONFIG.DEVICE)
    synthetic_path = args.synthetic_path or CONFIG.exp_path(CONFIG.SYNTHETIC_100K)
    syn_loader = load_synthetic_loader(synthetic_path, args.batch_size)
    train_real, _, test_real = get_dataloaders(batch_size=args.batch_size, num_workers=CONFIG.NUM_WORKERS)

    models = [train_one_model(i, args, train_real, syn_loader, device) for i in range(args.n_models)]

    logger.info(">>> Évaluation de l'Ensemble G-CNN MDN (Inférence AMP)...")
    z_true, z_pred, z_sigma = predict_ensemble(models, test_real, device)
    dz = (z_pred - z_true) / (1.0 + z_true)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    outliers = np.mean(np.abs(dz) > 0.15) * 100.0

    logger.info(f"--- RÉSULTATS FINAUX ---")
    logger.info(f"Sigma NMAD : {nmad:.5f} | Outliers : {outliers:.2f}%")
    np.savez(CONFIG.exp_path(CONFIG.SOTA_RESULTS), z_true=z_true, z_pred=z_pred, z_sigma=z_sigma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_syn", type=int, default=45)
    parser.add_argument("--ft_epochs", type=int, default=15)
    parser.add_argument("--n_models", type=int, default=5)
    parser.add_argument("--num_gaussians", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=CONFIG.SYNTH_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--synthetic_path", type=str, default=None)
    run_sota_experiment(parser.parse_args())
