import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config import CONFIG
from data_loader import get_dataloaders
from backbone import GalaxyEquivariantMDN, MDNLoss
from utils import set_global_seed, statistical_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_synthetic_loader(path: str, batch_size: int = CONFIG.SYNTH_BATCH_SIZE) -> DataLoader:
    '''
    actions : Charge les données synthétiques depuis le stockage avec prise en charge du format de conditionnement V2.
    inputs : path (str), batch_size (int)
    appels : os.path.exists, np.load, torch.tensor, TensorDataset, DataLoader
    outputs : Instance de DataLoader PyTorch
    '''
    if not os.path.exists(path):
        candidates = sorted([f for f in os.listdir(CONFIG.EXP_FOLDER) if f.startswith("synthetic") and f.endswith(".npz")])
        if candidates:
            path = os.path.join(CONFIG.EXP_FOLDER, candidates[-1])
            logger.info(f"Fichier auto-détecté : {path}")
        else:
            raise FileNotFoundError(f"Aucun dataset synthétique trouvé dans {CONFIG.EXP_FOLDER}")

    logger.info(f"Chargement synthétique : {path}")
    data = np.load(path)

    x = torch.tensor(data['x'], dtype=torch.float32)

    if 'cond' in data:
        z = torch.tensor(data['cond'][:, 0], dtype=torch.float32)
    else:
        z = torch.tensor(data['z'], dtype=torch.float32)

    ds = TensorDataset(x, z)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=CONFIG.NUM_WORKERS)

def train_epoch_mdn(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device) -> float:
    '''
    actions : Exécute une époque d'entraînement complète optimisée pour la régression par Mixture Density Network.
    inputs : model (torch.nn.Module), loader (DataLoader), optimizer (torch.optim.Optimizer), criterion (torch.nn.Module), device (torch.device)
    appels : model.train, optimizer.zero_grad, model.forward, criterion.forward, loss.backward, optimizer.step
    outputs : Moyenne des pertes (float)
    '''
    model.train()
    losses = []

    for batch in loader:
        if len(batch) == 2:
            x, z = batch
            cond = None
        else:
            x, cond = batch
            z = cond[:, 0]

        x, z = x.to(device), z.to(device)
        optimizer.zero_grad()

        pi, mu, sigma = model(x)
        loss = criterion(pi, mu, sigma, z)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))

def evaluate_and_report(model: torch.nn.Module, loader: DataLoader, device: torch.device, title: str = "Evaluation") -> None:
    '''
    actions : Évalue le modèle sur un jeu de données et génère le rapport statistique basé sur l'espérance mathématique de la mixture.
    inputs : model (torch.nn.Module), loader (DataLoader), device (torch.device), title (str)
    appels : model.eval, torch.no_grad, torch.sum, statistical_report
    outputs : None
    '''
    model.eval()
    preds, trues, mags_list = [], [], []

    with torch.no_grad():
        for x, cond in loader:
            x = x.to(device)
            z_true = cond[:, 0].to(device)
            mag_val = cond[:, 1].to(device)

            pi, mu, sigma = model(x)

            expected_mu = torch.sum(pi * mu, dim=1)

            preds.extend(expected_mu.cpu().numpy())
            trues.extend(z_true.cpu().numpy())
            mags_list.extend(mag_val.cpu().numpy())

    statistical_report(np.array(trues), np.array(preds), np.array(mags_list), title=title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_syn", type=int, default=15)
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_gaussians", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=CONFIG.SYNTH_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device(CONFIG.DEVICE)

    syn_loader = load_synthetic_loader(CONFIG.exp_path(CONFIG.SYNTHETIC_50K), batch_size=args.batch_size)
    train_real, _, test_real = get_dataloaders(batch_size=args.batch_size, num_workers=CONFIG.NUM_WORKERS)

    backbone = GalaxyEquivariantMDN(num_gaussians=args.num_gaussians, use_meta=False).to(device)
    criterion = MDNLoss()

    logger.info(">>> [1/2] Pré-entraînement Synthétique (Sim2Real Zero-Shot)...")
    optimizer = optim.Adam(backbone.parameters(), lr=args.lr)

    for epoch in range(args.epochs_syn):
        loss_val = train_epoch_mdn(backbone, syn_loader, optimizer, criterion, device)
        logger.info(f"[Synth] Epoch {epoch+1}/{args.epochs_syn} | NLL Loss: {loss_val:.4f}")

    logger.info(">>> Évaluation Zero-Shot...")
    evaluate_and_report(backbone, test_real, device, title="1_ZeroShot_MDN_GCNN")

    logger.info(">>> [2/2] Fine-Tuning Réel (Domain Adaptation)...")
    optimizer_ft = optim.Adam(backbone.parameters(), lr=args.lr * 0.1)

    for epoch in range(args.ft_epochs):
        loss_val = train_epoch_mdn(backbone, train_real, optimizer_ft, criterion, device)
        logger.info(f"[Real] Epoch {epoch+1}/{args.ft_epochs} | NLL Loss: {loss_val:.4f}")

    logger.info(">>> Évaluation Finale...")
    evaluate_and_report(backbone, test_real, device, title="2_FineTuned_MDN_GCNN")
