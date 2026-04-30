import argparse
import logging
import math
import os

import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm

from config import CONFIG
from data_loader import get_dataloaders
from model import ConditionalFlowMatching, OT_CFM_Physics_Wrapper
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
actions : Extrait un échantillon de vecteurs de conditionnement (redshift, magnitude, morphologie) depuis les données réelles pour guider la génération.
inputs : n_samples (int), loader (torch.utils.data.DataLoader)
appels : torch.cat
outputs : torch.Tensor
'''
def sample_conditions(n_samples: int, loader: torch.utils.data.DataLoader) -> torch.Tensor:
    conditions = []
    collected = 0
    for _, cond in loader:
        conditions.append(cond)
        collected += cond.shape[0]
        if collected >= n_samples:
            break
            
    return torch.cat(conditions, dim=0)[:n_samples]

'''
actions : Charge le générateur OT-CFM optimisé avec la régularisation photométrique depuis les poids sauvegardés.
inputs : model_path (str), device (torch.device)
appels : ConditionalFlowMatching, OT_CFM_Physics_Wrapper, torch.load
outputs : torch.nn.Module
'''
def load_generator(model_path: str, device: torch.device) -> torch.nn.Module:
    base_cfm = ConditionalFlowMatching(num_timesteps=CONFIG.TIMESTEPS)
    model = OT_CFM_Physics_Wrapper(base_cfm, lambda_photo=0.01)
    
    model.base_cfm.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model.base_cfm

'''
actions : Exécute l'inférence massive par lots pour synthétiser les images astrophysiques à partir du bruit stochastique.
inputs : model (torch.nn.Module), conditions (torch.Tensor), batch_size (int), device (torch.device)
appels : range, tqdm, torch.no_grad, amp.autocast, model.generate, np.concatenate
outputs : np.ndarray
'''
def generate_images(model: torch.nn.Module, conditions: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    n_samples = conditions.shape[0]
    n_batches = math.ceil(n_samples / batch_size)
    generated_images = []

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Génération V3 (Physique)"):
            batch_cond = conditions[i * batch_size : (i + 1) * batch_size].to(device, non_blocking=True)
            
            with amp.autocast():
                x_gen = model.generate(batch_cond, num_steps=50)
                
            generated_images.append(x_gen.cpu().numpy())

    return np.concatenate(generated_images, axis=0)

'''
actions : Pilote le processus de génération du nouveau jeu de données synthétique contraint par la physique.
inputs : args (argparse.Namespace)
appels : torch.device, get_dataloaders, sample_conditions, load_generator, generate_images, os.path.join, np.savez
outputs : None
'''
def run_generation(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    device = torch.device(CONFIG.DEVICE)
    model_path = args.checkpoint or CONFIG.exp_path(CONFIG.CFM_CHECKPOINT)
    output_path = args.output or CONFIG.exp_path(f"synthetic_cosmos_{args.n // 1000}k_v3.npz")

    if os.path.exists(output_path):
        logger.warning(f"Le fichier {output_path} existe déjà. Écrasement en cours.")

    logger.info(f"Cible : {args.n} galaxies synthétiques (Dim=7).")
    
    train_loader, _, _ = get_dataloaders(batch_size=CONFIG.BATCH_SIZE, num_workers=CONFIG.NUM_WORKERS)
    cond_tensor = sample_conditions(args.n, train_loader)
    
    logger.info(f"Chargement du modèle : {model_path}")
    generator = load_generator(model_path, device)
    
    logger.info("Démarrage de l'inférence ODE...")
    x_synthetic = generate_images(generator, cond_tensor, args.batch_size, device)
    
    logger.info("Sauvegarde sur disque...")
    np.savez(output_path, x=x_synthetic, cond=cond_tensor.numpy())
    logger.info(f"Dataset V3 généré avec succès : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=CONFIG.GENERATION_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    run_generation(parser.parse_args())
