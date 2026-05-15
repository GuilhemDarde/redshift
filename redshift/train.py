import argparse
import contextlib
import logging
import os

import torch
import torch.optim as optim

from config import CONFIG
from data_loader import get_dataloaders
from model import ConditionalFlowMatching, OT_CFM_Physics_Wrapper
from utils import set_global_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

'''
actions : Résout dynamiquement l'API de précision mixte selon la version de PyTorch disponible pour satisfaire à la fois l'exécution et l'analyseur statique.
inputs : device_type (str)
appels : hasattr, torch.amp.autocast, torch.cuda.amp.autocast, contextlib.nullcontext
outputs : contextlib.AbstractContextManager
'''
def get_autocast(device_type: str = 'cuda'):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type)
    elif hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()

'''
actions : Fournit l'objet GradScaler adapté à la version de PyTorch.
inputs : Aucun
appels : hasattr, torch.amp.GradScaler, torch.cuda.amp.GradScaler
outputs : any
'''
def get_scaler():
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler(device='cuda')
    return torch.cuda.amp.GradScaler()


def maybe_data_parallel(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Activation DataParallel sur %s GPU visibles.", torch.cuda.device_count())
        return torch.nn.DataParallel(model)
    if enabled:
        logger.info("DataParallel demandé mais un seul GPU est visible.")
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    return loss.mean() if loss.ndim > 0 else loss

'''
actions : Exécute une passe d'apprentissage complète sur le jeu de données d'entraînement.
inputs : model (torch.nn.Module), loader (torch.utils.data.DataLoader), optimizer (torch.optim.Optimizer), scaler (any), device (torch.device)
appels : get_autocast, optimizer.zero_grad, scaler.scale, scaler.step, scaler.update
outputs : float
'''
def train_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, scaler: any, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for x, cond in loader:
        x = x.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with get_autocast(device.type):
            loss, _ = model(x, cond)
            loss = scalar_loss(loss)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    return total_loss / len(loader)

'''
actions : Évalue les performances du générateur sur le jeu de données de validation.
inputs : model (torch.nn.Module), loader (torch.utils.data.DataLoader), device (torch.device)
appels : torch.no_grad, get_autocast
outputs : float
'''
def validate_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, cond in loader:
            x = x.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            
            with get_autocast(device.type):
                loss, _ = model(x, cond)
                loss = scalar_loss(loss)
            total_loss += loss.item()
            
    return total_loss / len(loader)

'''
actions : Point d'entrée principal orchestrant l'initialisation, la boucle temporelle et la sauvegarde des poids.
inputs : args (argparse.Namespace)
appels : get_dataloaders, ConditionalFlowMatching, OT_CFM_Physics_Wrapper, optim.Adam, get_scaler, train_epoch, validate_epoch, os.path.join, torch.save
outputs : None
'''
def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    device = torch.device(CONFIG.DEVICE)
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        region=args.region,
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
        split_strategy=args.split_strategy,
    )
    
    base_cfm = ConditionalFlowMatching(num_timesteps=CONFIG.TIMESTEPS)
    model = OT_CFM_Physics_Wrapper(base_cfm, lambda_photo=args.lambda_photo).to(device)
    model = maybe_data_parallel(model, args.data_parallel)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = get_scaler()
    best_loss = float('inf')
    save_path = args.output_checkpoint or CONFIG.exp_path(CONFIG.CFM_CHECKPOINT)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for ep in range(args.epochs):
        t_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        v_loss = validate_epoch(model, val_loader, device)
        
        logger.info(f"Epoch {ep+1}/{args.epochs} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(unwrap_model(model).base_cfm.state_dict(), save_path)
            
    logger.info(f"Meilleur modèle OT-CFM sauvegardé : {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=CONFIG.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CONFIG.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--lr", type=float, default=CONFIG.LR)
    parser.add_argument("--lambda_photo", type=float, default=0.01)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="all")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--split_strategy", choices=["spatial", "marie_regular", "marie_strict"], default="spatial")
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--data_parallel", action="store_true")
    train(parser.parse_args())
