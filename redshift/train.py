import argparse
import contextlib
import json
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cfm_conditioning import build_cfm_condition, build_cfm_photo_targets, condition_choices, condition_dim
from config import CONFIG
from data_loader import get_dataset_and_splits
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


class CFMArrayDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        cond: np.ndarray,
        indices: np.ndarray,
        target_mags: np.ndarray = None,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.cond = np.asarray(cond, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_mags = None if target_mags is None else np.asarray(target_mags, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        x = torch.tensor(self.x[real_idx], dtype=torch.float32)
        cond = torch.tensor(self.cond[real_idx], dtype=torch.float32)
        if self.target_mags is None:
            return x, cond
        return x, cond, torch.tensor(self.target_mags[real_idx], dtype=torch.float32)


def build_cfm_loaders(
    args: argparse.Namespace,
) -> tuple:
    dataset, split_indices = get_dataset_and_splits(
        region=args.region,
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
        split_strategy=args.split_strategy,
    )
    cond = build_cfm_condition(dataset.data, schema=args.condition_schema)
    target_mags = build_cfm_photo_targets(dataset.data, schema=args.condition_schema)

    train_ds = CFMArrayDataset(dataset.data["x"], cond, split_indices["train"], target_mags=target_mags)
    val_ds = CFMArrayDataset(dataset.data["x"], cond, split_indices["val"], target_mags=target_mags)
    test_ds = CFMArrayDataset(dataset.data["x"], cond, split_indices["test"], target_mags=target_mags)
    pin_memory = torch.device(CONFIG.DEVICE).type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    logger.info(
        "Split %s : Train=%s, Val=%s, Test=%s | CFM condition=%s Dim=%s",
        args.split_strategy,
        len(split_indices["train"]),
        len(split_indices["val"]),
        len(split_indices["test"]),
        args.condition_schema,
        cond.shape[1],
    )
    return dataset, split_indices, train_loader, val_loader, test_loader, target_mags


def estimate_band_zero_points(
    x: np.ndarray,
    target_mags: np.ndarray,
    train_indices: np.ndarray,
    flux_mode: str,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if target_mags is None:
        return np.asarray(CONFIG.MAG_MEAN, dtype=np.float32)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    if max_samples is not None and max_samples > 0 and len(train_indices) > max_samples:
        rng = np.random.default_rng(seed)
        train_indices = np.sort(rng.choice(train_indices, size=max_samples, replace=False))

    images = np.asarray(x[train_indices], dtype=np.float64)
    linear = np.sinh(np.clip(images, -20.0, 20.0)) if CONFIG.ASINH_NORM else images
    if flux_mode == "positive":
        linear = np.clip(linear, 0.0, None)
    fluxes = np.sum(linear, axis=(2, 3))
    mags = np.asarray(target_mags[train_indices], dtype=np.float64)

    zero_points = np.full(fluxes.shape[1], np.nan, dtype=np.float64)
    for band_idx in range(fluxes.shape[1]):
        mask = np.isfinite(fluxes[:, band_idx]) & np.isfinite(mags[:, band_idx]) & (fluxes[:, band_idx] > 1e-8)
        if np.any(mask):
            zero_points[band_idx] = float(np.nanmedian(mags[mask, band_idx] + 2.5 * np.log10(fluxes[mask, band_idx])))

    finite = np.isfinite(zero_points)
    fill_value = float(np.nanmedian(zero_points[finite])) if np.any(finite) else CONFIG.MAG_MEAN
    zero_points[~finite] = fill_value
    logger.info("Zéro-points photométriques CFM calibrés sur train: %s", np.array2string(zero_points, precision=6))
    return zero_points.astype(np.float32)


def unpack_cfm_batch(batch, device: torch.device):
    x = batch[0].to(device, non_blocking=True)
    cond = batch[1].to(device, non_blocking=True)
    target_mags = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
    return x, cond, target_mags

'''
actions : Exécute une passe d'apprentissage complète sur le jeu de données d'entraînement.
inputs : model (torch.nn.Module), loader (torch.utils.data.DataLoader), optimizer (torch.optim.Optimizer), scaler (any), device (torch.device)
appels : get_autocast, optimizer.zero_grad, scaler.scale, scaler.step, scaler.update
outputs : float
'''
def train_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, scaler: any, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, cond, target_mags = unpack_cfm_batch(batch, device)
        
        optimizer.zero_grad(set_to_none=True)
        with get_autocast(device.type):
            loss, _ = model(x, cond, target_mags=target_mags)
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
        for batch in loader:
            x, cond, target_mags = unpack_cfm_batch(batch, device)
            
            with get_autocast(device.type):
                loss, _ = model(x, cond, target_mags=target_mags)
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
    dataset, split_indices, train_loader, val_loader, _, target_mags = build_cfm_loaders(args)
    photo_mag_zp = estimate_band_zero_points(
        dataset.data["x"],
        target_mags,
        split_indices["train"],
        flux_mode=args.photo_flux_mode,
        max_samples=args.zp_max_samples,
        seed=args.seed,
    )
    
    base_cfm = ConditionalFlowMatching(
        num_timesteps=CONFIG.TIMESTEPS,
        condition_dim=condition_dim(args.condition_schema),
    )
    model = OT_CFM_Physics_Wrapper(
        base_cfm,
        lambda_photo=args.lambda_photo,
        lambda_color=args.lambda_color,
        mag_zp=photo_mag_zp,
        flux_mode=args.photo_flux_mode,
    ).to(device)
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
    metadata_path = save_path + ".json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "condition_schema": args.condition_schema,
                "condition_dim": condition_dim(args.condition_schema),
                "lambda_photo": args.lambda_photo,
                "lambda_color": args.lambda_color,
                "photo_flux_mode": args.photo_flux_mode,
                "photo_mag_zp": np.asarray(photo_mag_zp).tolist(),
                "split_strategy": args.split_strategy,
                "fold_id": args.fold_id,
                "n_folds": args.n_folds,
                "field": args.field,
                "sample_filter": args.sample_filter,
                "best_val_loss": best_loss,
            },
            f,
            indent=2,
        )
    logger.info("Métadonnées CFM sauvegardées : %s", metadata_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=CONFIG.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CONFIG.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    parser.add_argument("--lr", type=float, default=CONFIG.LR)
    parser.add_argument("--lambda_photo", type=float, default=0.01)
    parser.add_argument("--lambda_color", type=float, default=0.0)
    parser.add_argument("--condition_schema", choices=condition_choices(), default="legacy7")
    parser.add_argument("--photo_flux_mode", choices=["positive", "signed"], default="signed")
    parser.add_argument("--zp_max_samples", type=int, default=20000)
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
