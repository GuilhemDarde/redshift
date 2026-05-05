import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from config import CONFIG
from data_loader import build_metadata, get_dataset_and_splits
from density_utils import compute_train_knn_density, low_density_mask
from model import ConditionalFlowMatching
from utils import set_global_seed

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_cfm(checkpoint: str, device: torch.device) -> ConditionalFlowMatching:
    model = ConditionalFlowMatching(num_timesteps=CONFIG.TIMESTEPS).to(device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint CFM introuvable: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def sample_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def choose_partner_indices(cond: np.ndarray, source_indices: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    '''
    actions : Associe à chaque source un voisin proche dans l'espace conditionnel 7D.
    inputs : cond (np.ndarray), source_indices (np.ndarray), candidate_indices (np.ndarray)
    appels : cKDTree, np.argsort
    outputs : np.ndarray
    '''
    if len(candidate_indices) < 2:
        return np.full_like(source_indices, -1, dtype=np.int64)

    candidate_cond = cond[candidate_indices]
    mean = np.nanmean(candidate_cond, axis=0)
    std = np.nanstd(candidate_cond, axis=0) + 1e-6
    candidate_norm = (candidate_cond - mean) / std
    source_norm = (cond[source_indices] - mean) / std

    if cKDTree is not None:
        tree = cKDTree(candidate_norm)
        _, nn = tree.query(source_norm, k=min(3, len(candidate_indices)))
        if nn.ndim == 1:
            nn = nn[:, None]
        partners = []
        for src, row in zip(source_indices, nn):
            row_candidates = candidate_indices[row]
            other = row_candidates[row_candidates != src]
            partners.append(int(other[0]) if len(other) else -1)
        return np.asarray(partners, dtype=np.int64)

    partners = []
    for src, vec in zip(source_indices, source_norm):
        dist = np.sum((candidate_norm - vec[None, :]) ** 2, axis=1)
        order = np.argsort(dist)
        chosen = -1
        for pos in order:
            idx = int(candidate_indices[pos])
            if idx != src:
                chosen = idx
                break
        partners.append(chosen)
    return np.asarray(partners, dtype=np.int64)


def repeated_sources(source_indices: np.ndarray, n_aug_per_source: int) -> np.ndarray:
    if n_aug_per_source <= 0:
        raise ValueError("n_aug_per_source doit etre strictement positif.")
    return np.repeat(np.asarray(source_indices, dtype=np.int64), n_aug_per_source)


def append_batch(
    images: List[np.ndarray],
    conds: List[np.ndarray],
    sources: List[np.ndarray],
    partners: List[np.ndarray],
    densities: List[np.ndarray],
    modes: List[np.ndarray],
    strengths: List[np.ndarray],
    x_gen: torch.Tensor,
    cond: np.ndarray,
    source_idx: np.ndarray,
    partner_idx: np.ndarray,
    density: np.ndarray,
    mode: str,
    strength: float,
) -> None:
    images.append(x_gen.detach().cpu().numpy())
    conds.append(cond.astype(np.float32))
    sources.append(source_idx.astype(np.int64))
    partners.append(partner_idx.astype(np.int64))
    densities.append(density.astype(np.float64))
    modes.append(np.full(len(source_idx), mode, dtype="<U24"))
    strengths.append(np.full(len(source_idx), strength, dtype=np.float32))


@torch.no_grad()
def generate_for_indices(
    model: ConditionalFlowMatching,
    dataset_x: np.ndarray,
    dataset_cond: np.ndarray,
    density: np.ndarray,
    source_indices: np.ndarray,
    partner_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images: List[np.ndarray] = []
    conds: List[np.ndarray] = []
    sources: List[np.ndarray] = []
    partners: List[np.ndarray] = []
    densities: List[np.ndarray] = []
    modes: List[np.ndarray] = []
    strengths: List[np.ndarray] = []

    n_batches = int(np.ceil(len(source_indices) / args.batch_size))
    for batch_id in tqdm(range(n_batches), desc=f"Génération CFM {args.mode}"):
        lo = batch_id * args.batch_size
        hi = min((batch_id + 1) * args.batch_size, len(source_indices))
        batch_sources = source_indices[lo:hi]
        batch_partners = partner_indices[lo:hi]
        batch_cond_np = dataset_cond[batch_sources].astype(np.float32)
        batch_cond = torch.tensor(batch_cond_np, dtype=torch.float32, device=device)
        batch_density = density[batch_sources]

        if args.mode in {"global", "targeted_global"}:
            x_gen = model.generate(batch_cond, num_steps=args.steps)
            append_batch(
                images, conds, sources, partners, densities, modes, strengths,
                x_gen, batch_cond_np, batch_sources, np.full(len(batch_sources), -1, dtype=np.int64),
                batch_density, args.mode, 1.0,
            )
            continue

        batch_x = torch.tensor(dataset_x[batch_sources], dtype=torch.float32, device=device)
        if args.mode in {"i2i", "both"}:
            x_gen = model.augment_image_to_image(
                batch_x,
                batch_cond,
                t0=args.t0,
                noise_scale=args.noise_scale,
                num_steps=args.steps,
            )
            append_batch(
                images, conds, sources, partners, densities, modes, strengths,
                x_gen, batch_cond_np, batch_sources, np.full(len(batch_sources), -1, dtype=np.int64),
                batch_density, "i2i", args.noise_scale,
            )

        if args.mode in {"interp", "both"}:
            valid_partner = batch_partners >= 0
            if not np.any(valid_partner):
                continue
            interp_sources = batch_sources[valid_partner]
            interp_partners = batch_partners[valid_partner]
            source_x = torch.tensor(dataset_x[interp_sources], dtype=torch.float32, device=device)
            partner_x = torch.tensor(dataset_x[interp_partners], dtype=torch.float32, device=device)
            source_cond_np = dataset_cond[interp_sources].astype(np.float32)
            partner_cond_np = dataset_cond[interp_partners].astype(np.float32)
            source_cond = torch.tensor(source_cond_np, dtype=torch.float32, device=device)
            partner_cond = torch.tensor(partner_cond_np, dtype=torch.float32, device=device)
            latent_source = model.partial_invert(source_x, source_cond, t0=args.t0, num_steps=args.steps)
            latent_partner = model.partial_invert(partner_x, partner_cond, t0=args.t0, num_steps=args.steps)
            latent = (1.0 - args.alpha) * latent_source + args.alpha * latent_partner
            if args.noise_scale > 0.0:
                latent = latent + args.noise_scale * torch.randn_like(latent)
            x_gen = model.reconstruct_from_latent(latent, source_cond, t0=args.t0, num_steps=args.steps)
            append_batch(
                images, conds, sources, partners, densities, modes, strengths,
                x_gen, source_cond_np, interp_sources, interp_partners, density[interp_sources],
                "interp", args.alpha,
            )

    if not images:
        raise RuntimeError("Aucune image générée. Vérifiez les indices source et le mode demandé.")

    return (
        np.concatenate(images, axis=0),
        np.concatenate(conds, axis=0),
        np.concatenate(sources, axis=0),
        np.concatenate(partners, axis=0),
        np.concatenate(densities, axis=0),
        np.concatenate(modes, axis=0),
        np.concatenate(strengths, axis=0),
    )


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    device = torch.device(CONFIG.DEVICE)
    dataset, split_indices = get_dataset_and_splits(
        region=args.region,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
    )
    metadata = build_metadata(dataset, split_indices=split_indices)
    density, radius = compute_train_knn_density(metadata["ra"], metadata["dec"], split_indices["train"], k=args.knn_k)
    low_mask_all, density_threshold = low_density_mask(density, split_indices["train"], quantile=args.low_density_quantile)

    if args.mode == "global":
        selected = sample_indices(split_indices["train"], args.limit_sources, args.seed)
    else:
        low_train = split_indices["train"][low_mask_all[split_indices["train"]]]
        selected = sample_indices(low_train, args.limit_sources, args.seed)

    if len(selected) == 0:
        raise RuntimeError("Aucune source sélectionnée pour la génération.")

    expanded_sources = repeated_sources(selected, args.n_aug_per_source)
    partner_for_selected = choose_partner_indices(dataset.data["cond"], selected, selected)
    partner_lookup = {int(src): int(partner) for src, partner in zip(selected, partner_for_selected)}
    expanded_partners = np.asarray([partner_lookup[int(src)] for src in expanded_sources], dtype=np.int64)

    logger.info(
        "Sources sélectionnées: %s | candidats attendus par mode: %s | densité seuil %.6g",
        len(selected),
        len(expanded_sources),
        density_threshold,
    )
    model = load_cfm(args.checkpoint or CONFIG.exp_path(CONFIG.CFM_CHECKPOINT), device)
    x, cond, source_idx, partner_idx, source_density, mode, strength = generate_for_indices(
        model,
        dataset.data["x"],
        dataset.data["cond"],
        density,
        expanded_sources,
        expanded_partners,
        args,
        device,
    )

    output = args.output or CONFIG.exp_path(f"cfm_aug_candidates_{args.mode}.npz")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez(
        output,
        x=x.astype(np.float32),
        cond=cond.astype(np.float32),
        source_index=source_idx,
        partner_index=partner_idx,
        local_density=source_density,
        mode=mode,
        strength=strength,
        density_threshold=np.array(density_threshold, dtype=np.float64),
        knn_radius=radius[source_idx],
    )
    logger.info("Candidats sauvegardés: %s (%s images)", output, len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["global", "targeted_global", "i2i", "interp", "both"], default="i2i")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--low_density_quantile", type=float, default=0.20)
    parser.add_argument("--limit_sources", type=int, default=None)
    parser.add_argument("--n_aug_per_source", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=CONFIG.GENERATION_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--t0", type=float, default=0.55)
    parser.add_argument("--noise_scale", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=CONFIG.SEED)
    run(parser.parse_args())
