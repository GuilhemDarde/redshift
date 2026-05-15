import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from analysis_utils import magnitude_bin_edges, magnitude_support_definition_rows, magnitude_support_mask, write_rows_csv
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


class CFMGenerationWrapper(nn.Module):
    '''
    actions : Expose les méthodes génératives du CFM via forward pour permettre torch.nn.DataParallel.
    inputs : base_cfm (ConditionalFlowMatching)
    appels : ConditionalFlowMatching.generate, augment_image_to_image, partial_invert, reconstruct_from_latent
    outputs : Instance de CFMGenerationWrapper
    '''
    def __init__(self, base_cfm: ConditionalFlowMatching) -> None:
        super().__init__()
        self.base_cfm = base_cfm

    def forward(
        self,
        cond_vector: torch.Tensor,
        x_real: torch.Tensor = None,
        partner_x: torch.Tensor = None,
        partner_cond: torch.Tensor = None,
        mode: str = "global",
        t0: float = 0.55,
        noise_scale: float = 0.08,
        alpha: float = 0.25,
        num_steps: int = 50,
    ) -> torch.Tensor:
        if mode in {"global", "targeted_global"}:
            return self.base_cfm.generate(cond_vector, num_steps=num_steps)
        if mode == "i2i":
            if x_real is None:
                raise ValueError("x_real est requis pour le mode i2i.")
            return self.base_cfm.augment_image_to_image(
                x_real,
                cond_vector,
                t0=t0,
                noise_scale=noise_scale,
                num_steps=num_steps,
            )
        if mode == "interp":
            if x_real is None or partner_x is None or partner_cond is None:
                raise ValueError("x_real, partner_x et partner_cond sont requis pour le mode interp.")
            latent_source = self.base_cfm.partial_invert(x_real, cond_vector, t0=t0, num_steps=num_steps)
            latent_partner = self.base_cfm.partial_invert(partner_x, partner_cond, t0=t0, num_steps=num_steps)
            latent = (1.0 - alpha) * latent_source + alpha * latent_partner
            if noise_scale > 0.0:
                latent = latent + noise_scale * torch.randn_like(latent)
            return self.base_cfm.reconstruct_from_latent(latent, cond_vector, t0=t0, num_steps=num_steps)
        raise ValueError(f"Mode de génération inconnu: {mode}")


def load_cfm(checkpoint: str, device: torch.device) -> ConditionalFlowMatching:
    model = ConditionalFlowMatching(num_timesteps=CONFIG.TIMESTEPS).to(device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint CFM introuvable: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def build_generator(model: ConditionalFlowMatching, enabled: bool) -> nn.Module:
    generator = CFMGenerationWrapper(model)
    if enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Activation DataParallel génération sur %s GPU visibles.", torch.cuda.device_count())
        return nn.DataParallel(generator)
    if enabled:
        logger.info("DataParallel génération demandé mais un seul GPU est visible.")
    return generator


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


def select_source_pool(
    metadata: dict,
    split_indices: dict,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, dict]:
    """
    Selectionne les galaxies source a augmenter.

    Par defaut on suit le retour prof: les sources viennent des bins de magnitude
    i les moins representes dans le train. Le ciblage RA/DEC reste disponible en
    legacy pour relire les anciens runs.
    """
    n = len(metadata["mag_i"])
    train_indices = split_indices["train"]
    density = np.full(n, np.nan, dtype=np.float64)
    radius = np.full(n, np.nan, dtype=np.float64)
    density_threshold = float("nan")

    mag_edges = magnitude_bin_edges(args.mag_i_min, args.mag_i_max, args.mag_i_bins)
    low_mag_mask, mag_threshold, mag_support, mag_bin, mag_counts = magnitude_support_mask(
        metadata["mag_i"],
        metadata["mag_i"][train_indices],
        mag_edges,
        quantile=args.low_mag_support_quantile,
    )

    if args.selection_target == "all_train":
        selected_pool = train_indices
        target_description = "all_train"
    elif args.selection_target == "faint_mag":
        selected_pool = train_indices[metadata["mag_i"][train_indices] >= args.faint_mag_threshold]
        target_description = f"faint_mag>={args.faint_mag_threshold:.3f}"
    elif args.selection_target == "low_mag_support":
        selected_pool = train_indices[low_mag_mask[train_indices]]
        target_description = f"low_mag_support<=count {mag_threshold:.6g}"
    elif args.selection_target == "low_density":
        density, radius = compute_train_knn_density(metadata["ra"], metadata["dec"], train_indices, k=args.knn_k)
        low_density, density_threshold = low_density_mask(density, train_indices, quantile=args.low_density_quantile)
        selected_pool = train_indices[low_density[train_indices]]
        target_description = f"legacy_low_density<=rho {density_threshold:.6g}"
    else:
        raise ValueError(f"selection_target inconnu: {args.selection_target}")

    context = {
        "density": density,
        "radius": radius,
        "density_threshold": density_threshold,
        "mag_i_edges": mag_edges,
        "low_mag_support_mask": low_mag_mask,
        "mag_support_count": mag_support,
        "mag_bin": mag_bin,
        "mag_support_threshold": mag_threshold,
        "mag_bin_train_counts": mag_counts,
        "target_description": target_description,
    }
    return np.asarray(selected_pool, dtype=np.int64), context


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
    generator: nn.Module,
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
            x_gen = generator(
                batch_cond,
                mode=args.mode,
                num_steps=args.steps,
            )
            append_batch(
                images, conds, sources, partners, densities, modes, strengths,
                x_gen, batch_cond_np, batch_sources, np.full(len(batch_sources), -1, dtype=np.int64),
                batch_density, args.mode, 1.0,
            )
            continue

        batch_x = torch.tensor(dataset_x[batch_sources], dtype=torch.float32, device=device)
        if args.mode in {"i2i", "both"}:
            x_gen = generator(
                batch_cond,
                x_real=batch_x,
                mode="i2i",
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
            x_gen = generator(
                source_cond,
                x_real=source_x,
                partner_x=partner_x,
                partner_cond=partner_cond,
                mode="interp",
                t0=args.t0,
                noise_scale=args.noise_scale,
                alpha=args.alpha,
                num_steps=args.steps,
            )
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
        field=args.field,
        sample_filter=args.sample_filter,
        max_files=args.max_files,
        n_folds=args.n_folds if args.fold_id is not None else None,
        fold_id=args.fold_id,
        cache_path=args.cache_path,
        split_strategy=args.split_strategy,
    )
    metadata = build_metadata(dataset, split_indices=split_indices)
    selected_pool, target_context = select_source_pool(metadata, split_indices, args)

    selected = sample_indices(selected_pool, args.limit_sources, args.seed)

    if len(selected) == 0:
        raise RuntimeError("Aucune source sélectionnée pour la génération.")

    expanded_sources = repeated_sources(selected, args.n_aug_per_source)
    partner_for_selected = choose_partner_indices(dataset.data["cond"], selected, selected)
    partner_lookup = {int(src): int(partner) for src, partner in zip(selected, partner_for_selected)}
    expanded_partners = np.asarray([partner_lookup[int(src)] for src in expanded_sources], dtype=np.int64)

    logger.info(
        "Sources sélectionnées: %s | candidats attendus par mode: %s | cible: %s",
        len(selected),
        len(expanded_sources),
        target_context["target_description"],
    )
    model = load_cfm(args.checkpoint or CONFIG.exp_path(CONFIG.CFM_CHECKPOINT), device)
    generator = build_generator(model, args.data_parallel)
    generator.eval()
    x, cond, source_idx, partner_idx, source_density, mode, strength = generate_for_indices(
        generator,
        dataset.data["x"],
        dataset.data["cond"],
        target_context["density"],
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
        selection_target=np.array(args.selection_target),
        density_threshold=np.array(target_context["density_threshold"], dtype=np.float64),
        knn_radius=target_context["radius"][source_idx],
        mag_i=metadata["mag_i"][source_idx],
        mag_bin=target_context["mag_bin"][source_idx],
        mag_support_count=target_context["mag_support_count"][source_idx],
        low_mag_support_threshold=np.array(target_context["mag_support_threshold"], dtype=np.float64),
        mag_i_bin_edges=target_context["mag_i_edges"],
        mag_i_bin_train_counts=target_context["mag_bin_train_counts"],
    )
    write_rows_csv(
        os.path.splitext(output)[0] + "_mag_support_definition.csv",
        magnitude_support_definition_rows(
            target_context["mag_i_edges"],
            target_context["mag_bin_train_counts"],
            target_context["mag_support_threshold"],
        ),
    )
    logger.info("Candidats sauvegardés: %s (%s images)", output, len(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["global", "targeted_global", "i2i", "interp", "both"], default="i2i")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--region", choices=["all", "stripe82"], default="all")
    parser.add_argument("--field", type=str, default="all")
    parser.add_argument("--sample_filter", choices=["all", "spec"], default="spec")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=CONFIG.N_FOLDS)
    parser.add_argument("--fold_id", type=int, default=None)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--split_strategy", choices=["spatial", "marie_regular"], default="spatial")
    parser.add_argument("--selection_target", choices=["low_mag_support", "faint_mag", "all_train", "low_density"], default="low_mag_support")
    parser.add_argument("--faint_mag_threshold", type=float, default=23.5)
    parser.add_argument("--mag_i_min", type=float, default=CONFIG.I_MIN)
    parser.add_argument("--mag_i_max", type=float, default=CONFIG.I_MAX)
    parser.add_argument("--mag_i_bins", type=int, default=14)
    parser.add_argument("--low_mag_support_quantile", type=float, default=0.20)
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
    parser.add_argument("--data_parallel", action="store_true")
    run(parser.parse_args())
