import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as enn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GalaxyEquivariantMDN(nn.Module):
    '''
    actions : Extrait les caractéristiques géométriques invariantes (isotropie spatiale) et photométriques pour prédire les paramètres d'un Mixture Density Network.
    inputs : num_gaussians (int), use_meta (bool), meta_dim (int)
    appels : escnn.gspaces.flipRot2dOnR2, escnn.nn.FieldType, escnn.nn.SequentialModule, escnn.nn.R2Conv, escnn.nn.InnerBatchNorm, escnn.nn.ReLU, escnn.nn.PointwiseMaxPool, escnn.nn.GroupPooling
    outputs : Instance de modèle PyTorch équivariant
    '''

    def __init__(self, num_gaussians: int = 3, use_meta: bool = True, meta_dim: int = 5) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.use_meta = use_meta

        self.r2_act = gspaces.flipRot2dOnR2(N=8)

        self.in_type = enn.FieldType(self.r2_act, 6 * [self.r2_act.trivial_repr])

        out_type1 = enn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr])
        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, out_type1, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type1),
            enn.ReLU(out_type1, inplace=True),
            enn.PointwiseMaxPool(out_type1, kernel_size=2)
        )

        out_type2 = enn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            enn.R2Conv(out_type1, out_type2, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type2),
            enn.ReLU(out_type2, inplace=True),
            enn.PointwiseMaxPool(out_type2, kernel_size=2)
        )

        out_type3 = enn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            enn.R2Conv(out_type2, out_type3, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type3),
            enn.ReLU(out_type3, inplace=True),
            enn.PointwiseMaxPool(out_type3, kernel_size=2)
        )

        out_type4 = enn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr])
        self.block4 = enn.SequentialModule(
            enn.R2Conv(out_type3, out_type4, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type4),
            enn.ReLU(out_type4, inplace=True),
            enn.PointwiseMaxPool(out_type4, kernel_size=2)
        )

        self.gpool = enn.GroupPooling(out_type4)

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        num_image_features = 256

        meta_out_dim = 0
        if self.use_meta:
            meta_out_dim = 128
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, meta_out_dim),
                nn.BatchNorm1d(meta_out_dim),
                nn.ReLU()
            )
        else:
            self.meta_mlp = nn.Identity()

        input_dim_final = num_image_features + meta_out_dim

        self.shared_features = nn.Sequential(
            nn.Linear(input_dim_final, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.mdn_head = nn.Linear(128, 3 * self.num_gaussians)
        logger.info(f"Initialisation GalaxyEquivariantMDN terminée (G-CNN D8, K={self.num_gaussians}).")

    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        actions : Effectue la passe avant géométrique et photométrique pour générer les paramètres du mélange gaussien.
        inputs : x (torch.Tensor), meta (Optional[torch.Tensor])
        appels : escnn.nn.GeometricTensor, self.block1, self.block2, self.block3, self.block4, self.gpool, self.spatial_pool, self.meta_mlp, self.shared_features, self.mdn_head
        outputs : pi (torch.Tensor), mu (torch.Tensor), sigma (torch.Tensor)
        '''
        geo_x = enn.GeometricTensor(x, self.in_type)

        geo_h = self.block1(geo_x)
        geo_h = self.block2(geo_h)
        geo_h = self.block3(geo_h)
        geo_h = self.block4(geo_h)

        geo_h = self.gpool(geo_h)
        
        img_features = geo_h.tensor
        
        img_features = self.spatial_pool(img_features)
        img_features = torch.flatten(img_features, 1)

        if self.use_meta:
            if meta is None:
                raise ValueError("Le mode hybride est activé, un tenseur meta est requis.")
            meta_features = self.meta_mlp(meta)
            combined = torch.cat([img_features, meta_features], dim=1)
        else:
            combined = img_features

        features = self.shared_features(combined)
        mdn_out = self.mdn_head(features)

        pi_logits = mdn_out[:, :self.num_gaussians]
        mu = mdn_out[:, self.num_gaussians:2 * self.num_gaussians]
        sigma_pre = mdn_out[:, 2 * self.num_gaussians:]

        pi = F.softmax(pi_logits, dim=1)
        sigma = F.softplus(sigma_pre) + 1e-6

        return pi, mu, sigma

class MDNLoss(nn.Module):
    '''
    actions : Calcule la Negative Log-Likelihood stable pour un Mixture Density Network.
    inputs : None
    appels : torch.distributions.Normal
    outputs : Instance de fonction de perte
    '''

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        actions : Évalue la perte NLL via la méthode log-sum-exp pour la stabilité numérique.
        inputs : pi (torch.Tensor), mu (torch.Tensor), sigma (torch.Tensor), target (torch.Tensor)
        appels : torch.distributions.Normal, torch.logsumexp
        outputs : loss (torch.Tensor)
        '''
        target = target.unsqueeze(1).expand_as(mu)
        
        normal_dist = torch.distributions.Normal(mu, sigma)
        log_prob_per_gaussian = normal_dist.log_prob(target)
        
        log_pi = torch.log(pi + 1e-8)
        weighted_log_prob = log_pi + log_prob_per_gaussian
        log_prob_mixture = torch.logsumexp(weighted_log_prob, dim=1)
        
        return -torch.mean(log_prob_mixture)