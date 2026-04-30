import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    '''
    actions : Calcule l'encodage positionnel sinusoïdal pour les pas de temps continus du générateur de flux.
    inputs : timesteps (torch.Tensor), embedding_dim (int)
    appels : math.log, torch.exp, torch.arange, torch.cat, torch.sin, torch.cos, torch.zeros_like
    outputs : torch.Tensor
    '''
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    
    return emb


class ConditionEncoder(nn.Module):
    '''
    actions : Projette le vecteur physique continu (photométrie et morphologie) dans un espace latent riche pour le conditionnement.
    inputs : input_dim (int), output_dim (int)
    appels : super, nn.Sequential, nn.Linear, nn.SiLU
    outputs : Instance de ConditionEncoder
    '''
    def __init__(self, input_dim: int = 7, output_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        actions : Exécute la transformation non-linéaire du vecteur conditionnel.
        inputs : x (torch.Tensor)
        appels : self.net
        outputs : torch.Tensor
        '''
        return self.net(x)


class ConditionalDenoiser(nn.Module):
    '''
    actions : Modèle l'architecture U-Net pour la prédiction du champ de vecteurs conditionné par le temps et la physique spatiale.
    inputs : in_channels (int), cond_emb_dim (int)
    appels : super, nn.Sequential, nn.Linear, nn.SiLU, nn.Conv2d, nn.GroupNorm, nn.ConvTranspose2d
    outputs : Instance de ConditionalDenoiser
    '''
    def __init__(self, in_channels: int = 6, cond_emb_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(128, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim)
        )
        
        self.cond_proj = nn.Linear(cond_emb_dim, cond_emb_dim)
        
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        
        self.emb_proj1 = nn.Linear(cond_emb_dim, 128)
        self.emb_proj2 = nn.Linear(cond_emb_dim, 256)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        '''
        actions : Estime la dynamique spatiale du flux conditionnel via des blocs résiduels et temporels.
        inputs : x (torch.Tensor), t_emb (torch.Tensor), cond_emb (torch.Tensor)
        appels : self.time_embed, self.cond_proj, self.conv_in, self.down1, self.down2, self.emb_proj1, self.emb_proj2, self.mid, self.up1, self.up2, self.conv_out
        outputs : torch.Tensor
        '''
        t_hidden = self.time_embed(t_emb)
        c_hidden = self.cond_proj(cond_emb)
        emb_combined = t_hidden + c_hidden
        
        h1 = self.conv_in(x)
        
        h2 = self.down1(h1)
        emb1 = self.emb_proj1(emb_combined)[..., None, None]
        h2 = h2 + emb1
        
        h3 = self.down2(h2)
        emb2 = self.emb_proj2(emb_combined)[..., None, None]
        h3 = h3 + emb2
        
        h_mid = self.mid(h3)
        
        h_up1 = self.up1(h_mid) + h2
        h_up2 = self.up2(h_up1) + h1
        
        return self.conv_out(h_up2)


class ConditionalFlowMatching(nn.Module):
    '''
    actions : Implémente la logique d'adaptation de domaine par trajectoires de flux continus (OT-CFM).
    inputs : num_timesteps (int)
    appels : super, ConditionEncoder, ConditionalDenoiser, torch.rand, torch.randn_like, get_timestep_embedding, F.mse_loss
    outputs : Instance de ConditionalFlowMatching
    '''
    def __init__(self, num_timesteps: int = 100) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.condition_encoder = ConditionEncoder(input_dim=7, output_dim=256)
        self.denoiser = ConditionalDenoiser(in_channels=6, cond_emb_dim=256)
        self.sigma_min = 1e-4

    def estimate_x1_from_vector_field(self, x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        actions : Reconstruit l'estimation du point final x_1 a partir du champ de vecteurs predit au temps t.
        inputs : x_t (torch.Tensor), v_pred (torch.Tensor), t (torch.Tensor)
        appels : torch.Tensor.view
        outputs : torch.Tensor
        '''
        t_view = t.view(t.shape[0], 1, 1, 1)
        return (1.0 - self.sigma_min) * x_t + (1.0 - (1.0 - self.sigma_min) * t_view) * v_pred

    def forward(
        self,
        x_1: torch.Tensor,
        cond_vector: torch.Tensor,
        return_x1_pred: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        actions : Estime la perte de régression MSE sur l'interpolation temporelle entre le bruit et l'image.
        inputs : x_1 (torch.Tensor), cond_vector (torch.Tensor)
        appels : self.condition_encoder, torch.rand, torch.randn_like, get_timestep_embedding, self.denoiser, F.mse_loss
        outputs : torch.Tensor
        '''
        B = x_1.shape[0]
        device = x_1.device
        
        cond_emb = self.condition_encoder(cond_vector)
        
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        
        t_view = t.view(B, 1, 1, 1)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_view) * x_0 + t_view * x_1
        v_target = x_1 - (1.0 - self.sigma_min) * x_0
        
        t_emb = get_timestep_embedding(t * 1000.0, 128)
        v_pred = self.denoiser(x_t, t_emb, cond_emb)
        loss_vf = F.mse_loss(v_pred, v_target)
        
        if return_x1_pred:
            x1_pred = self.estimate_x1_from_vector_field(x_t, v_pred, t)
            return loss_vf, x1_pred

        return loss_vf

    @torch.no_grad()
    def generate(self, cond_vector: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        '''
        actions : Résout l'équation différentielle ordinaire d'Euler pour synthétiser des images à partir de bruit conditionné.
        inputs : cond_vector (torch.Tensor), num_steps (int)
        appels : self.condition_encoder, torch.randn, get_timestep_embedding, self.denoiser
        outputs : torch.Tensor
        '''
        B = cond_vector.shape[0]
        device = cond_vector.device
        
        cond_emb = self.condition_encoder(cond_vector)
        x_t = torch.randn(B, 6, 64, 64, device=device)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = i / num_steps
            t_tensor = torch.full((B,), t_val, device=device)
            
            t_emb = get_timestep_embedding(t_tensor * 1000.0, 128)
            v_pred = self.denoiser(x_t, t_emb, cond_emb)
            
            x_t = x_t + v_pred * dt
            
        return x_t


class PhysicsInformedLoss(nn.Module):
    '''
    actions : Calcule le terme de régularisation photométrique dans le domaine logarithmique des magnitudes pour garantir la stabilité des gradients.
    inputs : asinh_norm (bool), mag_zp (float)
    appels : super
    outputs : Instance de PhysicsInformedLoss
    '''
    def __init__(self, asinh_norm: bool = True, mag_zp: float = 23.563542) -> None:
        super().__init__()
        self.asinh_norm = asinh_norm
        self.mag_zp = mag_zp
        self.i_band_idx = 3

    def forward(self, x_pred: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        '''
        actions : Intègre le flux spatial de l'image, convertit la somme en magnitude apparente et pénalise l'écart via une perte de Huber robuste.
        inputs : x_pred (torch.Tensor), cond (torch.Tensor)
        appels : torch.clamp, torch.sinh, torch.sum, torch.log10, F.huber_loss
        outputs : torch.Tensor
        '''
        mag_i_target = cond[:, 1] * 2.0 + 22.0
        
        x_clipped = torch.clamp(x_pred, min=-10.0, max=10.0)
        img_linear = torch.sinh(x_clipped) if self.asinh_norm else x_clipped
        
        flux_pred = torch.sum(img_linear[:, self.i_band_idx, :, :], dim=(1, 2))
        flux_pred = torch.clamp(flux_pred, min=1e-8)
        
        mag_pred = self.mag_zp - 2.5 * torch.log10(flux_pred)
        
        return F.huber_loss(mag_pred, mag_i_target, delta=1.0)

class OT_CFM_Physics_Wrapper(nn.Module):
    '''
    actions : Encapsule le modèle génératif OT-CFM pour y intégrer la perte composite équilibrée.
    inputs : base_cfm (nn.Module), lambda_photo (float)
    appels : super, PhysicsInformedLoss
    outputs : Instance de OT_CFM_Physics_Wrapper
    '''
    def __init__(self, base_cfm: nn.Module, lambda_photo: float = 0.01) -> None:
        super().__init__()
        self.base_cfm = base_cfm
        self.lambda_photo = lambda_photo
        self.photo_loss_fn = PhysicsInformedLoss(asinh_norm=CONFIG.ASINH_NORM, mag_zp=CONFIG.MAG_MEAN)

    def forward(self, x_1: torch.Tensor, cond_vector: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        '''
        actions : Calcule la perte totale combinant le flot temporel classique et la contrainte de magnitude sur l'image finale estimee par le champ.
        inputs : x_1 (torch.Tensor), cond_vector (torch.Tensor)
        appels : self.base_cfm, self.photo_loss_fn
        outputs : Tuple[torch.Tensor, Dict[str, float]]
        '''
        loss_vf, x1_pred = self.base_cfm(x_1, cond_vector, return_x1_pred=True)
        loss_photo = self.photo_loss_fn(x1_pred, cond_vector)
        
        loss_total = loss_vf + self.lambda_photo * loss_photo
        
        metrics = {
            "loss_total": loss_total.item(),
            "loss_vf": loss_vf.item(),
            "loss_photo": loss_photo.item()
        }
        
        return loss_total, metrics
