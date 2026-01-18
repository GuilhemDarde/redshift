import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Crée les embeddings sinusoïdaux pour le temps.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    
    return emb

class ConditionEncoder(nn.Module):
    """
    Encode le redshift et la magnitude dans un vecteur latent.
    """
    def __init__(self, output_dim=256):
        super().__init__()
        self.redshift_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )
        self.magnitude_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )
        self.combiner = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.SiLU()
        )

    def forward(self, redshift, magnitudes):
        z_emb = self.redshift_embed(redshift)
        m_emb = self.magnitude_embed(magnitudes)
        combined = torch.cat([z_emb, m_emb], dim=1)
        return self.combiner(combined)

class UNetBlock(nn.Module):
    """
    Bloc de base du U-Net.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.cond_proj = nn.Linear(cond_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, t_emb, cond_emb):
        residual = self.residual(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = h + self.cond_proj(cond_emb)[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + residual

class ImageDiffusionUNet(nn.Module):
    """
    Backbone U-Net (inchangé).
    """
    def __init__(self, in_channels=6, time_emb_dim=128, cond_emb_dim=256):
        super().__init__()
        self.channels = [64, 128, 256, 512]
        self.init_conv = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)
        
        self.down_blocks = nn.ModuleList([
            UNetBlock(self.channels[0], self.channels[0], time_emb_dim, cond_emb_dim),
            UNetBlock(self.channels[0], self.channels[1], time_emb_dim, cond_emb_dim),
            UNetBlock(self.channels[1], self.channels[2], time_emb_dim, cond_emb_dim),
            UNetBlock(self.channels[2], self.channels[3], time_emb_dim, cond_emb_dim),
        ])
        
        self.down_samples = nn.ModuleList([
            nn.Conv2d(self.channels[0], self.channels[0], 3, stride=2, padding=1),
            nn.Conv2d(self.channels[1], self.channels[1], 3, stride=2, padding=1),
            nn.Conv2d(self.channels[2], self.channels[2], 3, stride=2, padding=1),
        ])
        
        self.mid_block = UNetBlock(self.channels[3], self.channels[3], time_emb_dim, cond_emb_dim)
        
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(self.channels[3], self.channels[2], 2, stride=2),
            nn.ConvTranspose2d(self.channels[2], self.channels[1], 2, stride=2),
            nn.ConvTranspose2d(self.channels[1], self.channels[0], 2, stride=2),
        ])
        
        self.up_blocks = nn.ModuleList([
            UNetBlock(self.channels[2] * 2, self.channels[2], time_emb_dim, cond_emb_dim),
            UNetBlock(self.channels[1] * 2, self.channels[1], time_emb_dim, cond_emb_dim),
            UNetBlock(self.channels[0] * 2, self.channels[0], time_emb_dim, cond_emb_dim),
            UNetBlock(self.channels[0], self.channels[0], time_emb_dim, cond_emb_dim),
        ])
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, self.channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channels[0], in_channels, 3, padding=1)
        )
    
    def forward(self, x, t_emb, cond_emb):
        h = self.init_conv(x)
        skips = []
        
        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb, cond_emb)
            if i < 3:
                skips.append(h)
                h = self.down_samples[i](h)
        
        h = self.mid_block(h, t_emb, cond_emb)
        
        for i, block in enumerate(self.up_blocks):
            if i < 3:
                h = self.up_samples[i](h)
                h = torch.cat([h, skips[2-i]], dim=1)
            h = block(h, t_emb, cond_emb)
            
        return self.final_conv(h)

class ConditionalFlowMatching(nn.Module):
    """
    Implémentation Optimal Transport Conditional Flow Matching (OT-CFM).
    Remplace la classe ImageDiffusion.
    """
    def __init__(self, num_timesteps=100): 
        super().__init__()
        self.condition_encoder = ConditionEncoder(output_dim=256)
        self.denoiser = ImageDiffusionUNet(in_channels=6, time_emb_dim=128, cond_emb_dim=256)
        self.sigma_min = 1e-4

    def forward(self, x_1, redshifts, magnitudes):
        """
        Training : Calcule la Loss sur le champ de vecteurs.
        x_1 : Images réelles [B, 6, 64, 64]
        """
        B = x_1.shape[0]
        device = x_1.device
        
        # 1. Encodage Condition
        cond_emb = self.condition_encoder(redshifts, magnitudes)
        
        # 2. Sampling du temps t ~ U[0, 1]
        t = torch.rand(B, device=device)
        
        # 3. Source (Bruit Gaussien) x_0
        x_0 = torch.randn_like(x_1)
        
        # 4. Interpolation Linéaire (OT path)
        t_view = t.view(B, 1, 1, 1)
        # x_t part de x_0 (t=0) vers x_1 (t=1)
        x_t = (1 - (1 - self.sigma_min) * t_view) * x_0 + t_view * x_1
        
        # 5. Cible : Le vecteur direction est simplement x_1 - x_0 (approximativement)
        v_target = x_1 - (1 - self.sigma_min) * x_0
        
        # 6. Prédiction
        t_emb = get_timestep_embedding(t * 1000, 128) # Scaling
        v_pred = self.denoiser(x_t, t_emb, cond_emb)
        
        # 7. Loss MSE
        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def generate(self, redshifts, magnitudes, num_steps=50):
        """
        Inférence : Résolution ODE (Euler).
        """
        B = redshifts.shape[0]
        device = redshifts.device
        cond_emb = self.condition_encoder(redshifts, magnitudes)
        
        # État initial : Bruit x_0
        x = torch.randn(B, 6, 64, 64, device=device)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = i / num_steps
            t_batch = torch.full((B,), t_val * 1000, device=device)
            t_emb = get_timestep_embedding(t_batch, 128)
            
            v_pred = self.denoiser(x, t_emb, cond_emb)
            x = x + v_pred * dt
            
        return x