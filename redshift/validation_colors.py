import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG

# Configuration pour un look "Papier Scientifique"
plt.style.use('default')
sns.set_theme(style="ticks", context="paper", font_scale=1.4)

def load_synthetic_loader(batch_size=CONFIG.BATCH_SIZE):
    """
    Charge le dernier dataset synthetique disponible dans le dossier d'experiences.
    """
    candidates = sorted(
        f for f in os.listdir(CONFIG.EXP_FOLDER)
        if f.startswith("synthetic") and f.endswith(".npz")
    )
    if not candidates:
        raise FileNotFoundError(f"Aucun dataset synthetique trouve dans {CONFIG.EXP_FOLDER}")

    path = os.path.join(CONFIG.EXP_FOLDER, candidates[-1])
    data = np.load(path)
    x = torch.tensor(data["x"], dtype=torch.float32)
    cond_key = "cond" if "cond" in data else "z"
    cond = torch.tensor(data[cond_key], dtype=torch.float32)

    return DataLoader(TensorDataset(x, cond), batch_size=batch_size, shuffle=False, num_workers=CONFIG.NUM_WORKERS)

def load_data_for_physics():
    """
    Charge les données réelles et simulées et extrait proprement les couleurs.
    """
    from data_loader import get_dataloaders
    
    print("Chargement des données pour analyse physique...")
    
    # --- REAL DATA ---
    train_real, _, _ = get_dataloaders()
    real_colors = []
    
    # On prend juste un sous-ensemble pour aller vite (ex: 5000 objets)
    count = 0
    for x, _ in train_real:
        if count > 5000:
            break
        # Calcul flux : Somme des pixels (Approximation)
        fluxes = torch.sum(x, dim=(2, 3)).numpy()
        real_colors.append(fluxes)
        count += len(x)
        
    real_fluxes = np.concatenate(real_colors, axis=0)
    
    # --- SIM DATA ---
    syn_loader = load_synthetic_loader(batch_size=CONFIG.BATCH_SIZE)
    sim_colors = []
    count = 0
    for x, _ in syn_loader:
        if count > 5000:
            break
        fluxes = torch.sum(x, dim=(2, 3)).numpy()
        sim_colors.append(fluxes)
        count += len(x)
        
    sim_fluxes = np.concatenate(sim_colors, axis=0)
    
    return real_fluxes, sim_fluxes

def compute_colors(fluxes):
    """
    Convertit les flux bruts en couleurs (g-r, r-i).
    Gère les log négatifs et filtre les outliers.
    """
    # Éviter log(<=0)
    fluxes = np.maximum(fluxes, 1e-3)
    mags = -2.5 * np.log10(fluxes)
    
    # Indices bandes : 0=u, 1=g, 2=r, 3=i, 4=z, 5=y
    g_r = mags[:, 1] - mags[:, 2]
    r_i = mags[:, 2] - mags[:, 3]
    
    return g_r, r_i

def clean_data(x, y):
    """Enlève les valeurs extrêmes (Top 1% et Bottom 1%) pour le plot"""
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Filtrage quantile pour enlever les artefacts de simulation
    q_low = df.quantile(0.01)
    q_high = df.quantile(0.99)
    
    df_clean = df[
        (df['x'] > q_low['x']) & (df['x'] < q_high['x']) &
        (df['y'] > q_low['y']) & (df['y'] < q_high['y'])
    ]
    return df_clean['x'].values, df_clean['y'].values

def plot_color_color_contours(real_fluxes, sim_fluxes):
    """
    Trace le diagramme SOTA : Contours Réels vs Contours Sim
    """
    gr_r, ri_r = compute_colors(real_fluxes)
    gr_s, ri_s = compute_colors(sim_fluxes)
    
    # Nettoyage
    gr_r, ri_r = clean_data(gr_r, ri_r)
    gr_s, ri_s = clean_data(gr_s, ri_s)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 1. REAL DATA (Contours Gris/Noir - Le "Background")
    sns.kdeplot(x=gr_r, y=ri_r, ax=ax, 
                levels=5, color=".2", linewidths=1, fill=True, alpha=0.1)
    sns.kdeplot(x=gr_r, y=ri_r, ax=ax, 
                levels=5, color=".2", linewidths=2, label='Real Data (COSMOS)')
    
    # 2. SIM DATA (Contours Colorés - Le "Test")
    # On utilise "magma" ou "viridis" pour montrer la densité
    sns.kdeplot(x=gr_s, y=ri_s, ax=ax, 
                levels=5, cmap="magma", linewidths=2, label='Sim Data (CFM)')
    
    ax.set_xlabel(r'Color $(g - r)$')
    ax.set_ylabel(r'Color $(r - i)$')
    ax.set_title('Validation: Color-Color Distribution')
    
    # Légende manuelle pour les contours
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='.2', lw=2),
                    Line2D([0], [0], color='purple', lw=2)]
    ax.legend(custom_lines, ['Real Data', 'Sim Data'], loc='upper left')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.EXP_FOLDER, "PHYSICS_color_contours.png"), dpi=150)
    print("Sauvegardé : PHYSICS_color_contours.png")

def plot_1d_distribution(real_fluxes, sim_fluxes):
    """
    Trace les histogrammes 1D superposés.
    """
    gr_r, _ = compute_colors(real_fluxes)
    gr_s, _ = compute_colors(sim_fluxes)
    
    # On filtre juste sur g-r
    df = pd.DataFrame({'Real': gr_r})
    df = df[(df['Real'] > df['Real'].quantile(0.01)) & (df['Real'] < df['Real'].quantile(0.99))]
    
    df2 = pd.DataFrame({'Sim': gr_s})
    df2 = df2[(df2['Sim'] > df2['Sim'].quantile(0.01)) & (df2['Sim'] < df2['Sim'].quantile(0.99))]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.histplot(df['Real'], stat="density", color="grey", alpha=0.4, label="Real", element="step")
    sns.histplot(df2['Sim'], stat="density", color="orange", alpha=0.4, label="Sim", element="step")
    sns.kdeplot(df['Real'], color="grey", linewidth=2)
    sns.kdeplot(df2['Sim'], color="orange", linewidth=2)
    
    ax.set_xlabel(r'Color $(g - r)$')
    ax.set_title('1D Color Distribution Match')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.EXP_FOLDER, "PHYSICS_1d_dist.png"), dpi=150)
    print("Sauvegardé : PHYSICS_1d_dist.png")

if __name__ == "__main__":
    r_flux, s_flux = load_data_for_physics()
    plot_color_color_contours(r_flux, s_flux)
    plot_1d_distribution(r_flux, s_flux)
