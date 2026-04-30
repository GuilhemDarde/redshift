import argparse
import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
actions : Calcule les métriques standard de cosmologie pour l'évaluation des redshifts photométriques.
inputs : z_true (np.ndarray), z_pred (np.ndarray)
appels : np.abs, np.median, np.mean
outputs : Tuple[float, float]
'''
def compute_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> Tuple[float, float]:
    delta_z = (z_pred - z_true) / (1.0 + z_true)
    sigma_nmad = 1.4826 * np.median(np.abs(delta_z - np.median(delta_z)))
    outlier_rate = np.mean(np.abs(delta_z) > 0.15) * 100.0
    return float(sigma_nmad), float(outlier_rate)

'''
actions : Génère et sauvegarde une figure de qualité publication comparant les prédictions au vecteur de vérité terrain.
inputs : z_true (np.ndarray), z_pred (np.ndarray), sigma_nmad (float), outlier_rate (float), output_path (str)
appels : plt.subplots, ax.scatter, ax.plot, ax.set_xlabel, ax.set_ylabel, ax.legend, fig.savefig, plt.close
outputs : None
'''
def plot_scatter_results(z_true: np.ndarray, z_pred: np.ndarray, sigma_nmad: float, outlier_rate: float, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(z_true, z_pred, s=1, alpha=0.1, color='blue', label='Prédictions MDN')
    
    z_max = max(z_true.max(), z_pred.max())
    z_range = np.linspace(0, z_max, 100)
    
    ax.plot(z_range, z_range, color='black', linestyle='--', label='Idéal ($z_{pred} = z_{true}$)')
    ax.plot(z_range, z_range + 0.15 * (1 + z_range), color='red', linestyle=':', label='Limite Outliers')
    ax.plot(z_range, z_range - 0.15 * (1 + z_range), color='red', linestyle=':')
    
    ax.set_xlabel('Redshift Spectroscopique ($z_{true}$)', fontsize=12)
    ax.set_ylabel('Redshift Photométrique Prédit ($z_{pred}$)', fontsize=12)
    ax.set_xlim([0, z_max])
    ax.set_ylim([0, z_max])
    ax.set_title(f'Performance V3 - $\\sigma_{{NMAD}}$: {sigma_nmad:.4f} | Outliers: {outlier_rate:.2f}%', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

'''
actions : Charge le fichier de résultats numpy, évalue les performances et produit le rapport visuel.
inputs : args (argparse.Namespace)
appels : os.path.join, os.path.exists, np.load, compute_metrics, plot_scatter_results
outputs : None
'''
def evaluate_results(args: argparse.Namespace) -> None:
    file_path = CONFIG.exp_path(CONFIG.SOTA_RESULTS)
    if not os.path.exists(file_path):
        logger.error(f"Fichier introuvable : {file_path}")
        return

    logger.info(f"Analyse du fichier de résultats : {file_path}")
    data = np.load(file_path)
    
    z_true = data['z_true']
    z_pred = data['z_pred']

    sigma_nmad, outlier_rate = compute_metrics(z_true, z_pred)
    
    logger.info("--- RÉSULTATS V3 (Physics-Informed) ---")
    logger.info(f"Sigma NMAD : {sigma_nmad:.5f}")
    logger.info(f"Outliers   : {outlier_rate:.2f}%")

    output_plot = os.path.join(CONFIG.EXP_FOLDER, "evaluation_v3.png")
    plot_scatter_results(z_true, z_pred, sigma_nmad, outlier_rate, output_plot)
    logger.info(f"Graphe de dispersion sauvegardé : {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    evaluate_results(parser.parse_args())
