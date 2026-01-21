def statistical_report(z_true, z_pred, mags, title="Analyse_Complete"):
    """
    Génère le rapport statistique complet avec 3 panneaux :
    1. Résidus vs Redshift (Biais Z)
    2. Résidus vs Magnitude (Biais de sélection / Robustesse)
    3. Distribution des erreurs (Gaussianité)
    """
    # 1. Calcul des Résidus Normalisés
    residuals = (z_pred - z_true) / (1 + z_true)
    
    # 2. Métriques
    bias = np.mean(residuals)
    sigma_nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    
    # Outliers (> 0.15)
    outlier_mask = np.abs(residuals) > 0.15
    eta = np.sum(outlier_mask) / len(z_true) * 100
    
    print(f"\n--- RAPPORT STATISTIQUE : {title} ---")
    print(f"Biais moyen       : {bias:.5f}")
    print(f"Sigma (NMAD)      : {sigma_nmad:.5f}")
    print(f"Outliers (>0.15)  : {eta:.2f}%")
    
    # 3. Plots de Diagnostic (1 ligne, 3 colonnes)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Plot A : Résidus vs Redshift ---
    axes[0].scatter(z_true, residuals, s=1, alpha=0.3, c='k')
    axes[0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0].axhline(0.15, color='orange', linestyle=':')
    axes[0].axhline(-0.15, color='orange', linestyle=':')
    axes[0].set_xlabel("True Redshift")
    axes[0].set_ylabel("$\Delta z / (1+z)$")
    axes[0].set_title(f"Biais vs Redshift")
    axes[0].set_ylim(-0.3, 0.3)
    axes[0].grid(True, alpha=0.3)

    # --- Plot B : Résidus vs Magnitude (NOUVEAU) ---
    axes[1].scatter(mags, residuals, s=1, alpha=0.3, c='blue')
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].axhline(0.15, color='orange', linestyle=':')
    axes[1].axhline(-0.15, color='orange', linestyle=':')
    axes[1].set_xlabel("Magnitude i (AB)")
    axes[1].set_title("Robustesse vs Magnitude\n(Vérifier absence de plongeon)")
    axes[1].set_ylim(-0.3, 0.3)
    # Inverse l'axe x car les mags élevées = objets faibles
    # axes[1].invert_xaxis() 
    axes[1].grid(True, alpha=0.3)

    # --- Plot C : Histogramme ---
    bins = np.linspace(-0.2, 0.2, 100)
    axes[2].hist(residuals, bins=bins, density=True, alpha=0.6, color='gray', label='Data')
    x = np.linspace(-0.2, 0.2, 200)
    pdf = (1 / (sigma_nmad * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - bias) / sigma_nmad)**2)
    axes[2].plot(x, pdf, 'r-', linewidth=2, label=f'Gauss (NMAD={sigma_nmad:.3f})')
    axes[2].set_title("Distribution des Erreurs")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG.EXP_FOLDER, f"residuals_FULL_{title}.png")
    plt.savefig(save_path)
    print(f"Grand Chelem Statistique sauvegardé : {save_path}")