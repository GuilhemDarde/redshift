# Rapport d'analyse des experiences redshift COSMOS

Artefacts analyses: `../../experiments/redshift_run_001`  
Date de generation des artefacts: 2026-05-01  
Pipeline: COSMOS NPZ + catalogue morphologique FITS + modeles CNN bins, Marie baseline et Marie+GCNN.

## 1. Resume executif

Le pipeline est operationnel sur le dataset COSMOS-US et produit les diagnostics demandes: analyse du dataset, heatmaps de dispersion, agregations par `z_pred`, agregations par bande I, cartes RA/DEC et comparaison de plusieurs architectures.

Le dataset final apres filtres et cross-match contient `162021` objets. Le split spatial canonique est `129616` objets en train, `16202` en validation et `16203` en test. Le cross-match complet effectue sur le serveur a donne `162021` objets matches sur `202889` objets apres filtres physiques, soit une efficacite de `79.86%`.

Conclusion modele: le CNN par bins sert de baseline classification mais reste le moins precis. La baseline Marie obtient la meilleure dispersion robuste (`sigma_NMAD=0.0345`), tandis que Marie+GCNN reduit le taux d'outliers (`3.78%`) et obtient le meilleur RMSE sur le split canonique (`0.2643`).

## 2. Protocole dataset

Les donnees d'entree sont les cubes COSMOS-US au format `.npz`, contenant les images multi-bandes, les metadonnees photometriques et les flags. Le catalogue morphologique `catalogue_morpho_cosmos.fits` fournit les colonnes `RAJ2000`, `DEJ2000`, `Re.G1` et `n.G1`. Le cross-match est effectue par plus proche voisin astrometrique avec une tolerance de `1 arcsec`.

Filtres appliques avant apprentissage:

- magnitude `18 <= I <= 25`;
- redshift spectroscopique `0.001 < z <= 6`;
- flags nuls sur les canaux utilises;
- morphologie valide (`Re > 0`, `n > 0`);
- cross-match FITS/NPZ a `1 arcsec`.

### Statistiques dataset

| Quantite | Valeur |
|---|---:|
| Objets totaux apres cross-match | 162021 |
| Train | 129616 |
| Validation | 16202 |
| Test | 16203 |
| `z_true` min / median / max | 0.005 / 0.921 / 5.670 |
| `mag I` min / median / max | 18.001 / 23.787 / 25.000 |
| RA min / median / max | 149.432 / 150.109 / 150.794 |
| DEC min / median / max | 1.576 / 2.219 / 2.885 |

### Repartition train/validation/test

Le split est spatial et deterministe par RA. La figure suivante montre que les sous-ensembles couvrent des zones distinctes du champ, ce qui limite la fuite spatiale entre apprentissage et evaluation.

![Repartition RA/DEC train-val-test](../../experiments/redshift_run_001/dataset_report/ra_dec_split.png)

### Distribution en redshift

La distribution en redshift est conservee entre les splits, avec une forte densite aux redshifts faibles et intermediaires. Les objets a haut redshift sont plus rares, ce qui doit etre garde en tete pour interpreter les bins `z_pred > 3`.

![Distribution redshift par split](../../experiments/redshift_run_001/dataset_report/z_distribution_by_split.png)

### Flags photometriques

Les flags bruts sont faibles dans toutes les bandes. La bande `u` concentre la plus grande fraction d'objets flagges (`0.449%`), tandis que les autres bandes restent autour de `0.05%`.

| Bande | Objets flagges | Total brut | Fraction |
|---|---:|---:|---:|
| u | 2063 | 459077 | 0.449% |
| g | 256 | 459077 | 0.056% |
| r | 240 | 459077 | 0.052% |
| i | 232 | 459077 | 0.051% |
| z | 229 | 459077 | 0.050% |
| y | 251 | 459077 | 0.055% |

![Flags bruts par bande](../../experiments/redshift_run_001/dataset_report/raw_flags_by_band.png)

### Bandes photometriques U, I et Z

Les distributions photometriques confirment que le dataset est domine par des objets faibles, notamment en bande I autour de `I~23-25`. Les bandes U, I et Z sont conservees comme axes de comparaison car elles encadrent des regimes photometriques differents.

![Focus magnitude U](../../experiments/redshift_run_001/dataset_report/focus_mag_u.png)

![Focus magnitude I](../../experiments/redshift_run_001/dataset_report/focus_mag_i.png)

![Focus magnitude Z](../../experiments/redshift_run_001/dataset_report/focus_mag_z.png)

Les distributions completes par bande sont disponibles dans:

- `dataset_report/mag_u_distribution_by_split.png`
- `dataset_report/mag_g_distribution_by_split.png`
- `dataset_report/mag_r_distribution_by_split.png`
- `dataset_report/mag_i_distribution_by_split.png`
- `dataset_report/mag_z_distribution_by_split.png`
- `dataset_report/mag_y_distribution_by_split.png`

## 3. Architectures evaluees

Trois familles de modeles ont ete comparees:

- **CNN bins**: classification par bins de redshift avec `CrossEntropyLoss`; la prediction continue est obtenue via le centre du bin predit.
- **Marie baseline**: modele local inspire du modele de Marie, combinant une branche image et une branche metadonnees, avec tete classification/regression.
- **Marie+GCNN**: premieres couches G-CNN equivariantes, puis tete de prediction style Marie.

Chaque modele a ete evalue sur le split canonique `80/10/10` et sur un `fold0` spatial avec `5 folds`, ou le test represente environ `20%` du dataset (`32405` objets).

## 4. Comparaison globale des performances

Metriques utilisees:

- `bias = mean((z_pred - z_true)/(1+z_true))`;
- `sigma_NMAD = 1.4826 * median(|dz - median(dz)|)`;
- `RMSE = sqrt(mean((z_pred - z_true)^2))`;
- outliers: fraction d'objets avec `|dz| > 0.15`.

### Split canonique 80/10/10

| Modele | n test | Bias | sigma_NMAD | RMSE | Outliers |
|---|---:|---:|---:|---:|---:|
| CNN bins | 16203 | 0.0181 | 0.0807 | 0.4159 | 8.33% |
| Marie baseline | 16203 | 0.0219 | **0.0345** | 0.2795 | 5.56% |
| Marie+GCNN | 16203 | **0.0158** | 0.0430 | **0.2643** | **3.78%** |

Interpretation: le modele CNN bins est nettement moins precis, ce qui est attendu pour une approche de classification discretisee. La baseline Marie donne la meilleure dispersion robuste. Marie+GCNN presente moins d'outliers et un RMSE plus faible, ce qui suggere une meilleure stabilite sur les erreurs fortes.

### Fold0 spatial

| Modele | n test | Bias | sigma_NMAD | RMSE | Outliers |
|---|---:|---:|---:|---:|---:|
| CNN bins fold0 | 32405 | **0.0038** | 0.0730 | 0.4574 | 8.95% |
| Marie baseline fold0 | 32405 | 0.0113 | **0.0382** | **0.2995** | 7.00% |
| Marie+GCNN fold0 | 32405 | 0.0090 | 0.0415 | 0.3011 | **5.09%** |

Interpretation: les tendances sont stables sur fold0. Marie baseline conserve la meilleure `sigma_NMAD`, tandis que Marie+GCNN garde l'avantage sur les outliers.

## 5. Analyse de dispersion des predictions

Les cartes de chaleur classiques permettent de visualiser les ecarts entre redshifts vrais et predits. Les figures ci-dessous montrent le comportement du modele Marie+GCNN, choisi comme modele principal pour l'analyse de dispersion car il obtient le meilleur compromis RMSE/outliers.

### Densite `z_true` vs `z_pred`

La diagonale est nette jusqu'a environ `z~2.5-3`. Au-dela, les objets sont beaucoup plus rares; les conclusions sur les hauts redshifts doivent donc rester prudentes.

![Marie+GCNN z_true vs z_pred](../../experiments/redshift_run_001/results_report_marie_gcnn/heatmap_ztrue_zpred.png)

### Residus en fonction de `z_pred`

Les residus restent globalement centres autour de zero. On observe toutefois des regimes plus bruites a tres bas `z_pred` et dans les zones ou les effectifs deviennent faibles.

![Marie+GCNN residus vs z_pred](../../experiments/redshift_run_001/results_report_marie_gcnn/heatmap_residual_zpred.png)

### Residus en fonction de la bande I

La majorite de la densite se trouve vers `I~23-25`, ce qui correspond au coeur du dataset. Les objets brillants `I in [18,20]` sont beaucoup moins nombreux et doivent etre interpretes comme une analyse specifique plutot qu'une estimation tres stable.

![Marie+GCNN residus vs mag I](../../experiments/redshift_run_001/results_report_marie_gcnn/heatmap_residual_mag_i.png)

Les memes heatmaps ont ete generees pour CNN bins, Marie baseline et les versions fold0 dans les dossiers `results_report_*`.

## 6. Moyennage des performances par `z_pred`

Les predictions ont ete regroupees par intervalles de `z_pred`, puis les metriques ont ete calculees dans chaque bin. Pour Marie+GCNN, la dispersion reste majoritairement dans la plage `sigma_NMAD ~0.03-0.06` sur les bins bien peuples. Les bins de haut redshift sont plus instables car certains contiennent moins de 30 objets.

![Marie+GCNN sigma_NMAD par z_pred](../../experiments/redshift_run_001/results_report_marie_gcnn/sigma_by_z_pred.png)

Points d'attention:

- les bins `z_pred > 3.5` sont souvent peu peuples;
- les variations locales de `sigma_NMAD` dans la queue haut redshift ne doivent pas etre sur-interpretees;
- fold0 fournit une validation spatiale plus large avec `32405` objets test, utile pour consolider les observations.

## 7. Moyennage des performances par bande I

La demande specifique du protocole etait de selectionner `I in [18,20]`, de decouper cette plage en `20` bins et de calculer une metrique robuste par bin. Cette analyse a ete realisee pour tous les modeles.

Pour Marie+GCNN, les bins contiennent entre `4` et `39` objets sur le split canonique. La courbe est donc informative mais bruitee. Sur fold0, les bins contiennent entre `8` et `73` objets, ce qui donne une estimation legerement plus stable.

![Marie+GCNN sigma_NMAD par bande I](../../experiments/redshift_run_001/results_report_marie_gcnn/sigma_by_mag_i.png)

Interpretation:

- la plage `I in [18,20]` correspond aux objets brillants, minoritaires dans ce dataset;
- les fluctuations fortes d'un bin a l'autre sont principalement dues aux petits effectifs;
- il est recommande de presenter cette figure comme une analyse ciblee demandee par le protocole, en complement d'une analyse plus globale sur toute la plage `I in [18,25]` si necessaire.

## 8. Analyse spatiale RA/DEC

Les cartes RA/DEC point par point colorent chaque objet par son erreur normalisee absolue. Elles permettent de verifier visuellement l'absence de zone catastrophique evidente dans le champ COSMOS.

![Marie+GCNN carte RA/DEC](../../experiments/redshift_run_001/results_report_marie_gcnn/radec_error_map.png)

Limite importante: la figure actuelle est une carte objet par objet. Elle repond partiellement a la demande “resultats par RA/DEC”, mais pas encore a une aggregation spatiale par cellules. Pour verrouiller completement ce point, il faut ajouter une heatmap RA/DEC binned, avec par cellule:

- nombre d'objets;
- erreur mediane `median(|dz|)`;
- ou `sigma_NMAD` locale.

Cette amelioration est recommandee pour la version finale du memoire.

## 9. Artefacts disponibles par modele

| Modele | Rapport | Fichiers principaux |
|---|---|---|
| CNN bins | `results_report_cnn_bins/` | heatmaps, `metrics_global.csv`, `metrics_by_z_pred.csv`, `metrics_by_mag_i.csv`, RA/DEC |
| CNN bins fold0 | `results_report_cnn_bins_fold0/` | memes artefacts sur fold0 |
| Marie baseline | `results_report_marie_baseline/` | heatmaps, agregations, RA/DEC |
| Marie baseline fold0 | `results_report_marie_baseline_fold0/` | memes artefacts sur fold0 |
| Marie+GCNN | `results_report_marie_gcnn/` | heatmaps, agregations, RA/DEC |
| Marie+GCNN fold0 | `results_report_marie_gcnn_fold0/` | memes artefacts sur fold0 |

Le dossier `results_report/` correspond au rapport Marie+GCNN de reference produit avant separation explicite par nom de modele.

## 10. Stripe82

La zone Stripe82 standard n'est pas couverte par les artefacts COSMOS actuels. Le masque standard selectionne `0` objet dans `dataset_metadata.npz`.

Constat:

- COSMOS actuel: `RA in [149.43, 150.79]`, `DEC in [1.58, 2.88]`;
- Stripe82 standard: bande equatoriale proche de `|DEC| <= 1.25`, avec RA autour de la zone Stripe82 SDSS, notamment la definition wrap-around `RA > 300` ou `RA < 60`;
- intersection actuelle: `0` objet.

Il ne faut donc pas presenter les resultats COSMOS comme des resultats Stripe82.

Sources utiles:

- [SDSS Legacy Stripe 82](https://classic.sdss.org/legacy/stripe82.php)
- [SDSS Stripe 82 Images Tutorial](https://www.sdss4.org/dr15/tutorials/get_stripe82_images/)

Plan pour ajouter une vraie experience Stripe82:

1. Recuperer un dataset Stripe82 compatible avec l'apprentissage image: images/cutouts, photometrie, redshifts de reference et flags.
2. Construire un convertisseur vers le format `.npz` attendu par le pipeline (`cube`, `info`, `flag`).
3. Adapter explicitement les bandes SDSS `ugriz` au schema actuel `u/g/r/i/z/y`, car Stripe82 SDSS ne fournit pas naturellement la bande `y`.
4. Definir le traitement morphologique: catalogue morphologique compatible ou descripteurs morphologiques alternatifs.
5. Relancer au minimum CNN bins, Marie baseline et Marie+GCNN sur cette base.
6. Generer `results_report_stripe82/` avec les memes metriques et figures.

## 11. Limites et points a consolider

| Point | Statut actuel | Action recommandee |
|---|---|---|
| Dataset COSMOS | OK | Conserver le protocole et les chiffres de cross-match |
| Heatmaps de dispersion | OK | Utiliser Marie+GCNN comme figure principale, comparer aux autres modeles en annexe |
| Agregation par `z_pred` | OK | Signaler les bins haut redshift a faible effectif |
| Agregation par bande I | OK | Signaler les faibles effectifs dans `I in [18,20]` |
| RA/DEC | Partiel | Ajouter une carte RA/DEC binned par cellule spatiale |
| Fold0 | OK | Presenter comme validation spatiale a test plus large |
| Stripe82 | Non disponible | Recuperer un dataset Stripe82 reel avant revendication experimentale |

## 12. Conclusion pour le memoire

Les analyses realisees repondent a la majorite des demandes: repartition train/test, distributions en redshift, flags, bandes photometriques, heatmaps de dispersion, metriques agregees par `z_pred`, metriques agregees par bande I, comparaison multi-architectures et validation sur fold spatial.

Le modele Marie baseline presente la meilleure dispersion robuste, tandis que Marie+GCNN reduit les outliers et donne le meilleur RMSE sur le split canonique. Cette complementarite est importante: elle suggere que les premieres couches equivariantes stabilisent les predictions extremes, meme si elles ne minimisent pas toujours la dispersion mediane.

Les deux elements a traiter avant une version finale totalement conforme sont la carte RA/DEC agregee et l'experience Stripe82 sur un vrai dataset Stripe82. Les artefacts actuels COSMOS ne permettent pas de revendiquer un resultat Stripe82.

