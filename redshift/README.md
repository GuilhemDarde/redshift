# Redshift CFM Predictions

Pipeline de recherche pour l'estimation de redshift photometrique sur COSMOS avec:

- generateur Conditional Flow Matching conditionne par redshift, magnitude, couleurs et morphologie;
- backbone G-CNN equivariant avec tete Mixture Density Network;
- experiences Sim2Real, deep ensembles et diagnostics d'incertitude.

## Installation

Le projet est un ensemble de scripts Python. Les chemins de donnees ne sont pas inclus dans le depot.

```bash
export REDSHIFT_ENV=/chemin/avec/espace/venvs/redshift
export PIP_CACHE_DIR=/chemin/avec/espace/pip-cache
export TMPDIR=/chemin/avec/espace/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$REDSHIFT_ENV")"

python3.8 -m venv "$REDSHIFT_ENV"
source "$REDSHIFT_ENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

Ne pas installer dans un environnement partage non inscriptible du type `/home/grp1/.../lib/python3.8/site-packages`. Si `pip` renvoie `Permission denied`, recreer un venv dans un espace ou vous avez les droits. Si `pip` renvoie `No space left on device`, placer `REDSHIFT_ENV`, `PIP_CACHE_DIR` et `TMPDIR` sur un disque projet/scratch avec plusieurs Go libres: les wheels CUDA de PyTorch sont volumineux.

Les dependances sont separees:

- `redshift/requirements-core.txt`: analyse, figures, FITS, CSV, sans PyTorch.
- `redshift/requirements-torch.txt`: pile complete pour entrainements Torch/G-CNN.
- `requirements.txt`: installation complete par defaut.

Si le serveur fournit deja PyTorch via un module ou un environnement central, activer ce module puis installer seulement les dependances manquantes:

```bash
python -m pip install --no-cache-dir -r redshift/requirements-core.txt
python -m pip install --no-cache-dir escnn
```

Pour une execution reproductible, declarer explicitement les chemins locaux:

```bash
export COSMOS_DATA_PATH=/chemin/vers/les/fichiers_npz_cosmos
export COSMOS_MORPHO_PATH=/chemin/vers/catalogue_morpho_cosmos.fits
export COSMOS_EXP_FOLDER=/chemin/vers/le/dossier_experiences
export COSMOS_METADATA_PATH=/chemin/optionnel/vers/dataset_metadata.npz
export COSMOS_PROCESSED_DATASET_PATH=/chemin/optionnel/vers/processed_cosmos_all.npz
export COSMOS_SEED=42
export COSMOS_NUM_WORKERS=2
export COSMOS_SYNTH_NUM_WORKERS=4
```

Les chemins, seeds, batch sizes, workers et noms d'artefacts canoniques sont centralises dans `config.py`. Les scripts peuvent encore surcharger certains parametres en CLI, mais leurs valeurs par defaut viennent de `CONFIG`.

Pour eviter de refaire le cross-match FITS/NPZ a chaque experience, definir `COSMOS_PROCESSED_DATASET_PATH` vers un fichier `.npz` sur un disque avec assez d'espace. Au premier chargement complet, le dataset pretraite est sauvegarde; les scripts suivants le rechargent directement. Ne pas utiliser ce cache avec `--max_files`, et utiliser un cache distinct par region (`all`, `stripe82`) si besoin.

## Structure

- `config.py`: configuration globale, chemins, bornes de selection, constantes et hyperparametres par defaut.
- `data_loader.py`: chargement COSMOS, filtrage, cross-match morphologique et split spatial train/val/test.
- `analysis_utils.py`: metriques, binning, masque Stripe82, exports metadata et helpers CSV.
- `model.py`: CFM conditionnel, U-Net leger et perte photometrique.
- `backbone.py`: G-CNN equivariant et tete MDN pour l'estimation probabiliste du redshift.
- `train.py`: entrainement du generateur CFM.
- `generate_mass.py`: generation de jeux synthetiques depuis le CFM entraine.
- `experiment_backbone.py`: pre-entrainement synthetique, fine-tuning reel et rapport statistique.
- `experiment_sota.py`: ensemble G-CNN/MDN avec evaluation NMAD/outliers.
- `experiment_uncertainty.py`: analyse PIT de la calibration.
- `analyze_dataset.py`: audit dataset, split, distributions, flags, bandes U/I/Z.
- `analyze_results.py`: heatmaps, agrégations par `z_pred`, bande I et cartes RA/DEC.
- `experiment_cnn_bins.py`: CNN de classification par bins de redshift.
- `experiment_marie_baseline.py`: baseline locale style Marie, fichiers originaux intacts.
- `experiment_marie_gcnn.py`: premières couches G-CNN + tête style Marie.
- `utils.py`: visualisations et metriques communes.

## Donnees Attendues

`COSMOS_DATA_PATH` doit contenir des fichiers `.npz` avec au minimum:

- `cube`: images multi-bandes, format attendu `[N, H, W, C]`;
- `info`: table structuree contenant magnitude `i`, redshift spectroscopique, RA et DEC;
- `flag` optionnel: masque qualite par objet et canal.

Les sorties lourdes ne doivent pas etre versionnees: checkpoints `.pt/.pth`, datasets `.npz`, logs `.log`, figures et dossiers d'experiences sont ignores par Git.

## Protocole Scientifique Canonique

Utiliser ce protocole pour produire un resultat comparable et archivable.

1. Preparer les donnees
   - `COSMOS_DATA_PATH`: fichiers `.npz` COSMOS.
   - `COSMOS_MORPHO_PATH`: catalogue FITS avec RA/DEC, rayon effectif et indice de Sersic.
   - `COSMOS_EXP_FOLDER`: dossier unique du run, hors depot Git.
   - Fixer `COSMOS_SEED`, `COSMOS_NUM_WORKERS` et `COSMOS_SYNTH_NUM_WORKERS`.

2. Construire le split
   - Le split est spatial et deterministe par RA dans `data_loader.py`.
   - Les proportions canoniques sont 80% train, 10% validation, 10% test.
   - Les folds optionnels sont des blocs RA deterministes via `--n_folds` et `--fold_id`.
   - Les filtres canoniques sont `I_MIN <= i <= I_MAX`, `0.001 < z <= Z_MAX`, flags nuls sur les canaux selectionnes, puis cross-match morphologique a 1 arcsec.
   - Reporter dans le journal du run le nombre d'objets train/val/test apres filtrage et cross-match.

3. Entrainer le generateur
   - Commande canonique:

```bash
python redshift/train.py --epochs 100 --batch_size 64 --lr 1e-4 --lambda_photo 0.01 --seed "$COSMOS_SEED"
```

   - Checkpoint attendu: `cfm_model_physics.pt` dans `COSMOS_EXP_FOLDER`.

4. Generer le jeu synthetique
   - Commande canonique:

```bash
python redshift/generate_mass.py --n 100000 --batch_size 256 --seed "$COSMOS_SEED"
```

   - Dataset attendu: `synthetic_cosmos_100k_v3.npz` ou chemin passe avec `--output`.

5. Evaluer Sim2Real
   - Commande canonique:

```bash
python redshift/experiment_sota.py --epochs_syn 45 --ft_epochs 15 --n_models 5 --num_gaussians 5 --batch_size 128 --seed "$COSMOS_SEED"
```

   - Resultat attendu: `results_sota_ensemble.npz` dans `COSMOS_EXP_FOLDER`.
   - Metriques canoniques: biais moyen de `dz = (z_pred - z_true)/(1+z_true)`, `sigma_NMAD`, fraction d'outliers `|dz| > 0.15`.

6. Controler la calibration et la physique
   - PIT:

```bash
python redshift/experiment_uncertainty.py --epochs 15 --num_gaussians 3 --batch_size 128 --seed "$COSMOS_SEED"
```

   - Couleurs:

```bash
python redshift/validation_colors.py
```

7. Archiver les resultats
   - Commit Git exact ou archive du code.
   - Variables d'environnement et commande CLI complete.
   - Versions Python, PyTorch, CUDA et dependances.
   - Nombre d'objets apres filtrage/cross-match et tailles train/val/test.
   - Checkpoints, fichiers `.npz`, figures et logs dans `COSMOS_EXP_FOLDER`.

## Commandes

Les commandes suivantes sont donnees depuis la racine du depot.

Audit dataset:

```bash
python redshift/analyze_dataset.py --region all
python redshift/analyze_dataset.py --region stripe82 --max_files 2
```

Entrainer le generateur CFM:

```bash
python redshift/train.py --epochs 100 --batch_size 64 --lr 1e-4 --lambda_photo 0.01 --seed "$COSMOS_SEED"
```

Generer un dataset synthetique:

```bash
python redshift/generate_mass.py --n 100000 --batch_size 256 --seed "$COSMOS_SEED"
```

Lancer l'experience ensemble G-CNN/MDN:

```bash
python redshift/experiment_sota.py --epochs_syn 45 --ft_epochs 15 --n_models 5 --num_gaussians 5 --batch_size 128 --seed "$COSMOS_SEED" --region all
python redshift/experiment_sota.py --epochs_syn 1 --ft_epochs 1 --n_models 1 --limit_batches 2 --fold_id 0
```

Analyser les resultats:

```bash
python redshift/analyze_results.py --results "$COSMOS_EXP_FOLDER/results_sota_ensemble.npz"
```

Nouvelles experiences demandees:

```bash
python redshift/experiment_cnn_bins.py --epochs 10 --region all
python redshift/experiment_marie_baseline.py --epochs 10 --region all
python redshift/experiment_marie_gcnn.py --epochs 10 --region all
python redshift/experiment_cnn_bins.py --epochs 1 --region stripe82 --limit_batches 2
```

Analyser la calibration PIT:

```bash
python redshift/experiment_uncertainty.py --epochs 15 --num_gaussians 3 --batch_size 128 --seed "$COSMOS_SEED"
```

Lancer les smoke tests sans donnees lourdes:

```bash
python -m unittest discover -s redshift/tests -v
```

## Reproductibilite

Avant de comparer des resultats, noter dans le dossier d'experience:

- commit Git ou archive exacte du code;
- valeurs de `COSMOS_DATA_PATH` et `COSMOS_EXP_FOLDER`;
- nombre d'objets apres filtrage et cross-match;
- hyperparametres CLI utilises;
- versions Python, PyTorch, CUDA et dependances.

Les scripts actuels restent des scripts de recherche. Les sorties de reference doivent etre archivees dans `COSMOS_EXP_FOLDER`, pas dans le depot.
