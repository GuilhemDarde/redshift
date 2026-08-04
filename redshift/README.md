# Redshift Photo-z: Marie/Treyer Baseline and OT-CFM i2i Augmentation

This repository contains research code for photometric redshift estimation on COSMOS Ultra Deep, with a focus on evaluating whether OT-CFM image-to-image augmentation can improve a Marie/Treyer-style baseline on difficult photometric regions.

## Current Scientific Scope

The current project should be understood in this order:

1. Reproduce a credible Marie/Treyer-style photo-z baseline.
2. Use a strict train/validation/test protocol with no test leakage.
3. Define difficult galaxies by low photometric support, not by sky position.
4. Generate targeted OT-CFM image-to-image augmentations.
5. Filter, recalibrate, or blend augmentations to preserve labels.
6. Evaluate on real test images only.

Main conclusion so far:

```text
OT-CFM i2i augmentation is useful as a diagnostic and weak regularizer, but it does not yet robustly improve the Marie/Treyer baseline on faint, low-photometric-support galaxies.
```

## Important Files

Core dataset and metrics:

- `config.py`: shared configuration and default paths.
- `data_loader.py`: COSMOS loading, filtering, metadata and split construction.
- `analysis_utils.py`: metrics, split helpers, metadata exports and audit utilities.
- `analyze_photometric_support.py`: kNN-based low photometric support analysis.
- `diagnose_performance_ceiling.py`: train-support, local-redshift-ambiguity and model-error surfaces.

Marie/Treyer baseline:

- `marie_treyer_exact.py`: Marie/Treyer-style model.
- `experiment_marie_exact_augmented.py`: real-only and augmented training/evaluation.
- `analyze_marie_cv_folds.py`: analysis of provided Marie/Treyer CV folds.
- `analyze_treyer_figure7.py`: Figure 7-style diagnostics.

OT-CFM and i2i augmentation:

- `model.py`: conditional flow model.
- `cfm_conditioning.py`: conditioning strategies.
- `train.py`: CFM training.
- `generate_cfm_i2i.py`: image-to-image generation.
- `photometric_validation.py`: photometric filtering of candidates.
- `visual_band_inspection.py`: multi-band visual/flux diagnostics.
- `renormalize_i2i_flux.py`: per-band source-flux recalibration.
- `blend_i2i_with_source.py`: residual blending with the source image.
- `plot_i2i_stage_comparison.py`: source / raw i2i / final fused image comparison.

Tests:

- `tests/`: unit/smoke tests for split logic, conditioning, density support, filtering and visual diagnostics.

## Installation

From repository root:

```bash
python3.8 -m venv /path/to/venvs/redshift
source /path/to/venvs/redshift/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

The root `requirements.txt` delegates to `redshift/requirements.txt`.

If PyTorch is already provided by the server environment, install only the non-torch stack and the missing ML packages as needed:

```bash
python -m pip install --no-cache-dir -r redshift/requirements-core.txt
python -m pip install --no-cache-dir escnn
```

## Data and Environment

Large data, checkpoints, generated images and experiment folders are intentionally not versioned.

Typical server paths:

```bash
export COSMOS_DATA_PATH=/home/barrage/SPECT_COSMOS_US/us
export COSMOS_MORPHO_PATH=/home/data/hugo/catalogs/catalogue_morpho_cosmos.fits
export COSMOS_EXP_FOLDER=/home/data/hugo/experiments/marie_exact_cfm_dafusion_fold0
export COSMOS_PROCESSED_DATASET_PATH=/home/data/hugo/experiments/marie_lowmag_overnight_fold0_20260512_113005/processed_cosmos_ud_spec_with_source_file.npz
export COSMOS_SEED=42
export COSMOS_NUM_WORKERS=0
```

Keep heavy files outside the git repository:

- FITS/NPZ datasets;
- Marie fold tar archives;
- `.pt/.pth` checkpoints;
- generated i2i candidates;
- full experiment folders.

## Strict Protocol

Final i2i comparisons use `marie_strict`, fold 0:

```text
train = 95409
validation = 31803
test = 31803
```

Rules:

- never tune on test;
- define low-support thresholds from train only;
- add synthetic images to train only;
- evaluate final metrics on real test images only;
- audit overlaps between train, validation and test.

## Difficult Subset

Low photometric support is defined by kNN radius in:

```text
[mag_i, g-r, r-i, i-z]
```

Main target:

```text
faint_and_low_photometric_support = mag_i >= 23.5 AND low photometric support
```

The strict real-only baseline on this target is much harder than global evaluation:

```text
global sigma_NMAD = 0.01516, RMSE = 0.24447, outliers = 3.10%
faint + low support sigma_NMAD = 0.02237, RMSE = 0.40062, outliers = 5.43%
```

## Key i2i Variants

Raw i2i:

```text
source image -> partial CFM perturbation -> raw generated image
```

Per-band recalibration:

```text
image_i2i_recalibrated = image_i2i * flux_source / flux_i2i
```

Residual blending:

```text
image_final = image_source + alpha * (image_i2i_raw - image_source)
```

Best visual compromise so far:

```text
alpha = 0.75
```

Important interpretation:

```text
The CFM output is a full image, not a noise or delta map. Difference maps are diagnostics only.
```

## Reports and Meeting Assets

Current local context and report files are in the repository root and `reports/`:

- `PROJECT_CONTEXT_NEW_CHAT.md`: compact context for continuing the memoir discussion.
- `reports/suivi_experience_i2i_strict_low_photo.md`: chronological experiment tracking.
- `reports/rapport_reunion_i2i_strict_faible_support_20260518.md`: meeting/report synthesis.
- `reports/rapport_reunion_i2i_strict_faible_support_20260518.pdf`: rendered report.
- `reports/script_oral_reunion_i2i_strict_faible_support_20260518.md`: oral script.
- `reports/meeting_assets_20260518/`: curated figures and CSV diagnostics kept for the memoir.

Older exploratory reports and full generated artifacts have been removed from the working tree.

## Useful Commands

Run tests:

```bash
python -m unittest discover -s redshift/tests -v
```

Analyze strict real baseline support:

```bash
python redshift/analyze_photometric_support.py \
  --predictions "$EXP/marie_exact_strict_fold0_real/predictions_marie_exact_real.npz" \
  --metadata "$EXP/marie_exact_strict_fold0_real/dataset_metadata_marie_exact.npz" \
  --output_dir "$EXP/marie_exact_strict_fold0_real/photometric_support_real" \
  --k 10 \
  --low_fraction 0.20
```

Diagnose whether the model is limited by data support, local redshift ambiguity or model error:

```bash
python redshift/diagnose_performance_ceiling.py \
  --predictions "$EXP/marie_exact_strict_fold0_real/predictions_marie_exact_real.npz" \
  --metadata "$EXP/marie_exact_strict_fold0_real/dataset_metadata_marie_exact.npz" \
  --output_dir "$EXP/marie_exact_strict_fold0_real/performance_ceiling" \
  --feature_space classic_colors \
  --k 10 \
  --low_fraction 0.20
```

Plot source / raw i2i / final fused images:

```bash
python redshift/plot_i2i_stage_comparison.py \
  --final_augmentations "$EXP/cfm_i2i_faint_low_photo_strict_fold0_legacy_blend_a0p75_visualfiltered.npz" \
  --output_dir "$EXP/stage_i2i_legacy_blend_a0p75" \
  --split_strategy marie_strict \
  --fold_id 0 \
  --n_folds 5 \
  --field cosmos_ud \
  --sample_filter spec \
  --max_examples 8 \
  --selection mixed
```

## Memoir Positioning

Do not claim:

```text
The CFM significantly improves Marie/Treyer.
```

Defensible claim:

```text
Under a strict protocol, label-preserving i2i augmentation for photo-z is highly constrained. The main bottleneck is preserving multi-band physical information: fluxes, colors, local spatial structure, background and neighbors. Current OT-CFM augmentations provide useful diagnostics but no robust improvement on the faint, low-support target subset.
```
