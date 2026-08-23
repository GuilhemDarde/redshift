# Redshift Photo-z: generative augmentation on COSMOS Ultra Deep

Research code for photometric redshift estimation on COSMOS Ultra Deep, investigating
whether conditional flow-matching image-to-image augmentation can improve a
Pasquet/Treyer-style baseline on difficult photometric regions.

**Status (August 2026):** experimental work complete. The M2 report
(`ALTERNANCE_M2_2025_2026_DARDE.pdf`, 77 pages) has been submitted; oral defence
preparation is under way. See `reports/HANDOFF_MEMOIRE.md` for the writing-side
handoff and `reports/soutenance_M2_plan_slides.md` for the defence plan.

---

## Main findings

The project ran in two missions. The first synthesised galaxy images from noise and
transferred the predictive model to real data (Sim2Real); it exposed a large
simulation-to-observation gap. The second transformed **real** images instead, which
is what this repository now mostly contains.

Four results survived statistical control:

```text
1. Error is governed by LOCAL AMBIGUITY, not by training-set COVERAGE.
   Spearman(error, kNN support radius) = -0.023 to +0.067   -> null
   Spearman(error, local label dispersion) = +0.241 to +0.278
   Robust across three feature spaces; magnitude confounder ruled out.

2. The effect of augmentation DEPENDS ON DOSE AND CHANGES SIGN.
   ~250 synthetic images (0.26 % of train): sigma_NMAD improves, t = +5.09 global
   12 766 images (full pool):                sigma_NMAD degrades, t = -4.40

3. The low-dose gain REACHES the targeted region.
   low_photometric_support (n = 6419): t = +4.25

4. NOISE SMOOTHING IS NOT THE CAUSE of the high-dose degradation.
   The generator removes up to 44 % of background noise amplitude in g.
   Restoring it (variance to 0.15 %, lag-1 autocorrelation to 0.001) changed
   nothing: 9 comparisons out of 9 still degrade. Hypothesis refuted by
   intervention, not by observation.
```

What is **not** established: the behaviour on the hardest target
(`faint_and_low_photometric_support`, n = 2780). The point estimate moves in the
favourable direction (−1.06 %) but the detection threshold there is 3.6 % relative.
The experiment cannot conclude either way — a retrospective power analysis shows the
original design was underpowered before the first image was generated.

Two candidate mechanisms remain untested for the high-dose degradation: spatial flux
redistribution (residual L1 ≈ 0.20 in band `g`), and distribution reweighting induced
by the targeting itself (selected sources are over-represented by a factor ≈ 2.1 at
full dose). The decisive experiment requires no generation: duplicate the **real**
source images at the same doses.

### A note on terminology

The generative model implements **conditional flow matching with the optimal-transport
conditional path** of Lipman et al. (2023). It does **not** implement the OT-CFM of
Tong et al. (2024), which couples noise and data by minibatch optimal transport:
`model.py` draws `x_0` independently for each image. Earlier documents in this
repository call it "OT-CFM"; read that as referring to the conditional path.

---

## Key numbers

| Quantity | Value |
|---|---|
| Dataset | COSMOS Ultra Deep, 159 015 objects with spectroscopic redshift |
| Bands | 6: `u, g, r, i, z, y` — `u` from CLAUDS/CFHT MegaCam, `grizy` from Subaru HSC |
| Strict split, fold 0 | 95 409 / 31 803 / 31 803, integrity audit all zero |
| Baseline, 5 seeds | `sigma_NMAD = 0.015168 ± 0.000050`, RMSE `0.242668`, outliers `3.137 %` |
| Detection threshold (2σ), global | `sigma_NMAD 0.000099`, outliers `0.093` |
| Detection threshold (2σ), hard target | `sigma_NMAD 0.000934`, outliers `0.795` |
| Best augmentation dose | 250 images, `sigma_NMAD = 0.015026 ± 0.000029` |

`sigma_NMAD`, bias and outlier rate are normalised by `(1+z)`. **RMSE is not** — it can
therefore move opposite to the others between experiments.

---

## Repository layout

Core dataset, metrics and protocol:

- `config.py` — shared configuration and default paths.
- `data_loader.py` — COSMOS loading, filtering, metadata, split construction.
- `analysis_utils.py` — metrics, split helpers, metadata export, integrity audit.
- `density_utils.py` — kNN support radius and local label dispersion.

Predictive baseline:

- `marie_treyer_exact.py` — inception-style multiband model, 360 redshift bins.
- `experiment_marie_exact_augmented.py` — real-only and augmented training/evaluation.
- `analyze_treyer_figure7.py` — Figure 7-style diagnostics.
- `analyze_marie_cv_folds.py` — analysis of provided CV folds.

Generative model and augmentation:

- `model.py` — conditional flow matching, partial inversion, i2i generation.
- `cfm_conditioning.py` — conditioning schemas (`legacy7`, `marie_mags`).
- `train.py` — generator training.
- `generate_cfm_i2i.py` — targeted image-to-image generation.
- `photometric_validation.py` — photometric filtering of candidates.
- `visual_band_inspection.py` — multiband visual and flux diagnostics.
- `renormalize_i2i_flux.py` — per-band source-flux recalibration.
- `blend_i2i_with_source.py` — residual blending with the source image.
- `restore_augmentation_noise.py` — calibrated per-band noise reinjection.

Diagnostics developed for this work:

- `analyze_photometric_support.py` — kNN-based low-support analysis.
- `diagnose_performance_ceiling.py` — support, local ambiguity and model-error surfaces.
- `aggregate_seed_variance.py` — inter-seed variance and detection thresholds.
- `plot_dose_response.py` — dose-response curves, with series overlay.
- `analyze_fidelity_by_regime.py` — reconstruction fidelity by redshift and magnitude.
- `analyze_background_noise.py` — robust background sigma and autocorrelation.
- `make_memoire_figures.py` — generates and collects all report figures.
- `plot_i2i_stage_comparison.py` — source / raw i2i / final blended comparison.

Scripts and tests:

- `run_h1_replication.sh` — resumable replication of the H1/H2 seed campaign.
- `tests/` — unit and smoke tests for splits, conditioning, support, filtering,
  visual diagnostics and the performance-ceiling tooling.

---

## Documentation

Working documents live in `reports/` and are written in French.

| File | Content |
|---|---|
| `HANDOFF_MEMOIRE.md` | **Start here.** Report state, frozen numbers, wording rules, corrections. |
| `synthese_projet_photoz_otcfm.md` | Full scientific synthesis, every figure and table. |
| `soutenance_M2_plan_slides.md` | Oral defence plan, 19 slides, anticipated questions. |
| `ch00`–`ch09` | Per-chapter drafts of the report. |
| `ch06_reecrit.tex`, `annexes.tex` | LaTeX, narrative rewrite of chapter 6 and appendices. |
| `references_memoire.bib` | 39 verified references, Overleaf-ready. |
| `audit_bibliographie.md` | Bibliographic audit: 8 corrections, 1 non-existent entry removed. |
| `audit_citations_propos.md` | Claim/citation audit: 5 misplacements, 13 unsourced claims. |
| `plan_experiences_cloture_memoire.md` | Hypotheses H1–H6 with pre-registered interpretations. |
| `checklist_memoire_qualite.md` | What a high-quality report on this topic should contain. |
| `figures_memoire/` | 20 figures, named after the report numbering. |

> `PROJECT_CONTEXT_NEW_CHAT.md` at the repository root is **stale**. Several of its
> conclusions were later disproven; it carries a warning header and is kept only for
> the historical record of server paths and early experiments.

---

## Installation

```bash
python3.8 -m venv /path/to/venvs/redshift
source /path/to/venvs/redshift/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

If PyTorch is provided by the environment, install only the non-torch stack:

```bash
python -m pip install --no-cache-dir -r redshift/requirements-core.txt
python -m pip install --no-cache-dir escnn
```

## Environment

Large data, checkpoints, generated images and experiment folders are deliberately not
versioned. Put the exports in a file and source it once per session — the compute node
has no outbound network access, so code is transferred with `scp`.

```bash
export COSMOS_DATA_PATH=/home/barrage/SPECT_COSMOS_US/us
export COSMOS_MORPHO_PATH=/home/data/hugo/catalogs/catalogue_morpho_cosmos.fits
export COSMOS_EXP_FOLDER=/home/data/hugo/experiments/marie_exact_cfm_dafusion_fold0
export COSMOS_PROCESSED_DATASET_PATH=/home/data/hugo/experiments/marie_lowmag_overnight_fold0_20260512_113005/processed_cosmos_ud_spec_with_source_file.npz
export COSMOS_NUM_WORKERS=0
export EXP="$COSMOS_EXP_FOLDER"
```

## Protocol rules

```text
- never tune on test;
- estimate every threshold on train only;
- add synthetic images to train only;
- evaluate final metrics on real test images;
- audit train/val/test overlaps on every run.
```

The strict split does **not** depend on the `--seed` argument: it is computed with a
fixed internal seed. Varying `--seed` therefore isolates training variance at identical
data, which is what makes the inter-seed variance measurement meaningful.

---

## Reproducing the main results

Run the tests first:

```bash
python -m unittest discover -s redshift/tests -v
```

**Baseline and seed variance**

```bash
python redshift/experiment_marie_exact_augmented.py --ablations real \
  --split_strategy marie_strict --fold_id 0 --n_folds 5 \
  --field cosmos_ud --sample_filter spec --epochs 50 --seed 42 \
  --cache_path "$COSMOS_PROCESSED_DATASET_PATH" \
  --output_dir "$EXP/marie_exact_strict_fold0_real"

python redshift/aggregate_seed_variance.py \
  --run_dirs "$EXP/marie_exact_strict_fold0_real" "$EXP/seed_variance/real_strict_fold0_seed"{43,44,45,46} \
  --ablation real --output_dir "$EXP/seed_variance"
```

**Performance ceiling — the central diagnostic**

```bash
for FS in classic_colors marie_magnitudes marie_magnitudes_colors; do
  python redshift/diagnose_performance_ceiling.py \
    --predictions "$EXP/marie_exact_strict_fold0_real/predictions_marie_exact_real.npz" \
    --metadata "$EXP/marie_exact_strict_fold0_real/dataset_metadata_marie_exact.npz" \
    --output_dir "$EXP/marie_exact_strict_fold0_real/performance_ceiling_$FS" \
    --feature_space "$FS" --k 10 --low_fraction 0.20
done
```

**Augmentation pipeline** — generation, photometric filtering, blending, visual
inspection, then optional noise restoration:

```bash
python redshift/generate_cfm_i2i.py --mode i2i \
  --checkpoint "$EXP/cfm_fold0_marie_strict.pt" \
  --output "$EXP/cfm_i2i_lowsupport_lowambig_fold0.npz" \
  --split_strategy marie_strict --fold_id 0 --n_folds 5 \
  --field cosmos_ud --sample_filter spec --cache_path "$COSMOS_PROCESSED_DATASET_PATH" \
  --selection_target low_support_low_ambiguity \
  --low_photometric_support_fraction 0.20 --low_ambiguity_quantile 0.50 \
  --photometric_support_k 10 --n_aug_per_source 2 --t0 0.25 --noise_scale 0.02 --steps 50
```

Pass `--max_metric_samples` to `visual_band_inspection.py` large enough to cover the
whole candidate pool: its default of 8000 silently discards the rest.

**Dose-response and mechanism test**

```bash
bash redshift/run_h1_replication.sh --dry-run

python redshift/plot_dose_response.py \
  --run_dirs "$EXP"/dose_response/n{1000,2500,5000,10000,12766} \
  --compare_run_dirs "$EXP"/dose_response_noiserestored/n{1000,2500,5000,10000,12766} \
  --labels "sans correction,bruit restaure" \
  --output_dir "$EXP/dose_response/comparison" --ablation i2i \
  --baseline_variance "$EXP/seed_variance/seed_variance_real.csv" --n_available 12766
```

**Figures**

```bash
python redshift/make_memoire_figures.py --exp_root "$EXP" --output_dir "$EXP/figures_memoire"
```

---

## How to state the results

```text
DO NOT SAY : "the augmentation does not help the hard target"
SAY        : "the experiment cannot conclude on the hard target"

DO NOT SAY : "CFM is the current state of the art"
SAY        : which properties motivated the choice

DO NOT SAY : "the model beats the problem's ceiling"
             (the 0.28-0.37 error/ambiguity ratio is an artefact of measuring
              ambiguity in a space poorer than the model's actual input)

ALWAYS     : recall that RMSE is not normalised by (1+z) when comparing it
```

The defensible summary:

```text
The effect of generative augmentation on photo-z depends on dose and changes sign.
Error is governed by local ambiguity rather than training-set coverage, which no
amount of added data can reduce. The contribution is a quantitative characterisation
of when such augmentation helps and when it hurts, together with the statistical
instrumentation needed to establish it.
```
