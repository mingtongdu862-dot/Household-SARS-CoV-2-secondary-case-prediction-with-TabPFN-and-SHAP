# Household SARS-CoV-2 secondary-case prediction with TabPFN and SHAP

Code for predicting, from Swedish national register data, whether a household
with a confirmed SARS-CoV-2 infection in 2020 will have a temporally compatible
secondary case, using a bagged TabPFN-2.5 ensemble compared against five
tabular baselines, and explaining the ensemble with Kernel SHAP.

> **Data availability.** The analysis uses pseudonymised individual-level
> register data from the SWECOV project (Public Health Agency of Sweden,
> Statistics Sweden, National Board of Health and Welfare). These data can
> only be accessed inside a secure analysis environment under ethical
> approval and cannot be shared. This repository therefore contains code and
> documentation only; no data, derived data, predictions, SHAP matrices or
> fitted models.

## Repository structure

| Path | Purpose |
|---|---|
| `data_preprocessing.py` | Time filtering, feature selection, population index, index/secondary/uninfected labelling, death-date filter |
| `feature_extraction.py` | Individual-level features anchored to the household anchor date |
| `feature_aggregation.py` | Household-level aggregation, encoding, chi-square selection; household-level 90/10 split and 5-fold CV partitions |
| `baseline_*.py` | Logistic regression, random forest, XGBoost, LightGBM, CatBoost (class weighting, validation-selected threshold) |
| `tabpfn_ensemble.py` | Bagged TabPFN ensemble used for explanation |
| `tabpfn_ratio_grid_search.py` | Selection of the positive-class ratio per bag (fold 1, validation only) |
| `tabpfn_train.py` | 5-fold evaluation of the TabPFN ensemble, isotonic calibration, threshold selection |
| `tabpfn_xai.py` | Global and local Kernel SHAP for the complete 8-bag ensemble; figures |
| `feature_labels.py` | Human-readable feature labels used in figures and in `docs/DATA_DICTIONARY.md` |
| `paired_model_comparison.py` | Paired TabPFN-vs-baseline differences with *t*-based 95% CIs |
| `code_dictionary_appendix.py` | Exports the diagnosis/prescription/contact-reason code vocabulary |
| `config.py` | TabPFN, ensemble and path settings |
| `slurm/` | Job scripts used on the UPPMAX Bianca cluster (edit the project id) |
| `docs/DATA_DICTIONARY.md` | Source, reference period, missing-data handling and transformation of every feature; labels of the 287 model features |
| `docs/MODEL_CONFIGURATION.md` | Hyperparameters, imbalance handling and selected thresholds of all models |
| `docs/code_dictionary_appendix_enriched.csv` | Code vocabulary with ICD-10 chapter / ATC group and translations |

## Environment

Python 3.12. Install the pinned packages:

```bash
pip install -r requirements.txt
```

Key versions: tabpfn 6.3.2, torch 2.4.1 (CUDA 12.4), scikit-learn 1.7.2,
xgboost 3.1.2, lightgbm 3.3.5, catboost 1.2.10, shap 0.50.0.

**TabPFN checkpoint.** The pipeline loads the TabPFN-2.5 classifier weights
from a local file and never downloads them (the secure environment has no
internet access). Obtain `tabpfn-v2.5-classifier-v2.5_default.ckpt` from the
official TabPFN distribution and place it at
`tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt`. The checkpoint used
for the reported results has SHA-256
`5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`.

Register-data locations are set with environment variables (defaults are
relative folders): `SWECOV_RAW_DIR` (raw extracts) and `SWECOV_FEATURES_DIR`
(selected-feature extracts).

## Running the pipeline

All steps run from the repository root. Steps 1–3 need no GPU; TabPFN steps
need one (an NVIDIA A100 40 GB was used).

| Step | Command | Output (under `experiment_results/`) |
|---|---|---|
| 1. Labels | `python data_preprocessing.py` | household index with index / secondary / uninfected members |
| 2. Features | `python feature_extraction.py` then `python feature_aggregation.py` | encoded train/val/test folds |
| 3. Baselines | `python baseline_<model>.py` (five scripts) | `*_results_Full/` |
| 4. Positive-class ratio | `python tabpfn_ratio_grid_search.py` | `ratio_grid_search/` |
| 5. TabPFN evaluation | `python tabpfn_train.py` | `TabPFN_XAI_Results_Full/improved_bagging_cv/` |
| 6. Model comparison | `python paired_model_comparison.py` | `paired_model_comparison.csv` (repository root) |
| 7. Explanations | `python tabpfn_xai.py` | `TabPFN_XAI_Results_Full/{global_importance,local_shap}/` |

On a SLURM cluster: `sbatch slurm/run_cpu.sh` (steps 1–3) and
`sbatch slurm/run_gpu.sh <script>` (steps 4, 5, 7).

**Figures only.** SHAP is expensive (every evaluation queries all eight
bags: about 67 GPU-hours for global and 13 GPU-hours for local SHAP). Once the
SHAP matrices exist, all figures can be redrawn in seconds without refitting
or recomputing SHAP:

```bash
python tabpfn_xai.py --figures-only
```

This writes `global_importance/beeswarm.png`, the example waterfalls
`local_shap/waterfall_household_{A,B}.png` and one waterfall per explained
household in `local_shap/all_waterfalls/`. Households whose Kernel SHAP
solution is numerically ill-conditioned (any |SHAP| > 1) are excluded from the
global summary and listed in `global_importance/excluded_rows.json`.

**Sensitivity analysis (alternative secondary-case definition).** The
primary analysis uses a 2-day serial interval and a 14-day wave interval. To
rerun with the 3-day / 18-day definition, set

```bash
export SERIAL_INTERVAL_DAYS=3
export WAVE_INTERVAL_DAYS=18
```

before steps 1–5 (use a separate working directory so that outputs are not
overwritten).

## Reproducibility

Random seeds are fixed: 42 for the household-level split, cross-validation
and bagging; 1 and 0 for the global SHAP explain and background samples; 0 and
42 for the local SHAP background and household selection. Exact package
versions are in `requirements.txt`.

## Citation

If you use this code, please cite the accompanying article (reference to be
added on publication).

## License

To be added.
