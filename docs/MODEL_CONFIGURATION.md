# Model Configuration

Model-configuration reference for the household SARS-CoV-2 secondary-case
prediction study: implementation details of all six models, kept with the
code.

**Shared preprocessing.** All six models (five baselines and TabPFN) are
trained on the identical preprocessed feature matrices produced by the
shared pipeline (`feature_aggregation.py`): the same categorical encoding
and the same per-fold `StandardScaler`, fit only on that fold's training
partition and applied without refitting to validation and test. There is no
asymmetry in feature representation between TabPFN and the baselines. See
`DATA_DICTIONARY.md` for the full feature-level documentation.

**Imbalance handling and missing-data treatment.** Each model uses the
mechanism natively available to its architecture rather than an identical
mechanism across all models -- TabPFN has no loss-reweighting parameter, so
positive-class oversampling within the bagging ensemble is the closest
architecturally available equivalent. What makes the six-way comparison fair
is not identical imbalance handling, but the identical, validation-only
classification-threshold-selection criterion applied to every model (grid
search for maximum F1+ on the validation partition only, applied fixed to
test -- see `tabpfn_train.py::select_threshold` and the matching function in
each `baseline_*.py`).

| Model | Imbalance handling | Missing data | Key hyperparameters |
|---|---|---|---|
| Logistic Regression | `class_weight='balanced'` | Mean imputation (`SimpleImputer`, fit on train only) | L2 penalty, C=1.0, solver=lbfgs, max_iter=1000 |
| Random Forest | `class_weight='balanced'` | Mean imputation | n_estimators=200, max_depth=15 |
| XGBoost | `scale_pos_weight` (from fold class ratio, ≈4:1) | Native | n_estimators=200, max_depth=8, tree_method=hist |
| LightGBM | `scale_pos_weight` | Native | n_estimators=200, max_depth=8 |
| CatBoost | `scale_pos_weight` | Native | iterations=200, depth=8, learning_rate=0.05 |
| TabPFN (per bag) | Positive-class oversampling within bagging (`target_ratio=50%`, validation-selected via `tabpfn_ratio_grid_search.py`) | Native | K=8 bags, n_bag=40,000; checkpoint `tabpfn-v2.5-classifier-v2.5_default.ckpt` |

**Validation-selected operating points** (mean ± SD across 5 folds; the
resulting test-set performance is reported in Table 2 of the paper):

| Model | Selected threshold |
|---|---|
| Logistic Regression | 0.462 ± 0.013 |
| Random Forest | 0.412 ± 0.015 |
| XGBoost | 0.420 ± 0.007 |
| LightGBM | 0.462 ± 0.008 |
| CatBoost | 0.464 ± 0.011 |
| TabPFN | 0.484 ± 0.015 |

**Reproducibility.** Random seeds: 42 for the train/test split and all
stratified k-fold partitioning (`feature_aggregation.py`); 42 for the
bagging ensemble (`tabpfn_train.py::CV_CONFIG`); 0/1 respectively for the
SHAP background/explain-sample draws (`tabpfn_xai.py`). Exact package
versions and the GitHub commit corresponding to the submitted manuscript:
Python 3.12.12; tabpfn 6.3.2 with checkpoint
`tabpfn-v2.5-classifier-v2.5_default.ckpt` (SHA-256
`5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`);
torch 2.4.1+cu124, scikit-learn 1.7.2, xgboost 3.1.2, lightgbm 3.3.5,
catboost 1.2.10, shap 0.50.0, numpy 2.2.6, pandas 2.3.2. Full list:
`requirements.txt`. Code version: the tagged release of this repository that
accompanies the publication.
