"""
Paired statistical comparison of TabPFN vs. each baseline across folds, so
that small performance differences are reported with confidence intervals
rather than described as superiority.

Reads the per-fold summary CSVs already written by tabpfn_train.py and each
baseline_*.py script (same 5 folds, same held-out test set per fold), and runs a paired comparison
(paired t-test and Wilcoxon signed-rank, since n=5 folds is small and the
t-test's normality assumption is shaky at that sample size -- reporting both
is more defensible than either alone) on the metrics Table 2 currently makes
comparative claims about: ROC-AUC, MCC, and class-1 (positive) F1.

No GPU needed; only pandas + scipy. Safe to run anywhere the *_summary.csv
files are present (this Mac or Bianca).

Usage:
    python paired_model_comparison.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

TABPFN_SUMMARY = 'experiment_results/TabPFN_XAI_Results_Full/improved_bagging_cv/tabpfn_cv_summary.csv'

BASELINES = {
    'Logistic Regression': 'experiment_results/LR_results_Full/household_lr_summary.csv',
    'Random Forest':       'experiment_results/RF_results_Full/household_rf_summary.csv',
    'XGBoost':             'experiment_results/XGB_results_Full/household_xgboost_summary.csv',
    'LightGBM':            'experiment_results/LightGBM_results_Full/household_lightgbm_summary.csv',
    'CatBoost':            'experiment_results/CatBoost_results_Full/household_catboost_summary.csv',
}

METRICS = {
    'test_roc_auc':     'ROC-AUC',
    'test_mcc':         'MCC',
    'test_class1_f1':   'F1+',
    'test_class1_recall': 'Recall+',
}


def paired_compare(a: np.ndarray, b: np.ndarray):
    """Paired t-test and Wilcoxon signed-rank test; returns dict of results.

    CI uses the t distribution's critical value at n-1 degrees of freedom,
    not the normal-approximation 1.96 -- with n=5 folds (df=4), t_crit
    (~2.776) is meaningfully larger than 1.96, and using 1.96 would make the
    CI artificially narrow (overconfident) at this small a sample size.
    """
    diff = a - b
    n = len(diff)
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # Wilcoxon fails if all differences are zero or n is too small
        w_stat, w_p = np.nan, np.nan
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return {
        'mean_diff': diff.mean(),
        'sd_diff': diff.std(ddof=1),
        'ci95_lo': diff.mean() - t_crit * se,
        'ci95_hi': diff.mean() + t_crit * se,
        't_stat': t_stat, 't_p': t_p,
        'w_stat': w_stat, 'w_p': w_p,
    }


def main():
    if not os.path.exists(TABPFN_SUMMARY):
        raise FileNotFoundError(
            f"{TABPFN_SUMMARY} not found -- run tabpfn_train.py first.")

    tabpfn = pd.read_csv(TABPFN_SUMMARY).sort_values('fold').reset_index(drop=True)
    print(f"Loaded TabPFN results: {len(tabpfn)} folds")

    rows = []
    for name, path in BASELINES.items():
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        base = pd.read_csv(path).sort_values('fold').reset_index(drop=True)
        if len(base) != len(tabpfn):
            print(f"  [skip] {name}: {len(base)} folds != TabPFN's {len(tabpfn)} folds")
            continue

        for metric_col, metric_label in METRICS.items():
            if metric_col not in base.columns or metric_col not in tabpfn.columns:
                continue
            a = tabpfn[metric_col].values
            b = base[metric_col].values
            res = paired_compare(a, b)
            rows.append({
                'baseline': name,
                'metric': metric_label,
                'tabpfn_mean': a.mean(),
                'baseline_mean': b.mean(),
                **res,
            })

    result_df = pd.DataFrame(rows)
    out_path = 'paired_model_comparison.csv'
    result_df.to_csv(out_path, index=False)

    print(f"\nSaved {len(result_df)} rows to {out_path}\n")
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', None)
    print(result_df[['baseline', 'metric', 'tabpfn_mean', 'baseline_mean',
                      'mean_diff', 'ci95_lo', 'ci95_hi', 't_p', 'w_p']]
          .round(4).to_string(index=False))
    print("\nInterpretation: mean_diff = TabPFN - baseline (paired across the "
          "same 5 folds). A 95% CI crossing zero, or t_p/w_p > 0.05, means the "
          "difference is not statistically distinguishable from zero at "
          "conventional significance and should not be described as "
          "'superior'.")


if __name__ == '__main__':
    main()
