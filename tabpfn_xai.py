"""
TabPFN Ensemble - Training + Inference + Explainability Analysis (v7 — Global + Local SHAP)
======================================================================

Usage:
    python tabpfn_xai.py                 # fit ensemble, predict, global + local SHAP
    python tabpfn_xai.py --figures-only  # redraw all figures from saved SHAP
                                         # matrices (no refit, no SHAP; seconds)

Changelog (v6 -> v7, figure fixes):
  - Global SHAP: Kernel SHAP with a small evaluation budget (50 evaluations
    for 287 features) can return an ill-conditioned solution for a few
    households: one pair of features receives huge, exactly cancelling
    attributions (|phi| ~ 1e13) while the row still sums correctly. Such rows
    (any |phi| > MAX_ABS_SHAP; impossible for a probability output) are now
    excluded from the global ranking and the beeswarm, and reported. In the
    reported run this affected 3 of 200 households.
  - Local waterfalls: the base value is now the explainer's expected value
    over its background (recovered as f(x) - sum(phi)), not the mean test-set
    score, so E[f(X)] + sum(phi) = f(x) in every plot.
  - Figure labels are human-readable (feature_labels.py, identical to
    docs/DATA_DICTIONARY.md Section 9); figure width is set explicitly
    because shap ignores plt.figure(figsize=...).
  - The sign-of-mean-SHAP bar chart is no longer produced (the sign of the
    mean SHAP value is not a valid global direction statistic).
  - Local SHAP reuses the saved household indices when present, so figures
    can be regenerated without recomputing SHAP. The two example households
    shown in the manuscript are set in EXAMPLE_HOUSEHOLDS.

Changelog (v5-ensemble-shap -> v6-no-subgroup):
  - Removed subgroup analysis entirely (_define_subgroups, subgroup_analysis,
    SUBGROUP_CONFIG). 8 of the 11 defined subgroups (predicted_pos/neg,
    TP/TN/FP/FN, correct/incorrect) were never referenced by the manuscript,
    and all of them are defined from the model's own output -- the exact
    circularity problem, which independently-defined
    subgroups (household size, age, region, ...) would be needed to answer,
    not a finer split of the same predicted probabilities. Given the full
    8-bag ensemble cost per SHAP call, this was also the single largest
    remaining cost driver. The manuscript's Local SHAP section will instead
    draw its representative examples from the global local_shap_analysis
    below (TP/FP/TN/FN across the whole test set), not from risk strata.

Changelog (v4-resume-top50 -> v5-ensemble-shap):
  - All explanations (global, local) now query the full 8-bag soft-voted
    ensemble via EnsembleModelAdapter, instead of the single highest-OOB-AUC
    bag. (v7 note: the explained ensemble is fitted once on the full
    training pool by tabpfn_ensemble.py; it is a separate fit from the five
    cross-validated ensembles of tabpfn_train.py reported in Table 2.)
  - Permutation importance is removed everywhere. It required hundreds of
    full predict_proba calls per feature and did not scale to 295 features
    x a huge test population, and became even more expensive once every
    predict_proba call queries 8 bags instead of 1. Global feature ranking
    now comes directly from mean(|SHAP|) on a single joint Kernel SHAP
    computation (all features at once), which also produces the beeswarm
    plot.
  - The quartile x label stratified sampler is removed (it only existed to
    pick which top-k features to stratify on using the now-removed PI
    ranking). Explanation/background samples are now drawn by plain
    label-stratified random sampling.
  - Local SHAP (waterfall plots, checkpointing) is functionally unchanged,
    aside from using the ensemble adapter and SHAP-based top-feature list.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             log_loss, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
import shap
import os
import json
import matplotlib.pyplot as plt
import time
import torch
import warnings
warnings.filterwarnings('ignore')

from config import *
from tabpfn_ensemble import TabPFNEnsemble
from feature_labels import label   # plain-English figure labels

# ===========================================================================
# ENSEMBLE MODEL ADAPTER
# ===========================================================================

class EnsembleModelAdapter:
    """
    Wraps a fitted TabPFNEnsemble behind a single-argument `.predict_proba(X)`
    method, so SHAP code can treat it exactly like one sklearn-style model
    while every call actually soft-votes across all bags, i.e. the complete
    ensemble is explained rather than one selected bag. Note that the ensemble
    fitted here (full training pool, tabpfn_ensemble.py) is a separate fit
    from the five cross-validated ensembles of tabpfn_train.py (Table 2).
    """

    def __init__(self, ensemble, feature_names, batch_size=None):
        self.ensemble = ensemble
        self.feature_names = feature_names
        self.batch_size = batch_size or PREDICT_BATCH_SIZE

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X, self.feature_names, batch_size=self.batch_size)


# ===========================================================================
# ANALYSIS CONFIGURATION
# ===========================================================================

GLOBAL_SHAP_CONFIG = {
    'n_explain':       200,   # households sampled (stratified by label) for global SHAP
    'n_background':     25,
    'max_evals':        50,
    'batch_size':       50,
    'top_n_display':    20,
}

LOCAL_SHAP_CONFIG = {
    'n_per_outcome':     4,
    'n_per_feat_value':  2,
    'n_top_features':    5,
    'max_evals':         120,
    'n_background':      30,
    'batch_size':        50,
    'max_waterfall':     12,
}

# Example households shown in the manuscript (test-set row indices, fold 1)
EXAMPLE_HOUSEHOLDS = {'A': 2536, 'B': 3238}

# Rows of the global SHAP matrix with any |phi| above this value come from an
# ill-conditioned Kernel SHAP solve and are excluded from the global summary
MAX_ABS_SHAP = 1.0

# shap ignores plt.figure(figsize=...); widths are applied via plot_size
# (beeswarm) and set_size_inches after drawing (waterfall)
FIG_WIDTH = 11   # inches


# ===========================================================================
# REMAINING TIME ESTIMATION
# ===========================================================================

def estimate_remaining_time(global_shap_done=False, global_bee_batches_done=0,
                            global_bee_total_batches=0):
    """
    Rough wall-clock estimate for the SHAP-only pipeline. Every predict_proba
    call now queries the full 8-bag ensemble instead of one bag (~8x more
    model forward passes per call than the earlier best-bag-only pipeline),
    but permutation importance and subgroup analysis -- previously the
    dominant costs -- are both removed entirely, so the net effect is
    normally a large reduction in total runtime.
    """
    min_per_batch = 5.0  # empirical minutes per 50-sample Kernel SHAP batch

    print(f"\n{'='*80}")
    print("Remaining Time Estimation")
    print(f"{'='*80}")

    total_min = 0.0

    # Global SHAP (beeswarm + ranking, one joint computation)
    if not global_shap_done:
        bee_remaining = max(0, global_bee_total_batches - global_bee_batches_done)
        global_min = bee_remaining * min_per_batch
        print(f"  Global SHAP         : {global_min:.0f} min  "
              f"({global_bee_batches_done}/{global_bee_total_batches} batches done)")
        total_min += global_min
    else:
        print(f"  Global SHAP         : already done (skipped)")

    # Local SHAP
    local_samples = (LOCAL_SHAP_CONFIG['n_per_outcome'] * 4 +
                     LOCAL_SHAP_CONFIG['n_per_feat_value'] * 2 *
                     LOCAL_SHAP_CONFIG['n_top_features'] * 2)
    local_batches = (local_samples + LOCAL_SHAP_CONFIG['batch_size'] - 1) \
                    // LOCAL_SHAP_CONFIG['batch_size']
    local_min = local_batches * min_per_batch
    print(f"  Local SHAP          : {local_min:.0f} min  (~{local_samples} samples)")
    total_min += local_min

    print(f"\n  {'─'*50}")
    print(f"  Total remaining    : {total_min:.0f} min  ({total_min/60:.1f} h)")
    print(f"  Recommended node   : {total_min/60*1.2:.0f} h  (+20% buffer)")
    print(f"{'='*80}\n")
    return total_min


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def load_and_preprocess(fold: int):
    base_path = FOLDS_PATH
    drop_cols  = DROP_COLS_BASE + DELETED_COLS
    label_col  = 'label'

    train_df = pd.read_csv(f'{base_path}/train_fold_{fold}.csv', encoding='latin1')
    val_df   = pd.read_csv(f'{base_path}/val_fold_{fold}.csv',   encoding='latin1')
    test_df  = pd.read_csv(f'{base_path}/test_fold_{fold}.csv',  encoding='latin1')

    for df in [train_df, val_df, test_df]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)
        df['label'] = (df['secondary_cases_count'] > 0).astype(int)
        df.drop('secondary_cases_count', axis=1, inplace=True)

    feature_names = [c for c in train_df.columns if c != label_col]

    pool_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train = pool_df[feature_names]
    y_train = pool_df[label_col].values

    # Use the full held-out test fold as-is (same 25,248-household test set
    # evaluated in tabpfn_train.py's Table 2). Earlier versions drew a further
    # TEST_SAMPLE_RATIO-sized stratified subsample here, which silently
    # evaluated XAI on a different, smaller test set than the one reported
    # for predictive performance.
    X_test = test_df[feature_names]
    y_test = test_df[label_col].values

    print(f"  Train : {len(X_train):,} | {len(feature_names):,} features")
    print(f"  Test  : {len(X_test):,}")
    print(f"  Pos rate (train): {y_train.mean()*100:.1f}%")
    return X_train, y_train, X_test, y_test, feature_names


def compute_detailed_metrics(y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy':           float((y_true == y_pred).mean()),
        'roc_auc':            float(roc_auc_score(y_true, y_prob)),
        'macro_f1':           float(report.get('macro avg', {}).get('f1-score', np.nan)),
        'weighted_f1':        float(report.get('weighted avg', {}).get('f1-score', np.nan)),
        'log_loss':           float(log_loss(y_true, np.column_stack([1-y_prob, y_prob]))),
        'balanced_accuracy':  float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa':        float(cohen_kappa_score(y_true, y_pred)),
        'mcc':                float(matthews_corrcoef(y_true, y_pred)),
        'class_1_auc':        roc_auc_score(y_true, y_prob),
        'class_1_pr_auc':     average_precision_score(y_true, y_prob),
        'class_1_f1':         float(report.get('1', {}).get('f1-score', np.nan)),
        'class_1_recall':     float(report.get('1', {}).get('recall', np.nan)),
        'class_1_precision':  float(report.get('1', {}).get('precision', np.nan)),
        'confusion_matrix':   confusion_matrix(y_true, y_pred).tolist(),
    }


# ===========================================================================
# SHAP WITH BATCH-LEVEL CHECKPOINTS
# ===========================================================================

def compute_shap_small_with_checkpoint(model, X_explain, X_background,
                                       feature_names, checkpoint_dir,
                                       prefix='shap',
                                       max_evals=100, batch_size=50):
    """
    KernelExplainer SHAP with per-batch checkpointing.
    Each batch is saved as {checkpoint_dir}/{prefix}_batch_{i}.npy.
    Completed batches are detected at startup and skipped automatically.

    Returns shap_matrix: (len(X_explain), len(feature_names))
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    n           = len(X_explain)
    n_batch     = (n + batch_size - 1) // batch_size
    batch_files = [os.path.join(checkpoint_dir, f'{prefix}_batch_{i}.npy')
                   for i in range(n_batch)]

    # Detect how many batches are already complete
    first_pending = 0
    for i, fp in enumerate(batch_files):
        if os.path.exists(fp):
            first_pending = i + 1
        else:
            break

    if first_pending == n_batch:
        print(f"    SHAP checkpoints complete ({n_batch} batches), loading directly")
        return np.vstack([np.load(fp) for fp in batch_files])

    if first_pending > 0:
        print(f"    Resuming from batch {first_pending+1}/{n_batch} "
              f"({first_pending} batches already done)")

    print(f"    SHAP: {n} samples x {max_evals} evals x {len(X_background)} background "
          f"(ensemble, all bags)")

    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer  = shap.KernelExplainer(predict_fn, X_background, link='identity')

    t0 = time.time()
    for i in range(first_pending, n_batch):
        start   = i * batch_size
        end     = min(start + batch_size, n)
        X_batch = X_explain.iloc[start:end]
        pct     = (i + 1) / n_batch * 100
        print(f"\r      batch [{i+1}/{n_batch}] {pct:.0f}%", end='', flush=True)

        sv = explainer.shap_values(X_batch, nsamples=max_evals, silent=True)
        np.save(batch_files[i], sv)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n    -> New batches complete, elapsed {elapsed/60:.1f} min")

    return np.vstack([np.load(fp) for fp in batch_files])


def _stratified_sample(X, y, n, random_state=42):
    """
    Draw a size-n sample of households, stratified by label when possible.
    Falls back to plain random sampling for single-class groups or when a
    stratified split cannot be formed. Returns (X_sample, y_sample) with a
    reset, contiguous index.
    """
    X_reset = X.reset_index(drop=True)
    if n >= len(X_reset):
        return X_reset, y

    rng = np.random.default_rng(random_state)
    if len(np.unique(y)) < 2:
        idx = rng.choice(len(X_reset), size=n, replace=False)
    else:
        try:
            idx, _ = train_test_split(
                np.arange(len(X_reset)), train_size=n,
                stratify=y, random_state=random_state)
        except ValueError:
            idx = rng.choice(len(X_reset), size=n, replace=False)

    idx = np.sort(idx)
    return X_reset.iloc[idx].reset_index(drop=True), y[idx]


def _shap_beeswarm(shap_matrix, X_explain, ordered_features, title, out_path, top_n):
    feat_index    = {f: i for i, f in enumerate(X_explain.columns)}
    ordered_idx   = [feat_index[f] for f in ordered_features if f in feat_index][:top_n]
    ordered_names = [X_explain.columns[i] for i in ordered_idx]

    shap.summary_plot(
        shap_matrix[:, ordered_idx], X_explain[ordered_names],
        feature_names=[label(f) for f in ordered_names],
        max_display=len(ordered_names), plot_type='dot', show=False,
        plot_size=(FIG_WIDTH, 0.42 * len(ordered_names) + 1.5))
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def _waterfall(shap_values, base_value, data_row, feature_names, title, out_path,
               max_display):
    shap.waterfall_plot(
        shap.Explanation(values=shap_values, base_values=base_value,
                         data=data_row, feature_names=[label(f) for f in feature_names]),
        max_display=max_display, show=False)
    # Widen after drawing: bars only get longer, so shap's placement of the
    # value labels (inside / beside each bar) stays valid.
    plt.gcf().set_size_inches(FIG_WIDTH, 0.5 * max_display + 1.5)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ===========================================================================
# GLOBAL ANALYSIS WITH CHECKPOINT RESUME
# ===========================================================================

def global_shap_analysis(model, X_test, y_test, feature_names,
                          cfg=GLOBAL_SHAP_CONFIG, save_dir=None):
    """
    Single joint Kernel SHAP computation over all features on a
    label-stratified sample of the test set. Produces both the global
    feature ranking (mean |SHAP|) and the beeswarm plot -- there is no
    separate permutation-importance pass.
    """
    print(f"\n{'='*80}")
    print("GLOBAL SHAP ANALYSIS")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'global_importance')
    os.makedirs(save_dir, exist_ok=True)

    top_n     = cfg['top_n_display']
    shap_csv  = os.path.join(save_dir, 'global_shap_importance.csv')
    shap_npy  = os.path.join(save_dir, 'global_shap_values.npy')
    exp_csv   = os.path.join(save_dir, 'global_shap_explain_sample.csv')
    bee_path  = os.path.join(save_dir, 'beeswarm.png')
    ckpt_dir  = os.path.join(save_dir, 'shap_checkpoints')

    X_explain, y_explain = _stratified_sample(X_test, y_test, cfg['n_explain'], random_state=1)
    X_bg, _              = _stratified_sample(X_test, y_test, cfg['n_background'], random_state=0)

    if os.path.exists(shap_npy) and os.path.exists(exp_csv):
        print(f"\n  Saved SHAP matrix found, loading: {shap_npy}")
        shap_matrix = np.load(shap_npy)
        X_explain   = pd.read_csv(exp_csv)[feature_names]
    else:
        if model is None:
            raise FileNotFoundError(f'{shap_npy} not found; run without --figures-only first')
        print(f"\n  Kernel SHAP over all {len(feature_names)} features "
              f"(n={len(X_explain):,} households)...")
        t0 = time.time()
        shap_matrix = compute_shap_small_with_checkpoint(
            model, X_explain, X_bg, feature_names,
            checkpoint_dir=ckpt_dir, prefix='global',
            max_evals=cfg['max_evals'], batch_size=cfg['batch_size'])
        print(f"  Global SHAP done in {(time.time()-t0)/60:.1f} min")
        np.save(shap_npy, shap_matrix)          # raw matrix, all households
        X_explain.to_csv(exp_csv, index=False)

    # Exclude ill-conditioned rows (see module docstring) from the summary
    bad  = np.where((np.abs(shap_matrix) > MAX_ABS_SHAP).any(axis=1))[0]
    keep = np.setdiff1d(np.arange(len(shap_matrix)), bad)
    if len(bad):
        print(f"  Excluding {len(bad)} ill-conditioned household(s) from the "
              f"global summary: rows {bad.tolist()}")
    phi, X_keep = shap_matrix[keep], X_explain.iloc[keep].reset_index(drop=True)
    nonzero = (phi != 0).sum(axis=1)
    print(f"  Non-zero attributions per household: min={nonzero.min()} "
          f"median={int(np.median(nonzero))} max={nonzero.max()} (of {phi.shape[1]})")

    shap_df = pd.DataFrame({
        'feature':        feature_names,
        'label':          [label(f) for f in feature_names],
        'mean_abs_shap':  np.abs(phi).mean(axis=0),
        'std_abs_shap':   np.abs(phi).std(axis=0),
        'mean_shap':      phi.mean(axis=0),
        'n_nonzero':      (phi != 0).sum(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_df['rank'] = np.arange(1, len(shap_df) + 1)
    shap_df.to_csv(shap_csv, index=False)
    with open(os.path.join(save_dir, 'excluded_rows.json'), 'w') as fh:
        json.dump({'excluded_rows': bad.tolist(), 'n_used': int(len(keep)),
                   'max_abs_shap': MAX_ABS_SHAP}, fh, indent=2)

    _shap_beeswarm(
        phi, X_keep, shap_df['feature'].tolist(),
        title=f'Kernel SHAP, full 8-bag ensemble (n={len(keep)} households)',
        out_path=bee_path, top_n=top_n)
    print(f"  Beeswarm saved: {bee_path}")

    print(f"\n  Top 10 features (mean |SHAP|):")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {int(row['rank']):2d}. {row['label']:40s} | {row['mean_abs_shap']:.4f}")

    return shap_df, shap_matrix, X_explain


# ===========================================================================
# LOCAL SHAP
# ===========================================================================

def _collect_local_indices(y_test, y_pred, y_prob, feature_vals,
                           top_feature_names, cfg):
    """
    Select a representative set of sample indices for local SHAP waterfall plots.
    Covers TP/FP/TN/FN outcomes as well as high/low values of the top features.
    """
    rng    = np.random.default_rng(42)
    chosen = set()

    def pick(mask, n):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        k = min(n, len(idx))
        return rng.choice(idx, k, replace=False).tolist()

    # One sample per confusion-matrix cell
    n  = cfg['n_per_outcome']
    tp = (y_pred == 1) & (y_test == 1)
    fp = (y_pred == 1) & (y_test == 0)
    tn = (y_pred == 0) & (y_test == 0)
    fn = (y_pred == 0) & (y_test == 1)
    for mask in [tp, fp, tn, fn]:
        chosen.update(pick(mask, n))

    # Samples split by above/below-median feature value x label
    n2      = cfg['n_per_feat_value']
    medians = feature_vals.median()
    for feat in top_feature_names[:cfg['n_top_features']]:
        if feat not in feature_vals.columns:
            continue
        high = feature_vals[feat] >= medians[feat]
        low  = ~high
        for val_mask in [high, low]:
            for lab in [0, 1]:
                lab_mask = y_test == lab
                combined = val_mask.values & lab_mask
                chosen.update(pick(combined, n2))

    return sorted(chosen)


def local_shap_analysis(model, X_test, y_test, y_pred, y_prob, feature_names,
                        top_feature_names, cfg=LOCAL_SHAP_CONFIG, save_dir=None):
    print(f"\n{'='*80}")
    print("LOCAL SHAP ANALYSIS")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'local_shap')
    os.makedirs(save_dir, exist_ok=True)

    npy_path = os.path.join(save_dir, 'local_shap_values.npy')
    idx_path = os.path.join(save_dir, 'local_shap_indices.npy')
    ckpt_dir = os.path.join(save_dir, 'shap_checkpoints')

    if os.path.exists(npy_path) and os.path.exists(idx_path):
        # Reuse the saved selection so that saved SHAP values stay aligned
        # (and figures can be redrawn without recomputing SHAP)
        indices     = np.load(idx_path).tolist()
        shap_matrix = np.load(npy_path)
        print(f"  Saved local SHAP found ({len(indices)} households), loading")
    else:
        if model is None:
            raise FileNotFoundError(f'{npy_path} not found; run without --figures-only first')
        indices = _collect_local_indices(
            y_test, y_pred, y_prob, X_test, top_feature_names, cfg)
        print(f"  Selected {len(indices)} samples for local SHAP")
        bg_idx = np.random.default_rng(0).choice(
            len(X_test), cfg['n_background'], replace=False)
        X_bg = X_test.iloc[bg_idx].reset_index(drop=True)
        t0 = time.time()
        shap_matrix = compute_shap_small_with_checkpoint(
            model, X_test.iloc[indices].reset_index(drop=True), X_bg, feature_names,
            checkpoint_dir=ckpt_dir, prefix='local',
            max_evals=cfg['max_evals'], batch_size=cfg['batch_size'])
        np.save(npy_path, shap_matrix)
        np.save(idx_path, np.array(indices))
        print(f"  Local SHAP done in {(time.time()-t0)/60:.1f} min")

    X_local = X_test.iloc[indices].reset_index(drop=True)

    # Build JSON explanations and waterfall plots
    explanations = []
    for i, orig_idx in enumerate(indices):
        shap_vals = shap_matrix[i]
        feat_shap = sorted(zip(feature_names, shap_vals),
                           key=lambda x: x[1], reverse=True)
        y_pred_i = int(y_pred[orig_idx])
        y_true_i = int(y_test[orig_idx])
        outcome  = {(1, 1): 'TP', (1, 0): 'FP',
                    (0, 0): 'TN', (0, 1): 'FN'}.get((y_pred_i, y_true_i), '?')
        explanations.append({
            'sample_index': int(orig_idx), 'outcome': outcome,
            'prediction':   y_pred_i, 'probability': float(y_prob[orig_idx]),
            'true_label':   y_true_i,
            'top_positive': {f: float(s) for f, s in feat_shap[:5]},
            'top_negative': {f: float(s) for f, s in feat_shap[-5:]},
        })

    with open(os.path.join(save_dir, 'local_explanations.json'), 'w') as fh:
        json.dump(explanations, fh, indent=2, ensure_ascii=False)

    # Base value = explainer expected value over the background, recovered as
    # f(x) - sum(phi); identical for every household by construction
    base_values = np.array([y_prob[idx] - shap_matrix[k].sum()
                            for k, idx in enumerate(indices)])
    print(f"  E[f(X)] over background: {base_values.mean():.3f} "
          f"(spread {base_values.max() - base_values.min():.1e})")

    all_dir = os.path.join(save_dir, 'all_waterfalls')
    os.makedirs(all_dir, exist_ok=True)
    for k, idx in enumerate(indices):
        _waterfall(shap_matrix[k], base_values[k], X_local.iloc[k].values,
                   feature_names, title=f'Test household {idx}',
                   out_path=os.path.join(all_dir, f'waterfall_sample{idx}.png'),
                   max_display=cfg['max_waterfall'])
    for tag, idx in EXAMPLE_HOUSEHOLDS.items():
        if idx not in indices:
            print(f"  Example household {tag} (index {idx}) not among the explained households")
            continue
        k = indices.index(idx)
        _waterfall(shap_matrix[k], base_values[k], X_local.iloc[k].values,
                   feature_names, title=f'Example household {tag}',
                   out_path=os.path.join(save_dir, f'waterfall_household_{tag}.png'),
                   max_display=cfg['max_waterfall'])
    print(f"  Waterfalls: {len(indices)} in all_waterfalls/, "
          f"examples {list(EXAMPLE_HOUSEHOLDS)} in {save_dir}")
    print(f"  Local SHAP complete")
    return explanations


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_tabpfn_pipeline(
    fold: int = 1,
    run_global:   bool = True,
    run_local:    bool = True,
    figures_only: bool = False,
):
    """
    figures_only=True skips fitting, prediction and SHAP computation and only
    redraws the figures from saved outputs (predictions CSV, SHAP matrices).
    """
    print(f"\n{'='*80}")
    print("TabPFN Pipeline v7 — Global + Local SHAP")
    print(f"{'='*80}")
    t_pipeline = time.time()

    for sub in ['predictions', 'global_importance', 'local_shap']:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # ---- 1. Load ----
    print(f"\n{'='*80}")
    print("Step 1: Load data")
    print(f"{'='*80}")
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess(fold)

    # Detect checkpoint state for time estimation
    global_shap_path = os.path.join(OUTPUT_DIR, 'global_importance',
                                    'global_shap_importance.csv')
    global_shap_done = os.path.exists(global_shap_path)
    ckpt_dir_bee     = os.path.join(OUTPUT_DIR, 'global_importance', 'shap_checkpoints')
    bee_done_batches = 0
    if os.path.exists(ckpt_dir_bee):
        files = [f for f in os.listdir(ckpt_dir_bee) if f.startswith('global_batch_')]
        bee_done_batches = len(files)
    bee_total = (GLOBAL_SHAP_CONFIG['n_explain'] + GLOBAL_SHAP_CONFIG['batch_size'] - 1) \
                // GLOBAL_SHAP_CONFIG['batch_size']

    pred_csv = os.path.join(OUTPUT_DIR, 'predictions', f'predictions_fold_{fold}.csv')
    xai_results = {}

    if figures_only:
        if not os.path.exists(pred_csv):
            raise FileNotFoundError(f'{pred_csv} not found; run the full pipeline first')
        pred_df = pd.read_csv(pred_csv)
        y_prob, y_pred = pred_df['prob_1'].values, pred_df['prediction'].values
        metrics = compute_detailed_metrics(y_test, y_pred, y_prob)
        shap_df = global_shap_analysis(
            None, X_test, y_test, feature_names,
            save_dir=os.path.join(OUTPUT_DIR, 'global_importance'))[0]
        local_shap_analysis(
            None, X_test, y_test, y_pred, y_prob, feature_names,
            top_feature_names=shap_df['feature'].tolist()[:LOCAL_SHAP_CONFIG['n_top_features']],
            save_dir=os.path.join(OUTPUT_DIR, 'local_shap'))
        print(f"\nFigures redrawn -> {OUTPUT_DIR}")
        return pred_df, metrics, xai_results

    estimate_remaining_time(
        global_shap_done=global_shap_done,
        global_bee_batches_done=bee_done_batches,
        global_bee_total_batches=bee_total)

    # ---- 2. Train ----
    print(f"\n{'='*80}")
    print(f"Step 2: Train TabPFN ensemble  ({len(feature_names)} features)")
    print(f"{'='*80}")
    t0 = time.time()
    ensemble = TabPFNEnsemble(
        ENSEMBLE_CONFIG, TABPFN_PARAMS,
        max_samples=TABPFN_MAX_SAMPLES,
        max_features=TABPFN_MAX_FEATURES)
    ensemble.fit(X_train, y_train, feature_names)
    print(f"  {(time.time()-t0)/60:.1f} min")

    # ---- 3. Predict ----
    print(f"\n{'='*80}")
    print("Step 3: Predictions")
    print(f"{'='*80}")
    if os.path.exists(pred_csv):
        print(f"  Predictions already exist, loading")
        pred_df = pd.read_csv(pred_csv)
        y_prob  = pred_df['prob_1'].values
        y_pred  = pred_df['prediction'].values
    else:
        t0 = time.time()
        y_proba = ensemble.predict_proba(X_test, feature_names,
                                         batch_size=PREDICT_BATCH_SIZE)
        y_prob  = y_proba[:, 1]
        y_pred  = (y_prob > 0.5).astype(int)
        pred_df = pd.DataFrame({
            'prob_0':     y_proba[:, 0], 'prob_1': y_prob,
            'prediction': y_pred,        'true_label': y_test,
        })
        pred_df.to_csv(pred_csv, index=False)
        print(f"  {(time.time()-t0):.1f}s")

    metrics_path = os.path.join(OUTPUT_DIR, 'predictions',
                                f'metrics_fold_{fold}.json')
    if not os.path.exists(metrics_path):
        metrics = compute_detailed_metrics(y_test, y_pred, y_prob)
        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh, indent=2)
    else:
        with open(metrics_path) as fh:
            metrics = json.load(fh)

    print(f"  ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['macro_f1']:.4f}")

    # All explanations query the full soft-voted ensemble (all bags), not a
    # single "best" bag.
    model_for_xai = EnsembleModelAdapter(ensemble, feature_names)
    print(f"  Explaining the full {len(ensemble.models)}-bag ensemble "
          f"(bag OOB AUCs: min={min(ensemble.oob_scores):.4f}, "
          f"max={max(ensemble.oob_scores):.4f})")

    # ---- 4. Global ----
    if run_global:
        print(f"\n{'='*80}")
        print("Step 4: Global SHAP Analysis")
        print(f"{'='*80}")
        t0 = time.time()
        shap_df, _, _ = global_shap_analysis(
            model_for_xai, X_test, y_test, feature_names,
            save_dir=os.path.join(OUTPUT_DIR, 'global_importance'))
        xai_results['global_shap'] = shap_df
        print(f"  {(time.time()-t0)/60:.1f} min")
    else:
        # Even when skipping global analysis, load existing ranking for local sample selection
        if os.path.exists(global_shap_path):
            shap_df = pd.read_csv(global_shap_path)
            print(f"  [Global SHAP ranking loaded from checkpoint]")
        else:
            shap_df = pd.DataFrame({'feature':       feature_names,
                                    'mean_abs_shap': 0,
                                    'rank':          range(1, len(feature_names)+1)})

    top_feat_names = shap_df['feature'].tolist()[:LOCAL_SHAP_CONFIG['n_top_features']]

    # ---- 5. Local SHAP ----
    if run_local:
        print(f"\n{'='*80}")
        print("Step 5: Local SHAP Analysis")
        print(f"{'='*80}")
        t0 = time.time()
        local_exp = local_shap_analysis(
            model_for_xai, X_test, y_test, y_pred, y_prob, feature_names,
            top_feature_names=top_feat_names,
            save_dir=os.path.join(OUTPUT_DIR, 'local_shap'))
        xai_results['local'] = local_exp
        print(f"  {(time.time()-t0)/60:.1f} min")

    total = time.time() - t_pipeline
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE — {total/3600:.2f} h  ({total/60:.0f} min)")
    print(f"Results -> {OUTPUT_DIR}")
    print(f"{'='*80}")
    return pred_df, metrics, xai_results


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--figures-only', action='store_true',
                    help='redraw figures from saved SHAP matrices; no fitting or SHAP')
    ap.add_argument('--skip-global', action='store_true')
    ap.add_argument('--skip-local', action='store_true')
    args = ap.parse_args()

    if TABPFN_PARAMS.get('device') == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        TABPFN_PARAMS['device'] = 'cpu'
    elif torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pred_df, metrics, results = run_tabpfn_pipeline(
        fold=args.fold,
        run_global=not args.skip_global,
        run_local=not args.skip_local,
        figures_only=args.figures_only,
    )