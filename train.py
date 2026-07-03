"""
Full model training pipeline — correct methodology from the start.

Approach (learned from diagnostic analysis):
  - Purged walk-forward CV: no data leakage between folds
  - Sparse training and evaluation: no label overlap inflation
  - 75s gap (60s feature window + 15s target horizon) between train/val
  - Final test = last fold, never seen during model selection
  - All models trained with same constraints

Models:
  Linear (RobustScaler + TimeSeriesSplit CV):
    OLS (HAC standard errors), Ridge, Lasso
  Tree (raw features, depth=3):
    Decision Tree, Random Forest (300 trees)
    XGBoost (up to 300 trees, early stopping, depth=3)
    LightGBM (up to 300 trees, early stopping, depth=3)
  Quantile XGBoost (10th/25th/50th/75th/90th percentile)

Horizons: 1s, 3s, 5s, 10s, 15s
Folds: 4 (last fold = final test, never used for model selection)

Run:
    python train.py
"""

import gc
import logging
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.stats as stats
import xgboost as xgb
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

from features.targets import build_targets

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

HORIZONS_SEC    = [1, 3, 5, 10, 15]
QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# Nested intervals for fan chart calibration check
# Each tuple: (lower_q, upper_q, label)
INTERVALS = [
    (0.05, 0.95, "90pct"),
    (0.10, 0.90, "80pct"),
    (0.20, 0.80, "60pct"),
    (0.30, 0.70, "40pct"),
    (0.40, 0.60, "20pct"),
]
N_FOLDS         = 4        # last fold = final test
PURGE_S         = 15       # max target horizon
EMBARGO_S       = 60       # max feature window (300s/180s removed, 60s is longest)
GAP_S           = PURGE_S + EMBARGO_S   # 75 seconds total gap
OUT_DIR         = "data/results"
os.makedirs(OUT_DIR, exist_ok=True)

# Feature sets loaded from premodel_checks output
FEATURES_VIF10 = [
    "vwap_dev_1s", "vwap_dev_5s", "vwap_dev_10s", "vwap_dev_30s",
    "trade_imbalance_1s", "trade_imbalance_5s",
    "trade_imbalance_10s", "trade_imbalance_30s",
    "book_imbalance_l5",
    "ofi_l1",
    "momentum_2s", "momentum_5s", "momentum_10s",
]

ALL_FEATURES = [
    "vwap_dev_1s", "vwap_dev_5s", "vwap_dev_10s", "vwap_dev_30s",
    "trade_imbalance_1s", "trade_imbalance_5s",
    "trade_imbalance_10s", "trade_imbalance_30s",
    "trade_volume_5s", "trade_volume_30s", "trade_volume_60s",
    "trade_count_5s", "trade_count_30s", "trade_count_60s",
    "realized_vol_10s", "realized_vol_30s", "realized_vol_60s",
    "book_imbalance_l1", "book_imbalance_l5", "book_imbalance_l10",
    "book_imbalance_l15", "book_imbalance_l20",
    "ofi_l1",
    "momentum_2s", "momentum_5s", "momentum_10s",
]


# ── Metrics ────────────────────────────────────────────────────────────

def da(y_true, y_pred):
    """Directional accuracy on non-zero actual moves."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())

def pinball(y_true, y_pred, q):
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))

def coverage(y_true, y_lo, y_hi):
    return float(((y_true >= y_lo) & (y_true <= y_hi)).mean())

def metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_r2":   round(r2_score(y_true, y_pred), 4),
        f"{prefix}_rmse": round(float(np.sqrt(np.mean((y_true-y_pred)**2))), 5),
        f"{prefix}_da":   round(da(y_true, y_pred), 4),
    }

def diagnostics(y_true, y_pred, prefix):
    """Residual diagnostics from Peng Ding framework."""
    from scipy.stats import spearmanr
    resid   = y_true - y_pred
    sample  = resid[np.random.choice(len(resid), min(5000, len(resid)), replace=False)]
    _, sw_p = stats.shapiro(sample)
    het_r,_ = spearmanr(np.abs(resid), np.abs(y_pred))
    dw      = float(np.sum(np.diff(resid)**2) / np.sum(resid**2))
    return {
        f"{prefix}_normality_p": round(sw_p, 4),
        f"{prefix}_het_r":       round(het_r, 4),
        f"{prefix}_dw":          round(dw, 4),
    }


# ── Data helpers ───────────────────────────────────────────────────────

def downsample(df, h_sec):
    """Keep rows >= h_sec apart. Removes overlapping labels."""
    times  = df.index.astype("int64") / 1e3   # ms -> seconds
    sel    = []
    last_t = -np.inf
    for i, t in enumerate(times):
        if t - last_t >= h_sec:
            sel.append(i)
            last_t = t
    return df.iloc[sel]


def make_folds(df, n_folds):
    """
    Expanding-window walk-forward folds with purge+embargo gap.
    Last fold is the final test — never used for model selection.

    Structure:
    [==TRAIN==][GAP 75s][==VAL==] for each fold
    """
    times_s = df.index.astype("int64") / 1e3
    t_min   = times_s.min()
    t_max   = times_s.max()
    t_range = t_max - t_min

    # Val windows span the last 60% of data, divided into n_folds parts
    cuts = np.linspace(0.40, 1.00, n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end_t = t_min + t_range * cuts[i]
        val_start_t = train_end_t + GAP_S
        val_end_t   = t_min + t_range * cuts[i + 1]

        if val_start_t >= val_end_t:
            continue

        train_mask = times_s <= train_end_t
        val_mask   = (times_s >= val_start_t) & (times_s <= val_end_t)

        if train_mask.sum() < 500 or val_mask.sum() < 100:
            continue

        folds.append({
            "fold":       i + 1,
            "is_test":    (i == n_folds - 1),   # last fold = final test
            "train_mask": train_mask,
            "val_mask":   val_mask,
            "n_train":    int(train_mask.sum()),
            "n_val":      int(val_mask.sum()),
            "train_end":  pd.Timestamp(train_end_t, unit="s", tz="UTC").strftime("%m-%d %H:%M"),
            "val_start":  pd.Timestamp(val_start_t, unit="s", tz="UTC").strftime("%m-%d %H:%M"),
            "val_end":    pd.Timestamp(val_end_t,   unit="s", tz="UTC").strftime("%m-%d %H:%M"),
        })

    return folds


def get_Xy(df, feats, target):
    clean = df[feats + [target]].dropna()
    return (
        clean[feats].values.astype("float32"),
        clean[target].values.astype("float32"),
    )


# ── Linear models ──────────────────────────────────────────────────────

def train_ols(X_tr, y_tr, X_va, y_va, feats, label):
    """OLS with HAC (Newey-West) for autocorrelation in residuals."""
    X_sm    = sm.add_constant(X_tr)
    model   = sm.OLS(y_tr, X_sm).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    X_va_sm = sm.add_constant(X_va, has_constant="add")
    pred_tr = model.predict(X_sm)
    pred_va = model.predict(X_va_sm)
    coef    = pd.Series(model.params[1:], index=feats)
    top5    = ", ".join(coef.abs().sort_values(ascending=False).head(5).index.tolist())
    row = {f"ols_{label}_r2_train": round(r2_score(y_tr, pred_tr), 4), f"ols_{label}_top5": top5}
    row.update(metrics(y_va, pred_va, f"ols_{label}"))
    row.update(diagnostics(y_va, pred_va, f"ols_{label}"))
    logger.info("  OLS   [%s]  train_R2=%+.4f  val_R2=%+.4f  DA=%.3f", label,
                row[f"ols_{label}_r2_train"], row[f"ols_{label}_r2"], row[f"ols_{label}_da"])
    return row

def train_ridge(X_tr, y_tr, X_va, y_va, label):
    """Ridge with TimeSeriesSplit CV — no temporal leakage in lambda selection."""
    model   = RidgeCV(alphas=np.logspace(-4, 4, 30), cv=TimeSeriesSplit(n_splits=3))
    model.fit(X_tr, y_tr)
    pred_va = model.predict(X_va)
    row     = {f"ridge_{label}_alpha": round(float(model.alpha_), 4)}
    row.update(metrics(y_va, pred_va, f"ridge_{label}"))
    logger.info("  Ridge [%s]  alpha=%.4f  val_R2=%+.4f  DA=%.3f", label,
                model.alpha_, row[f"ridge_{label}_r2"], row[f"ridge_{label}_da"])
    return row

def train_lasso(X_tr, y_tr, X_va, y_va, label):
    """Lasso with TimeSeriesSplit CV — automatic feature selection."""
    model     = LassoCV(cv=TimeSeriesSplit(n_splits=3), max_iter=2000,
                        tol=1e-2, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred_va   = model.predict(X_va)
    n_nonzero = int((model.coef_ != 0).sum())
    row       = {f"lasso_{label}_alpha": round(float(model.alpha_), 6),
                 f"lasso_{label}_nonzero": n_nonzero}
    row.update(metrics(y_va, pred_va, f"lasso_{label}"))
    logger.info("  Lasso [%s]  alpha=%.6f  nonzero=%d  val_R2=%+.4f  DA=%.3f", label,
                model.alpha_, n_nonzero, row[f"lasso_{label}_r2"], row[f"lasso_{label}_da"])
    return row


# ── Tree models ────────────────────────────────────────────────────────

def train_dtree(X_tr, y_tr, X_va, y_va, feats):
    """Single decision tree — depth=3, interpretable, shows first split."""
    model   = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=42)
    model.fit(X_tr, y_tr)
    pred_va = model.predict(X_va)
    imp     = pd.Series(model.feature_importances_, index=feats)
    first   = feats[model.tree_.feature[0]]
    row     = {"dtree_first_split": first,
               "dtree_top5": ", ".join(imp.sort_values(ascending=False).head(5).index.tolist())}
    row.update(metrics(y_va, pred_va, "dtree"))
    logger.info("  DTree        val_R2=%+.4f  DA=%.3f  first=%s",
                row["dtree_r2"], row["dtree_da"], first)
    del model; gc.collect()
    return row

def train_rf(X_tr, y_tr, X_va, y_va, feats):
    """Random forest — 300 trees, depth=3."""
    model   = RandomForestRegressor(n_estimators=300, max_depth=3,
                                     min_samples_leaf=100, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    pred_va = model.predict(X_va)
    imp     = pd.Series(model.feature_importances_, index=feats)
    row     = {"rf_top5": ", ".join(imp.sort_values(ascending=False).head(5).index.tolist())}
    row.update(metrics(y_va, pred_va, "rf"))
    logger.info("  RF           val_R2=%+.4f  DA=%.3f", row["rf_r2"], row["rf_da"])
    del model; gc.collect()
    return row

def train_xgb(X_tr, y_tr, X_va, y_va, feats):
    """XGBoost — depth=3, conservative regularization, early stopping."""
    n_val   = int(len(X_tr) * 0.15)
    X_t, y_t   = X_tr[:-n_val], y_tr[:-n_val]
    X_v, y_v   = X_tr[-n_val:], y_tr[-n_val:]
    model   = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=100,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", n_jobs=-1, random_state=42,
        objective="reg:squarederror",
        early_stopping_rounds=20, eval_metric="rmse",
    )
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    imp     = pd.Series(model.feature_importances_, index=feats)
    row     = {
        "xgb_n_trees":  model.best_iteration + 1,
        "xgb_r2_train": round(r2_score(y_tr, pred_tr), 4),
        "xgb_top5":     ", ".join(imp.sort_values(ascending=False).head(5).index.tolist()),
    }
    row.update(metrics(y_va, pred_va, "xgb"))
    logger.info("  XGBoost      train_R2=%+.4f  val_R2=%+.4f  DA=%.3f  trees=%d",
                row["xgb_r2_train"], row["xgb_r2"], row["xgb_da"], row["xgb_n_trees"])
    del model; gc.collect()
    return row

def train_lgbm(X_tr, y_tr, X_va, y_va, feats):
    """LightGBM — depth=3, conservative regularization, early stopping."""
    n_val   = int(len(X_tr) * 0.15)
    X_t, y_t   = X_tr[:-n_val], y_tr[:-n_val]
    X_v, y_v   = X_tr[-n_val:], y_tr[-n_val:]
    model   = lgb.LGBMRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=100,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=42, objective="regression", verbose=-1,
    )
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
    pred_tr = model.predict(X_tr)
    pred_va = model.predict(X_va)
    imp     = pd.Series(model.feature_importances_, index=feats)
    row     = {
        "lgbm_n_trees":  model.best_iteration_,
        "lgbm_r2_train": round(r2_score(y_tr, pred_tr), 4),
        "lgbm_top5":     ", ".join(imp.sort_values(ascending=False).head(5).index.tolist()),
    }
    row.update(metrics(y_va, pred_va, "lgbm"))
    logger.info("  LightGBM     train_R2=%+.4f  val_R2=%+.4f  DA=%.3f  trees=%d",
                row["lgbm_r2_train"], row["lgbm_r2"], row["lgbm_da"], row["lgbm_n_trees"])
    del model; gc.collect()
    return row

def train_quantile_xgb(X_tr, y_tr, X_va, y_va):
    """
    Quantile XGBoost fan chart — 11 quantile models.
    Produces nested prediction intervals like Bollinger Bands:
      90% band: q5  → q95
      80% band: q10 → q90
      60% band: q20 → q80
      40% band: q30 → q70
      20% band: q40 → q60
      Median:   q50

    Calibration: each interval should contain its stated % of outcomes.
    Well-calibrated = coverage close to the interval label.
    """
    n_val = int(len(X_tr) * 0.15)
    X_t, y_t = X_tr[:-n_val], y_tr[:-n_val]
    X_v, y_v = X_tr[-n_val:], y_tr[-n_val:]
    preds = {}
    for q in QUANTILES:
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=100,
            tree_method="hist", n_jobs=-1, random_state=42,
            objective="reg:quantileerror", quantile_alpha=q,
            early_stopping_rounds=20, eval_metric="quantile",
        )
        m.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        preds[q] = m.predict(X_va)
        del m; gc.collect()

    row = {}
    # Pinball loss per quantile
    for q in QUANTILES:
        row[f"q{int(q*100):02d}_pinball"] = round(pinball(y_va, preds[q], q), 6)

    # Coverage for each nested interval
    for lo, hi, label in INTERVALS:
        cov = coverage(y_va, preds[lo], preds[hi])
        row[f"coverage_{label}"] = round(cov, 4)

    # Log all coverages
    cov_str = "  ".join(
        f"{label}={row[f'coverage_{label}']:.3f}"
        for _, _, label in INTERVALS
    )
    logger.info("  Quantile     %s", cov_str)
    return row


# ── Per-fold per-horizon ───────────────────────────────────────────────

def run_fold_horizon(df, fold, h_sec):
    target = f"fwd_change_{h_sec}s"
    train_df = df[fold["train_mask"]]
    val_df   = df[fold["val_mask"]]

    # Sparse: rows >= h_sec apart — removes overlapping labels
    cols_needed = list(set(ALL_FEATURES + FEATURES_VIF10 + [target]))
    cols_needed = [c for c in cols_needed if c in df.columns]

    train_sp = downsample(train_df[cols_needed].dropna(subset=[target]), h_sec)
    val_sp   = downsample(val_df[cols_needed].dropna(subset=[target]),   h_sec)

    if len(train_sp) < 50 or len(val_sp) < 20:
        logger.info("  h=%2ds  too few sparse rows (train=%d val=%d), skipping",
                    h_sec, len(train_sp), len(val_sp))
        return {"horizon_s": h_sec, "fold": fold["fold"], "skipped": True}

    logger.info("  h=%2ds  train_sp=%d  val_sp=%d", h_sec, len(train_sp), len(val_sp))
    row = {"horizon_s": h_sec, "fold": fold["fold"],
           "is_test": fold["is_test"], "skipped": False,
           "n_train_sparse": len(train_sp), "n_val_sparse": len(val_sp)}

    # Linear pipeline (VIF10 features, standardized)
    lin_feats = [f for f in FEATURES_VIF10 if f in df.columns]
    X_tr_l, y_tr = get_Xy(train_sp, lin_feats, target)
    X_va_l, y_va = get_Xy(val_sp,   lin_feats, target)

    sc      = RobustScaler(quantile_range=(5, 95))
    X_tr_ls = np.clip(sc.fit_transform(X_tr_l), -10, 10)
    X_va_ls = np.clip(sc.transform(X_va_l),     -10, 10)

    row.update(train_ols(  X_tr_ls, y_tr, X_va_ls, y_va, lin_feats, "vif10"))
    row.update(train_ridge(X_tr_ls, y_tr, X_va_ls, y_va, "vif10"))
    row.update(train_lasso(X_tr_ls, y_tr, X_va_ls, y_va, "vif10"))

    # Tree pipeline (all features, no scaling)
    X_tr_t, y_tr_t = get_Xy(train_sp, ALL_FEATURES, target)
    X_va_t, y_va_t = get_Xy(val_sp,   ALL_FEATURES, target)

    X_tr_t = X_tr_t.astype("float32"); y_tr_t = y_tr_t.astype("float32")
    X_va_t = X_va_t.astype("float32"); y_va_t = y_va_t.astype("float32")

    row.update(train_dtree(X_tr_t, y_tr_t, X_va_t, y_va_t, ALL_FEATURES))
    row.update(train_rf(   X_tr_t, y_tr_t, X_va_t, y_va_t, ALL_FEATURES))
    row.update(train_xgb(  X_tr_t, y_tr_t, X_va_t, y_va_t, ALL_FEATURES))
    row.update(train_lgbm( X_tr_t, y_tr_t, X_va_t, y_va_t, ALL_FEATURES))
    row.update(train_quantile_xgb(X_tr_t, y_tr_t, X_va_t, y_va_t))

    # Winner by val R2
    r2s    = {k: v for k, v in row.items() if k.endswith("_r2") and "train" not in k}
    winner = max(r2s, key=r2s.get)
    row["winner"]    = winner.replace("_r2", "")
    row["winner_r2"] = r2s[winner]
    logger.info("  Winner: %s  (R2=%.4f)", row["winner"], row["winner_r2"])
    return row


# ── Main ───────────────────────────────────────────────────────────────

def main():
    logger.info("=== Training pipeline: sparse + purged walk-forward ===")
    logger.info("Gap: %ds (purge %ds + embargo %ds)  Folds: %d (last=test)",
                GAP_S, PURGE_S, EMBARGO_S, N_FOLDS)

    # Load features, compute targets
    logger.info("Loading features ...")
    df = pd.read_parquet("data/features/BTC_features.parquet")
    logger.info("Computing targets ...")
    df = build_targets(df)
    logger.info("Dataset: %d rows, %d columns", len(df), df.shape[1])

    # Build folds
    folds = make_folds(df, N_FOLDS)
    logger.info("Folds created:")
    for f in folds:
        marker = " <- FINAL TEST" if f["is_test"] else ""
        logger.info("  Fold %d: train->%s | 75s gap | val %s->%s  (n_train=%d  n_val=%d)%s",
                    f["fold"], f["train_end"], f["val_start"], f["val_end"],
                    f["n_train"], f["n_val"], marker)

    # Train
    results = []
    for fold in folds:
        fold_type = "FINAL TEST" if fold["is_test"] else f"Fold {fold['fold']}"
        logger.info("\n========== %s ==========", fold_type)
        for h in HORIZONS_SEC:
            row = run_fold_horizon(df, fold, h)
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "BTC_results.csv"), index=False)

    # Summary: CV folds (model selection) vs final test
    cv_df   = results_df[~results_df["is_test"] & ~results_df.get("skipped", False)]
    test_df = results_df[ results_df["is_test"] & ~results_df.get("skipped", False)]

    logger.info("\n=== CV SUMMARY (mean across folds, model selection) ===")
    r2_cols = [c for c in results_df.columns if c.endswith("_r2") and "train" not in c and c != "winner_r2"]
    for h in HORIZONS_SEC:
        sub = cv_df[cv_df["horizon_s"] == h]
        if sub.empty: continue
        logger.info("  h=%2ds  winner=%s  XGB=%.4f  LGBM=%.4f  Ridge=%.4f  OLS=%.4f",
                    h,
                    sub["winner"].mode()[0] if not sub.empty else "?",
                    sub["xgb_r2"].mean()          if "xgb_r2"          in sub else np.nan,
                    sub["lgbm_r2"].mean()          if "lgbm_r2"         in sub else np.nan,
                    sub["ridge_vif10_r2"].mean()   if "ridge_vif10_r2"  in sub else np.nan,
                    sub["ols_vif10_r2"].mean()     if "ols_vif10_r2"    in sub else np.nan)

    logger.info("\n=== FINAL TEST RESULTS (unseen data) ===")
    for h in HORIZONS_SEC:
        sub = test_df[test_df["horizon_s"] == h]
        if sub.empty: continue
        logger.info("  h=%2ds  winner=%s(%.4f)  XGB=%.4f  LGBM=%.4f  Ridge=%.4f  coverage=%.3f",
                    h,
                    sub["winner"].values[0], sub["winner_r2"].values[0],
                    sub["xgb_r2"].values[0]          if "xgb_r2"          in sub else np.nan,
                    sub["lgbm_r2"].values[0]          if "lgbm_r2"         in sub else np.nan,
                    sub["ridge_vif10_r2"].values[0]   if "ridge_vif10_r2"  in sub else np.nan,
                    sub["coverage_80pct"].values[0] if "coverage_80pct" in sub else np.nan)

    logger.info("\nFull results: data/results/BTC_results.csv")


if __name__ == "__main__":
    main()