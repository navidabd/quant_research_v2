"""
inspect_models.py — Print features, coefficients and importances
for every saved model in data/models/

Run: python inspect_models.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MODEL_DIR = "data/models"

FEATURES_VIF10 = [
    "vwap_dev_1s", "vwap_dev_5s", "vwap_dev_10s", "vwap_dev_30s",
    "trade_imbalance_1s", "trade_imbalance_5s",
    "trade_imbalance_10s", "trade_imbalance_30s",
    "book_imbalance_l5", "ofi_l1",
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
    "ofi_l1", "momentum_2s", "momentum_5s", "momentum_10s",
]

HORIZONS = [1, 3, 5, 10, 15]

def bar(val, max_val, width=30):
    filled = int(abs(val) / max_val * width) if max_val > 0 else 0
    sign   = "+" if val >= 0 else "-"
    return sign + "█" * filled

def inspect_linear(name, path, features):
    if not os.path.exists(path):
        print(f"  [not found: {path}]")
        return
    model = joblib.load(path)
    print(f"\n  ── {name} ──")
    # statsmodels OLS
    if hasattr(model, "params"):
        coefs = pd.Series(model.params[1:], index=features)
        pvals = pd.Series(model.pvalues[1:], index=features)
        max_abs = coefs.abs().max()
        for feat in coefs.abs().sort_values(ascending=False).index:
            c = coefs[feat]; p = pvals[feat]
            sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            print(f"    {feat:<35} {c:>+10.6f}  {sig}  {bar(c, max_abs)}")
    # sklearn Ridge / Lasso
    elif hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_, index=features)
        max_abs = coefs.abs().max()
        for feat in coefs.abs().sort_values(ascending=False).index:
            c = coefs[feat]
            print(f"    {feat:<35} {c:>+10.6f}  {bar(c, max_abs)}")

def inspect_tree(name, path, features):
    if not os.path.exists(path):
        print(f"  [not found: {path}]")
        return
    model = joblib.load(path)
    print(f"\n  ── {name} ──")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features)
    elif hasattr(model, "feature_importance"):
        imp = pd.Series(model.feature_importance(), index=features)
    else:
        print("    [no feature_importances_ attribute]")
        return
    imp = imp.sort_values(ascending=False)
    max_imp = imp.max()
    for feat, val in imp.items():
        if val > 0.001:
            print(f"    {feat:<35} {val:>8.4f}  {bar(val, max_imp)}")

def main():
    for h in HORIZONS:
        tag = f"h{h}s"
        print("\n" + "=" * 70)
        print(f"  HORIZON: {h}s  (target: fwd_change_{h}s)")
        print("=" * 70)
        print(f"\n  Features for linear models (VIF10, {len(FEATURES_VIF10)} features):")
        print(f"  Features for tree   models (ALL,   {len(ALL_FEATURES)} features):")

        scaler_path = f"{MODEL_DIR}/scaler_vif10_{tag}.pkl"
        if os.path.exists(scaler_path):
            sc = joblib.load(scaler_path)
            print(f"\n  Scaler (RobustScaler) center/scale on VIF10 features:")
            for i, feat in enumerate(FEATURES_VIF10):
                center = sc.center_[i] if hasattr(sc, 'center_') else sc.scale_[i]
                scale  = sc.scale_[i]
                print(f"    {feat:<35} center={center:>+10.4f}  scale={scale:>10.4f}")

        print(f"\n  LINEAR MODELS (coefficients — scaled space):")
        inspect_linear("OLS (HAC)",
                       f"{MODEL_DIR}/ols_vif10_{tag}.pkl", FEATURES_VIF10)
        inspect_linear("Ridge",
                       f"{MODEL_DIR}/ridge_vif10_{tag}.pkl", FEATURES_VIF10)
        inspect_linear("Lasso",
                       f"{MODEL_DIR}/lasso_vif10_{tag}.pkl", FEATURES_VIF10)

        print(f"\n  TREE MODELS (feature importances — top features only):")
        inspect_tree("Decision Tree", f"{MODEL_DIR}/dtree_{tag}.pkl", ALL_FEATURES)
        inspect_tree("Random Forest", f"{MODEL_DIR}/rf_{tag}.pkl",    ALL_FEATURES)
        inspect_tree("XGBoost",       f"{MODEL_DIR}/xgb_{tag}.pkl",   ALL_FEATURES)
        inspect_tree("LightGBM",      f"{MODEL_DIR}/lgbm_{tag}.pkl",  ALL_FEATURES)

    print("\n")

if __name__ == "__main__":
    main()