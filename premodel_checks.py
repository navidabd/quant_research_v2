"""
Pre-modeling checks — clean version.

Runs on TRAIN SET ONLY. Test set is never touched here.

Steps:
  1. Load train set
  2. Spearman correlation: every feature vs every target horizon
  3. VIF: measures multicollinearity between features
  4. Per-horizon feature selection:
       keep features passing BOTH:
         - VIF <= threshold (not collinear)
         - |Spearman r| > 0.05 with that horizon's target

Three VIF thresholds tested: 5, 10, 15

Manual overrides (agreed before coding):
  - Drop weighted_mid_dev if present (same info as book_imbalance_l1)
  - Drop rel_spread if present (collinear with spread)
  - Force-keep book_imbalance_l5 as depth cluster representative

Outputs saved to data/premodel_v2/

Run:
    python premodel_checks_v2.py
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

OUT_DIR        = "data/premodel_v2"
CORR_THRESHOLD = 0.05
HORIZONS_SEC   = [1, 3, 5, 10, 15]
os.makedirs(OUT_DIR, exist_ok=True)

# All candidate features — must match engineer_v2.py output
CANDIDATE_FEATURES = [
    # VWAP deviation (trade tape)
    "vwap_dev_1s", "vwap_dev_5s", "vwap_dev_10s", "vwap_dev_30s",
    # Trade imbalance (trade tape)
    "trade_imbalance_1s", "trade_imbalance_5s",
    "trade_imbalance_10s", "trade_imbalance_30s",
    # Trade volume (trade tape)
    "trade_volume_5s", "trade_volume_30s", "trade_volume_60s",
    "trade_volume_180s", "trade_volume_300s",
    # Trade count (trade tape)
    "trade_count_5s", "trade_count_30s", "trade_count_60s",
    "trade_count_180s", "trade_count_300s",
    # Realized volatility (trade tape)
    "realized_vol_10s", "realized_vol_30s", "realized_vol_60s",
    "realized_vol_180s", "realized_vol_300s",
    # Book imbalance (book tape)
    "book_imbalance_l1", "book_imbalance_l5", "book_imbalance_l10",
    "book_imbalance_l15", "book_imbalance_l20",
    # Order flow imbalance L1 (book tape)
    "ofi_l1",
    # Momentum (book tape — midprice log returns)
    "momentum_2s", "momentum_5s", "momentum_10s",
]

TARGET_COLS  = [f"fwd_change_{h}s" for h in HORIZONS_SEC]
FORCE_DROP   = {"weighted_mid_dev", "rel_spread"}
FORCE_KEEP   = {"book_imbalance_l5"}


def load_train() -> pd.DataFrame:
    path = "data/processed_v2/BTC_train.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Train file not found: {path}\n"
            "Run build_dataset_v2.py first."
        )
    df = pd.read_parquet(path)
    logger.info("Train set: %d rows, %d columns", len(df), df.shape[1])

    missing_f = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    missing_t = [c for c in TARGET_COLS if c not in df.columns]
    if missing_f:
        raise ValueError(f"Missing features: {missing_f}")
    if missing_t:
        raise ValueError(f"Missing targets: {missing_t}")

    return df


def correlation_check(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing Spearman correlations (train set only) ...")
    rows = []

    for feat in CANDIDATE_FEATURES:
        row = {"feature": feat}
        for tgt in TARGET_COLS:
            # Drop rows where either feature or target is NaN
            clean = df[[feat, tgt]].dropna()
            if len(clean) < 1000:
                row[tgt] = np.nan
                continue
            r, _ = spearmanr(clean[feat], clean[tgt])
            row[tgt] = round(r, 5)
        rows.append(row)

    corr_df = pd.DataFrame(rows).set_index("feature")
    corr_df.to_csv(os.path.join(OUT_DIR, "BTC_correlations.csv"))

    logger.info("\n--- Top 10 features per horizon (|Spearman r|) ---")
    for h in HORIZONS_SEC:
        tgt    = f"fwd_change_{h}s"
        ranked = corr_df[tgt].abs().sort_values(ascending=False).head(10)
        logger.info("\n  Horizon %2ds:", h)
        for feat, _ in ranked.items():
            logger.info(
                "    %-35s  r = %+.4f", feat, corr_df.loc[feat, tgt]
            )

    return corr_df


def vif_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIF (Variance Inflation Factor) measures multicollinearity.
    VIF = 1/(1 - R^2) where R^2 comes from regressing that feature
    on all other features.
    VIF = 1   : no collinearity
    VIF = 5   : moderate (strict threshold)
    VIF = 10  : high (moderate threshold)
    VIF = 15  : very high (permissive threshold)
    inf       : perfect collinearity — must drop

    Sample 100K rows for speed — VIF is stable with this many rows.
    """
    logger.info("Computing VIF (sample 100K rows from train) ...")

    sample = df[CANDIDATE_FEATURES].dropna()
    if len(sample) > 100_000:
        # Take first 100K rows (chronological) — preserves time order
        sample = sample.iloc[:100_000]

    X = sample.values.astype("float32")

    vif_scores = []
    for i, feat in enumerate(CANDIDATE_FEATURES):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.nan
        vif_scores.append({"feature": feat, "vif": round(v, 2)})
        if (i + 1) % 10 == 0:
            logger.info("  VIF progress: %d/%d", i + 1, len(CANDIDATE_FEATURES))

    vif_df = pd.DataFrame(vif_scores).set_index("feature").sort_values(
        "vif", ascending=False
    )
    vif_df.to_csv(os.path.join(OUT_DIR, "BTC_vif_scores.csv"))
    logger.info("\n--- VIF scores ---\n%s", vif_df.to_string())
    return vif_df


def per_horizon_selection(corr_df: pd.DataFrame,
                           vif_df: pd.DataFrame) -> None:
    logger.info(
        "\n--- Per-horizon feature selection "
        "(VIF threshold + |r| > %.2f) ---", CORR_THRESHOLD
    )

    # Per-horizon book imbalance representative (agreed before coding):
    #   1s/3s/5s  -> L5  (strongest signal at short horizons)
    #   10s/15s   -> L10 (stronger signal at longer horizons)
    # All other depth levels dropped from linear models.
    # Trees will use all depths and find this pattern naturally.
    DEPTH_KEEP = {
        1:  "book_imbalance_l5",
        3:  "book_imbalance_l5",
        5:  "book_imbalance_l5",
        10: "book_imbalance_l10",
        15: "book_imbalance_l10",
    }
    ALL_DEPTH_COLS = {f"book_imbalance_l{d}" for d in [1, 5, 10, 15, 20]}

    for threshold, label in [(5, "vif5"), (10, "vif10"), (15, "vif15")]:
        vif_pass = set(vif_df[vif_df["vif"] <= threshold].index.tolist())
        vif_pass = vif_pass - FORCE_DROP

        for h in HORIZONS_SEC:
            tgt = f"fwd_change_{h}s"

            depth_keep = DEPTH_KEEP[h]

            # Remove all depth cols, add back only the chosen representative
            h_vif_pass = (vif_pass - ALL_DEPTH_COLS) | {depth_keep}

            corr_pass = set(
                corr_df[tgt][corr_df[tgt].abs() > CORR_THRESHOLD].index.tolist()
            )

            # depth_keep bypasses correlation filter — it's a deliberate choice
            final     = (h_vif_pass & corr_pass) | {depth_keep}
            final_ord = [f for f in CANDIDATE_FEATURES if f in final]
            dropped_c = h_vif_pass - corr_pass - {depth_keep}

            out = os.path.join(OUT_DIR, f"BTC_features_{h}s_{label}.txt")
            with open(out, "w") as f:
                f.write("\n".join(final_ord))

            logger.info(
                "\n  Horizon %2ds [%s]: keep %d  depth_rep=%s  "
                "dropped_by_corr=%d: %s",
                h, label, len(final_ord), depth_keep,
                len(dropped_c),
                sorted(dropped_c) if dropped_c else "none"
            )
            logger.info("  Kept: %s", final_ord)


def main():
    logger.info("=== Pre-modeling checks (train set only) ===")
    df      = load_train()
    corr_df = correlation_check(df)
    vif_df  = vif_check(df)
    per_horizon_selection(corr_df, vif_df)
    logger.info("\n=== Done. Review results then proceed to modeling ===")


if __name__ == "__main__":
    main()