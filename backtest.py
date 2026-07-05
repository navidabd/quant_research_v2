"""
backtest.py — Passive maker strategy, flat regime only, BTC / Hyperliquid

═══════════════════════════════════════════════════════════════
STRATEGY ASSUMPTIONS
═══════════════════════════════════════════════════════════════

MODEL:
  LightGBM h5s — predicts 5s price change
  Loaded from data/models/lgbm_h5s.pkl

REGIME DETECTION:
  momentum_10s × mid = approximate dollar move over last 10s
  |dollar_move| < FLAT_THRESHOLD ($20) → FLAT → place orders
  |dollar_move| >= FLAT_THRESHOLD     → TRENDING → do nothing
  (existing open positions keep their TP/SL regardless of regime)

SIGNAL FILTER:
  Even in flat regime: |pred| must exceed SIGNAL_THRESHOLD
  to confirm there is some microstructure edge

ENTRY (flat regime only, maker, zero fee):
  Bid at floor(fair_value - ENTRY_OFFSET)
  Ask at ceil(fair_value + ENTRY_OFFSET)
  fair_value = mid + pred
  1 active entry order per side at a time
  Max MAX_POS_PER_SIDE open positions per side
  Engine auto-cancels and reprices entry on each new signal

SIZE:
  ORDER_SIZE_BTC = 0.2 BTC (reduced to avoid hard stop)
  No half/full split in this strategy version

EXIT:
  TP: resting limit at entry ± TP_OFFSET (maker, 0 fee)
  SL: market close when mid crosses entry ± SL_OFFSET (taker fee)

RISK:
  Max daily loss: 3% of capital
  Max drawdown:   10% of capital
  Max positions per side: 3
═══════════════════════════════════════════════════════════════
"""

import logging
import math
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from execution import BacktestEngine

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  ASSUMPTIONS
# ═══════════════════════════════════════════════════════════════════

MODEL_PATH     = "data/models/lgbm_h5s.pkl"

SIGNAL_THRESHOLD  = 1.0    # |pred| must exceed this to place orders  [tune]
FLAT_THRESHOLD    = 10     # |momentum_10s × mid| < $20 → flat regime [tune]

ENTRY_OFFSET      = 10     # $ from fair value for entry orders        [tune]
TP_OFFSET         = 5     # ticks profit target from entry price      [tune]
SL_OFFSET         = 5     # ticks stop loss from entry price          [tune]

ORDER_SIZE_BTC    = 0.01    # BTC per order                             [tune]
MAX_POS_PER_SIDE  = 3      # max simultaneous longs or shorts

CAPITAL           = 10_000
MAX_DAILY_LOSS    = 0.03
MAX_DRAWDOWN      = 0.10

FEAT_PATH  = "data/features/BTC_features.parquet"
TEST_START = pd.Timestamp("2026-06-21 08:10:15", tz="UTC")

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

# ═══════════════════════════════════════════════════════════════════
#  LOAD + PREDICT
# ═══════════════════════════════════════════════════════════════════

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    log.info(f"Loaded: {MODEL_PATH}")
    return model

def get_predictions(df: pd.DataFrame, model) -> np.ndarray:
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X     = df[ALL_FEATURES].values.astype(float)
    preds = np.full(len(df), np.nan)
    valid = ~np.any(np.isnan(X), axis=1)
    if valid.sum() > 0:
        preds[valid] = model.predict(X[valid])
    log.info(f"Predictions: {valid.sum():,} valid / {len(df):,} rows")
    return preds

# ═══════════════════════════════════════════════════════════════════
#  STRATEGY
# ═══════════════════════════════════════════════════════════════════

def make_strategy(predictions: np.ndarray,
                  mom10_arr: np.ndarray,
                  mid_arr: np.ndarray):
    """
    Returns strategy function closed over pre-computed arrays.
    Index i corresponds to df_test row i.
    """
    def strategy(state: dict, open_positions: list) -> list:
        ts      = state["ts"]
        mid     = state["mid"]
        tob_bid = state["tob_bid"]
        tob_ask = state["tob_ask"]

        # We need row index — use ts to find it
        # (passed via state["_i"] set in wrapper below)
        i = state.get("_i")
        if i is None or i >= len(predictions):
            return []

        pred = predictions[i]
        if np.isnan(pred):
            return []

        # ── Regime check ─────────────────────────────────────────
        mom10 = mom10_arr[i]
        if np.isnan(mom10):
            return []
        dollar_move = abs(float(mom10) * mid)
        if dollar_move >= FLAT_THRESHOLD:
            return []   # trending — no new orders

        # ── Signal filter ─────────────────────────────────────────
        if abs(pred) < SIGNAL_THRESHOLD:
            return []

        fv = mid + pred

        # ── Count open positions ──────────────────────────────────
        longs  = sum(1 for p in open_positions if p.side == "long")
        shorts = sum(1 for p in open_positions if p.side == "short")

        intents = []

        # ── Bid ───────────────────────────────────────────────────
        if longs < MAX_POS_PER_SIDE:
            bid_price = math.floor(fv - ENTRY_OFFSET)
            bid_price = min(bid_price, tob_bid)   # never cross TOB
            if bid_price > 0:
                intents.append({
                    "action":     "place",
                    "side":       "bid",
                    "order_type": "limit",
                    "price":      bid_price,
                    "size":       ORDER_SIZE_BTC,
                    "tp_offset":  TP_OFFSET,
                    "sl_offset":  SL_OFFSET,
                })

        # ── Ask ───────────────────────────────────────────────────
        if shorts < MAX_POS_PER_SIDE:
            ask_price = math.ceil(fv + ENTRY_OFFSET)
            ask_price = max(ask_price, tob_ask)   # never cross TOB
            intents.append({
                "action":     "place",
                "side":       "ask",
                "order_type": "limit",
                "price":      ask_price,
                "size":       ORDER_SIZE_BTC,
                "tp_offset":  TP_OFFSET,
                "sl_offset":  SL_OFFSET,
            })

        return intents

    return strategy

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info(f"Loading {FEAT_PATH}")
    df = pd.read_parquet(FEAT_PATH)
    log.info(f"Dataset: {len(df):,} rows × {df.shape[1]} cols")

    df_test = df[df.index >= TEST_START].copy()
    log.info(f"Test rows: {len(df_test):,}  "
             f"({df_test.index[0]} → {df_test.index[-1]})")

    model       = load_model()
    predictions = get_predictions(df_test, model)
    mom10_arr   = df_test["momentum_10s"].values.astype(float)
    mid_arr     = df_test["mid"].values.astype(float)

    # Wrap strategy to inject row index via state["_i"]
    raw_strategy = make_strategy(predictions, mom10_arr, mid_arr)
    ts_arr       = np.array([t.timestamp() for t in df_test.index])
    ts_to_i      = {ts: i for i, ts in enumerate(ts_arr)}

    def strategy_with_index(state, open_positions):
        state["_i"] = ts_to_i.get(state["ts"])
        return raw_strategy(state, open_positions)

    engine = BacktestEngine(
        df=df_test,
        capital=CAPITAL,
        fee_maker=0.0,
        fee_taker=0.00045,
    )

    np.random.seed(42)
    engine.run(
        predictions=predictions,
        strategy_fn=strategy_with_index,
        max_daily_loss_pct=MAX_DAILY_LOSS,
        max_drawdown_pct=MAX_DRAWDOWN,
    )

    engine.print_report()

    os.makedirs("data/results", exist_ok=True)
    pd.DataFrame([engine.compute_metrics()]).to_csv(
        "data/results/backtest_maker_metrics.csv", index=False
    )
    log.info("Saved → data/results/backtest_maker_metrics.csv")

if __name__ == "__main__":
    main()