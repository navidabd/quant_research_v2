"""
Feature engineering — BTC only, union grid (one row per book update OR trade).

Book features computed on book_df, trade features on trade_df,
then merged onto union grid via merge_asof backward.
No carry-forward contamination in rolling windows.

Windows (agreed, 300s and 180s removed — not useful, high VIF):
  momentum:        2s, 5s, 10s
  realized_vol:    10s, 30s, 60s
  trade_imbalance: 1s, 5s, 10s, 30s
  vwap_dev:        1s, 5s, 10s, 30s
  trade_volume:    5s, 30s, 60s
  trade_count:     5s, 30s, 60s
  book_imbalance:  L1, L5, L10, L15, L20
"""

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = ["2s", "5s", "10s"]
VOL_WINDOWS      = ["10s", "30s", "60s"]
FLOW_WINDOWS     = ["1s", "5s", "10s", "30s"]
VOLUME_WINDOWS   = ["5s", "30s", "60s"]
IMBALANCE_DEPTHS = [1, 5, 10, 15, 20]


def compute_book_features(book_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features computed on book snapshots only.
    mid, book_imbalance at each depth, ofi_l1, momentum.
    """
    df = book_df.copy()

    # Fair price
    df["mid"] = (df["bid_px_1"] + df["ask_px_1"]) / 2

    # Book imbalance at each depth (size only, not notional)
    # +1 = all size on bid (buying pressure), -1 = all on ask
    for d in IMBALANCE_DEPTHS:
        bid = sum(df[f"bid_sz_{i}"] for i in range(1, d + 1))
        ask = sum(df[f"ask_sz_{i}"] for i in range(1, d + 1))
        df[f"book_imbalance_l{d}"] = (bid - ask) / (bid + ask)

    # Order flow imbalance L1 (Cont-Kukanov-Stoikov)
    # Tracks what changed at the touch between consecutive snapshots
    bid_up = df["bid_px_1"] > df["bid_px_1"].shift(1)
    bid_dn = df["bid_px_1"] < df["bid_px_1"].shift(1)
    bid_term = np.where(bid_up,  df["bid_sz_1"],
               np.where(bid_dn, -df["bid_sz_1"].shift(1),
                        df["bid_sz_1"] - df["bid_sz_1"].shift(1)))

    ask_up = df["ask_px_1"] > df["ask_px_1"].shift(1)
    ask_dn = df["ask_px_1"] < df["ask_px_1"].shift(1)
    ask_term = np.where(ask_dn,  df["ask_sz_1"],
               np.where(ask_up, -df["ask_sz_1"].shift(1),
                        df["ask_sz_1"] - df["ask_sz_1"].shift(1)))

    # Positive = net buying at touch, negative = net selling
    df["ofi_l1"] = bid_term - ask_term

    # Momentum: cumulative log return of mid over rolling window
    # Captures short-term price drift direction
    df["log_ret_mid"] = np.log(df["mid"]).diff()
    for w in MOMENTUM_WINDOWS:
        df[f"momentum_{w}"] = df["log_ret_mid"].rolling(w, min_periods=1).sum()

    return df


def compute_trade_features(trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features computed on trade tape only.
    Rolling windows here count only actual trades — no contamination
    from carried-forward book rows since we compute on trade_df directly.
    """
    t = trade_df.copy()
    t["signed_size"] = np.where(t["side"] == "B", t["size"], -t["size"])
    t["notional"]    = t["price"] * t["size"]
    t["log_ret"]     = np.log(t["price"]).diff()

    out = pd.DataFrame(index=t.index)

    for w in FLOW_WINDOWS:
        vol      = t["size"].rolling(w).sum()
        sig_vol  = t["signed_size"].rolling(w).sum()
        notional = t["notional"].rolling(w).sum()
        vwap     = notional / vol

        # How far current trade price is from rolling VWAP
        # Positive = trading above average = buying pressure
        out[f"vwap_dev_{w}"]        = (t["price"] - vwap) / t["price"]

        # Net buy volume ratio: +1=all buys, -1=all sells
        out[f"trade_imbalance_{w}"] = sig_vol / vol

    for w in VOLUME_WINDOWS:
        # Total BTC traded — market activity proxy
        out[f"trade_volume_{w}"] = t["size"].rolling(w).sum()

        # Number of trades — order frequency
        out[f"trade_count_{w}"]  = t["size"].rolling(w).count()

    for w in VOL_WINDOWS:
        # Realized volatility on trade tape — dense enough for std()
        # min_periods=5: need at least 5 trades for meaningful estimate
        out[f"realized_vol_{w}"] = t["log_ret"].rolling(w, min_periods=5).std()

    return out


def build_features(book_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Compute book features on book_df
    2. Compute trade features on trade_df
    3. Union grid: one row per book update OR trade event
    4. Merge book features backward (carry last known book state to trade rows)
    5. Merge trade features backward (carry last trade window to book rows)
    """
    book_feats  = compute_book_features(book_df)
    trade_feats = compute_trade_features(trade_df)

    union_idx = book_df.index.union(trade_df.index).sort_values()
    grid = pd.DataFrame(index=union_idx)
    grid["event_type"] = np.where(union_idx.isin(book_df.index), "book", "trade")

    grid = pd.merge_asof(grid, book_feats,  left_index=True, right_index=True, direction="backward")
    grid = pd.merge_asof(grid, trade_feats, left_index=True, right_index=True, direction="backward")
    # Raw trade columns for fill simulation
    trade_positions = np.where(grid["event_type"].values == "trade")[0]
    n = len(trade_positions)
    
    price_arr = np.full(len(grid), np.nan)
    size_arr  = np.full(len(grid), np.nan)
    price_arr[trade_positions] = trade_df["price"].values[:n]
    size_arr[trade_positions]  = trade_df["size"].values[:n]
    
    grid["trade_price"] = price_arr
    grid["trade_size"]  = size_arr
    return grid