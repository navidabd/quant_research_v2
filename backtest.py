"""
backtest.py — Passive MM backtest v2.0  BTC / Hyperliquid
Fast version: numpy arrays, no iterrows, no L1-20 scanning.

Entry  : bid/ask at FV ± ENTRY_OFFSET ($10)
         1 entry order per side, max 3 positions per side
Size   : BASE_SIZE if flat/with-trend, HALF_SIZE if against trend
TP     : fixed at mid at fill time (never adjusted)
SL     : $50 MTM loss per position
Fee    : zero (maker, base tier)
"""

import math, logging, warnings, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  ASSUMPTIONS
# ═══════════════════════════════════════════════════════════════════

LATENCY_BASE_MS    = 450
LATENCY_MAX_MS     = 1200
LATENCY_VOL_THRESH = 0.0002
LATENCY_JITTER     = 0.10

MODEL_HORIZON      = "5s"
ENTRY_OFFSET       = 10        # $ from FV for entry orders
REPRICE_THRESHOLD  = 2         # ticks before cancel+reprice

BASE_SIZE_BTC      = 0.5       # 0.01 BTC × 50 leverage
HALF_SIZE_BTC      = 0.25      # against-trend
FLAT_THRESHOLD     = 20        # |momentum_10s × mid| < $20 → FLAT
MAX_POS_PER_SIDE   = 3
SL_PRICE_DIST      = 10       # $10 price move against entry → stop loss (same as TP distance)

CAPITAL            = 10_000
MAX_DAILY_LOSS_PCT = 0.03
MAX_DRAWDOWN_PCT   = 0.10
MAKER_FEE          = 0.0

FEATURES = [
    "vwap_dev_1s", "vwap_dev_5s", "vwap_dev_10s", "vwap_dev_30s",
    "trade_imbalance_1s", "trade_imbalance_5s",
    "trade_imbalance_10s", "trade_imbalance_30s",
    "book_imbalance_l5", "ofi_l1",
    "momentum_2s", "momentum_5s", "momentum_10s",
]
TARGET     = f"fwd_change_{MODEL_HORIZON}"
TRAIN_END  = pd.Timestamp("2026-06-21 08:08:00", tz="UTC")
TEST_START = pd.Timestamp("2026-06-21 08:10:15", tz="UTC")
FEAT_PATH  = "data/features/BTC_features.parquet"
OUT_PATH   = "data/results/backtest_v2.csv"

# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

_oid = 0
def next_id():
    global _oid; _oid += 1; return _oid

def sim_lat(vol10: float) -> float:
    v   = min(1.0, vol10 / max(LATENCY_VOL_THRESH, 1e-9))
    ms  = LATENCY_BASE_MS + (LATENCY_MAX_MS - LATENCY_BASE_MS) * (1 - (1-v)**2)
    return max(LATENCY_BASE_MS, ms + np.random.normal(0, ms * LATENCY_JITTER)) / 1000.0

def compute_target(df: pd.DataFrame, h: int = 5) -> pd.Series:
    fwd = df.index.searchsorted(df.index + pd.Timedelta(seconds=h))
    fwd = np.clip(fwd, 0, len(df) - 1)
    dt  = (df.index[fwd] - df.index).total_seconds()
    chg = df["mid"].values[fwd] - df["mid"].values
    return pd.Series(np.where(np.abs(dt - h) <= 2, chg, np.nan),
                     index=df.index, name=TARGET)

# ═══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES (plain dicts for speed)
# ═══════════════════════════════════════════════════════════════════
# Order dict keys:
#   id, type ('entry'|'tp'), pos_id, side ('bid'|'ask'), price,
#   size, sub_ts, arr_ts, status ('pending'|'live'|'filled'|'cancelled'|'cancel_pending'),
#   fill_type, can_arr_ts
#
# Position dict keys:
#   id, dir ('long'|'short'), entry_px, size, entry_ts, tp_px,
#   tp_ord_id, status ('open'|'closed'), close_px, close_ts,
#   close_reason, pnl

def make_order(otype, pos_id, side, price, size, ts, lat):
    return dict(id=next_id(), type=otype, pos_id=pos_id,
                side=side, price=price, size=size,
                sub_ts=ts, arr_ts=ts+lat,
                status="pending", fill_type="", can_arr_ts=None)

def make_position(direction, entry_px, size, ts, tp_px):
    return dict(id=next_id(), dir=direction, entry_px=entry_px,
                size=size, entry_ts=ts, tp_px=tp_px,
                tp_ord_id=None, sl_ord_id=None,
                status="open", close_px=None, close_ts=None,
                close_reason="", pnl=0.0)

def pos_mtm(p, mid):
    if p["dir"] == "long":
        return (mid - p["entry_px"]) * p["size"]
    return (p["entry_px"] - mid) * p["size"]

# ═══════════════════════════════════════════════════════════════════
#  BACKTEST LOOP
# ═══════════════════════════════════════════════════════════════════

def run_backtest(df_train: pd.DataFrame, df_test: pd.DataFrame):

    # ── Train ────────────────────────────────────────────────────
    log.info(f"Training on {len(df_train):,} rows ...")
    clean  = df_train[FEATURES + [TARGET]].dropna()
    scaler = RobustScaler(quantile_range=(5, 95))
    X_sc   = np.clip(scaler.fit_transform(clean[FEATURES].values), -10, 10)
    y_tr   = clean[TARGET].values
    model  = RidgeCV(alphas=np.logspace(-4, 4, 30), cv=3)
    model.fit(X_sc, y_tr)
    log.info(f"  alpha={model.alpha_:.4f}  train_R2={model.score(X_sc, y_tr):.3f}")

    # ── Pre-extract test arrays for speed ────────────────────────
    ts_arr  = np.array([t.timestamp() for t in df_test.index])
    mid_arr = df_test["mid"].values.astype(float)
    vol_arr = df_test["realized_vol_10s"].values.astype(float)
    mom_arr = df_test["momentum_10s"].values.astype(float)
    bid1_arr= df_test["bid_px_1"].values.astype(float)
    ask1_arr= df_test["ask_px_1"].values.astype(float)
    feat_arr= df_test[FEATURES].values.astype(float)

    # ── State ────────────────────────────────────────────────────
    orders:    List[dict] = []
    positions: List[dict] = []
    realized_pnl = 0.0
    daily_pnl    = 0.0
    equity       = float(CAPITAL)
    peak_equity  = float(CAPITAL)
    trade_log:   List[dict] = []
    rows_out:    List[dict] = []

    n = len(ts_arr)
    log.info(f"Running backtest on {n:,} rows ...")

    for i in range(n):
        ts  = ts_arr[i]
        mid = mid_arr[i]
        vol = vol_arr[i] if not np.isnan(vol_arr[i]) else 0.0
        if np.isnan(mid): continue

        tob_bid = int(bid1_arr[i]) if not np.isnan(bid1_arr[i]) else math.floor(mid)-1
        tob_ask = int(ask1_arr[i]) if not np.isnan(ask1_arr[i]) else math.ceil(mid)+1

        # MTM + equity
        mtm = sum(pos_mtm(p, mid) for p in positions if p["status"] == "open")
        equity = CAPITAL + realized_pnl + mtm
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0

        # Hard stop
        if daily_pnl < -CAPITAL * MAX_DAILY_LOSS_PCT or drawdown > MAX_DRAWDOWN_PCT:
            log.warning(f"Hard stop at row {i}"); break

        # ── Activate pending ──────────────────────────────────────
        for o in orders:
            if o["status"] == "pending" and ts >= o["arr_ts"]:
                o["status"] = "live"

        # ── Cancel arrivals ───────────────────────────────────────
        for o in orders:
            if o["status"] == "cancel_pending" and o["can_arr_ts"] and ts >= o["can_arr_ts"]:
                o["status"] = "cancelled"

        # ── Fill check ────────────────────────────────────────────
        # bid fills when tob_bid drops below order price (sweep)
        # ask fills when tob_ask rises above order price (sweep)
        newly_filled = []
        for o in orders:
            if o["status"] not in ("live", "cancel_pending"):
                continue
            hit = False
            if o["side"] == "bid" and tob_bid < o["price"]:
                hit = True
            elif o["side"] == "ask" and tob_ask > o["price"]:
                hit = True
            if hit:
                o["status"]    = "filled"
                o["fill_type"] = "stale_cancel" if o["status"] == "cancel_pending" else "sweep"
                newly_filled.append(o)

        for o in newly_filled:
            if o["type"] == "entry":
                mom    = mom_arr[i]
                dollar_move = abs(float(mom) * mid) if not np.isnan(mom) else 0.0
                phase  = "flat" if dollar_move < FLAT_THRESHOLD else ("up" if mom > 0 else "down")
                with_t = (phase == "flat"
                          or (phase == "up"   and o["side"] == "bid")
                          or (phase == "down" and o["side"] == "ask"))
                size   = BASE_SIZE_BTC if with_t else HALF_SIZE_BTC
                dirn   = "long" if o["side"] == "bid" else "short"
                # TP is ENTRY_OFFSET dollars from entry price — guarantees profitable round trip
                # NOT mid at fill time (mid has moved against us on a sweep fill)
                tp_px  = o["price"] + ENTRY_OFFSET if dirn == "long" else o["price"] - ENTRY_OFFSET
                pos    = make_position(dirn, o["price"], size, ts, tp_px)
                positions.append(pos)
                # TP order
                tp_side = "ask" if dirn == "long" else "bid"
                lat  = sim_lat(vol)
                tp_o = make_order("tp", pos["id"], tp_side, tp_px, size, ts, lat)
                pos["tp_ord_id"] = tp_o["id"]
                orders.append(tp_o)
                # SL order — resting limit on the loss side, same latency
                sl_px   = o["price"] - SL_PRICE_DIST if dirn == "long" else o["price"] + SL_PRICE_DIST
                sl_side = "bid" if dirn == "long" else "ask"
                lat2 = sim_lat(vol)
                sl_o = make_order("sl", pos["id"], sl_side, sl_px, size, ts, lat2)
                pos["sl_ord_id"] = sl_o["id"]
                orders.append(sl_o)
                log.debug(f"  OPEN {dirn}@{o['price']} TP={tp_px} SL={sl_px} phase={phase}")

            elif o["type"] in ("tp", "sl"):
                pos = next((p for p in positions
                            if p["id"] == o["pos_id"] and p["status"] == "open"), None)
                if pos:
                    pnl = ((o["price"] - pos["entry_px"]) * pos["size"]
                           if pos["dir"] == "long"
                           else (pos["entry_px"] - o["price"]) * pos["size"])
                    reason = o["type"]  # 'tp' or 'sl'
                    pos.update(status="closed", close_px=o["price"],
                               close_ts=ts, close_reason=reason, pnl=pnl,
                               hold_sec=ts - pos["entry_ts"])
                    realized_pnl += pnl; daily_pnl += pnl
                    trade_log.append({**pos})
                    # Cancel the counterpart order (tp cancels sl, sl cancels tp)
                    cancel_id = pos["sl_ord_id"] if reason == "tp" else pos["tp_ord_id"]
                    for co in orders:
                        if co["id"] == cancel_id and co["status"] in ("live","pending"):
                            co["status"] = "cancelled"
                    log.debug(f"  {reason.upper()} {pos['dir']}@{pos['entry_px']} "
                              f"close={o['price']} pnl={pnl:+.3f}")

        # ── Model prediction ──────────────────────────────────────
        x = feat_arr[i]
        if np.any(np.isnan(x)):
            orders = [o for o in orders if o["status"] in ("pending","live","cancel_pending")]
            continue
        pred = float(model.predict(np.clip(scaler.transform(x.reshape(1,-1)), -10, 10))[0])
        fv   = mid + pred

        # ── Entry management ──────────────────────────────────────
        longs  = sum(1 for p in positions if p["dir"] == "long"  and p["status"] == "open")
        shorts = sum(1 for p in positions if p["dir"] == "short" and p["status"] == "open")

        t_bid = math.floor(fv - ENTRY_OFFSET) if longs  < MAX_POS_PER_SIDE else None
        t_ask = math.ceil(fv  + ENTRY_OFFSET) if shorts < MAX_POS_PER_SIDE else None
        if t_bid is not None: t_bid = min(t_bid, tob_bid)
        if t_ask is not None: t_ask = max(t_ask, tob_ask)

        # Cancel stale entry orders FIRST
        for o in orders:
            if o["status"] != "live" or o["type"] != "entry": continue
            tgt = t_bid if o["side"] == "bid" else t_ask
            if tgt is None or abs(o["price"] - tgt) > REPRICE_THRESHOLD:
                lat = sim_lat(vol)
                o["status"] = "cancel_pending"
                o["can_arr_ts"] = ts + lat

        # Place new entry orders
        e_bids = [o for o in orders if o["type"]=="entry" and o["side"]=="bid"
                  and o["status"] in ("pending","live")]
        e_asks = [o for o in orders if o["type"]=="entry" and o["side"]=="ask"
                  and o["status"] in ("pending","live")]

        if t_bid is not None and not e_bids:
            orders.append(make_order("entry", None, "bid", t_bid,
                                     BASE_SIZE_BTC, ts, sim_lat(vol)))
        if t_ask is not None and not e_asks:
            orders.append(make_order("entry", None, "ask", t_ask,
                                     BASE_SIZE_BTC, ts, sim_lat(vol)))

        orders = [o for o in orders if o["status"] in ("pending","live","cancel_pending")]

        rows_out.append({"ts": ts, "mid": mid, "pred": pred,
                         "longs": longs, "shorts": shorts,
                         "equity": equity, "realized": realized_pnl, "mtm": mtm})

    # ═══════════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════════
    final_mid = mid_arr[-1]
    final_mtm = sum(pos_mtm(p, final_mid) for p in positions if p["status"]=="open")
    log.info("═" * 58)
    log.info(f"  Rows processed:  {len(rows_out):,}")
    log.info(f"  Trades closed:   {len(trade_log)}")
    log.info(f"  Realized PnL:    ${realized_pnl:+,.4f}")
    log.info(f"  MTM PnL:         ${final_mtm:+,.4f}")
    log.info(f"  Final equity:    ${equity:,.2f}")
    open_pos = [p for p in positions if p["status"]=="open"]
    log.info(f"  Still open:      {len(open_pos)} positions")
    log.info("═" * 58)

    if trade_log:
        trades = pd.DataFrame(trade_log)
        tp_count = (trades["close_reason"]=="tp").sum()
        sl_count = (trades["close_reason"]=="sl").sum()
        log.info(f"  TP closes: {tp_count}   SL closes: {sl_count}")

        for d in ["long", "short"]:
            t = trades[trades["dir"]==d]
            if t.empty: continue
            w = t[t["pnl"] > 0]
            l = t[t["pnl"] <= 0]
            log.info(f"\n  ── {d.upper()} ({len(t)} trades) ──")
            log.info(f"    Win rate:   {len(w)/len(t)*100:.1f}%")
            log.info(f"    Avg PnL:    ${t['pnl'].mean():+.4f}")
            log.info(f"    Avg win:    ${w['pnl'].mean():+.4f}" if len(w) else "    Avg win:    n/a")
            log.info(f"    Avg loss:   ${l['pnl'].mean():+.4f}" if len(l) else "    Avg loss:   n/a")
            log.info(f"    Avg hold:   {t['hold_sec'].mean():.1f}s")
            log.info(f"    Avg size:   {t['size'].mean():.3f} BTC")
        log.info("═" * 58)

        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        trades.to_csv(OUT_PATH.replace(".csv","_trades.csv"), index=False)

    pd.DataFrame(rows_out).to_csv(OUT_PATH, index=False)
    log.info(f"Results → {OUT_PATH}")

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info(f"Loading {FEAT_PATH}")
    df = pd.read_parquet(FEAT_PATH)
    log.info(f"Dataset: {len(df):,} rows")
    if TARGET not in df.columns:
        log.info("Computing target ...")
        df[TARGET] = compute_target(df)
    df_train = df[df.index <= TRAIN_END]
    df_test  = df[df.index >= TEST_START]
    log.info(f"Train: {len(df_train):,}  Test: {len(df_test):,}")
    np.random.seed(42)
    run_backtest(df_train, df_test)

if __name__ == "__main__":
    main()