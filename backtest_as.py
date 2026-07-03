"""
backtest_as.py — Avellaneda-Stoikov market making, BTC / Hyperliquid

A-S formulas:
  s(t)   = mid + model_pred          fair value (our Ridge model provides drift)
  r(t)   = s - q * γ * σ² * T       reservation price (skewed by inventory)
  δ(t)   = γ * σ² * T + (2/γ) * ln(1 + γ/κ)   optimal spread
  bid    = r - δ/2
  ask    = r + δ/2

Parameters:
  γ (GAMMA)    : risk aversion. higher → wider spread, faster inventory unwind
  T (T_HORIZ)  : time horizon in seconds
  κ (kappa)    : market order arrival rate (trades/sec), from trade_count_5s
  σ (sigma)    : dollar volatility, from realized_vol_10s × mid
  q            : current inventory in BTC (positive=long, negative=short)
"""

import math, logging, warnings, os
import numpy as np
import pandas as pd
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

# Latency
LATENCY_BASE_MS    = 450
LATENCY_MAX_MS     = 1200
LATENCY_VOL_THRESH = 0.0002
LATENCY_JITTER     = 0.10

# Model
MODEL_HORIZON      = "5s"

# A-S parameters
GAMMA              = 0.01    # risk aversion [tune: 0.001=tight, 0.1=wide]
T_HORIZ            = 10.0    # time horizon seconds [tune]
MIN_SPREAD_TICKS   = 1       # minimum half-spread from TOB (always at least L2)
MAX_SPREAD_TICKS   = 20      # cap spread so orders don't go too far

# Sizing
ORDER_SIZE_BTC     = 0.5     # 0.01 BTC × 50 leverage
MAX_INV_BTC        = 1.5     # max inventory → suppress adding side

# Risk
CAPITAL            = 10_000
MAX_DAILY_LOSS_PCT = 0.03
MAX_DRAWDOWN_PCT   = 0.10
REPRICE_THRESHOLD  = 2       # ticks before cancel+reprice

# Features
FEATURES = [
    "vwap_dev_1s", "vwap_dev_5s", "vwap_dev_10s", "vwap_dev_30s",
    "trade_imbalance_1s", "trade_imbalance_5s",
    "trade_imbalance_10s", "trade_imbalance_30s",
    "book_imbalance_l5", "ofi_l1",
    "momentum_2s", "momentum_5s", "momentum_10s",
]
TARGET     = "fwd_change_5s"
TRAIN_END  = pd.Timestamp("2026-06-21 08:08:00", tz="UTC")
TEST_START = pd.Timestamp("2026-06-21 08:10:15", tz="UTC")
FEAT_PATH  = "data/features/BTC_features.parquet"
OUT_PATH   = "data/results/backtest_as.csv"

# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

_oid = 0
def next_id():
    global _oid; _oid += 1; return _oid

def sim_lat(vol10: float) -> float:
    v  = min(1.0, vol10 / max(LATENCY_VOL_THRESH, 1e-9))
    ms = LATENCY_BASE_MS + (LATENCY_MAX_MS - LATENCY_BASE_MS) * (1 - (1-v)**2)
    return max(LATENCY_BASE_MS, ms + np.random.normal(0, ms * LATENCY_JITTER)) / 1000.0

def compute_target(df: pd.DataFrame, h: int = 5) -> pd.Series:
    fwd = df.index.searchsorted(df.index + pd.Timedelta(seconds=h))
    fwd = np.clip(fwd, 0, len(df) - 1)
    dt  = (df.index[fwd] - df.index).total_seconds()
    chg = df["mid"].values[fwd] - df["mid"].values
    return pd.Series(np.where(np.abs(dt - h) <= 2, chg, np.nan),
                     index=df.index, name=TARGET)

def make_order(side, price, ts, lat):
    return dict(id=next_id(), side=side, price=price,
                sub_ts=ts, arr_ts=ts+lat,
                status="pending", can_arr_ts=None)

# ═══════════════════════════════════════════════════════════════════
#  A-S QUOTE CALCULATION
# ═══════════════════════════════════════════════════════════════════

def as_quotes(s: float, q: float, sigma_dollar: float,
              kappa: float, tob_bid: int, tob_ask: int):
    """
    Returns (bid_price, ask_price) as integers using A-S formulas.
    s     : fair value (mid + model prediction)
    q     : current inventory in BTC
    sigma_dollar : dollar std of price moves (realized_vol × mid)
    kappa : market order arrival rate (trades/sec)
    """
    sigma2 = sigma_dollar ** 2
    kappa  = max(kappa, 0.1)   # floor to avoid log(0)

    # Reservation price — shifts quotes based on inventory
    r = s - q * GAMMA * sigma2 * T_HORIZ

    # Optimal half-spread
    half_spread = (GAMMA * sigma2 * T_HORIZ / 2.0
                   + (1.0 / GAMMA) * math.log(1.0 + GAMMA / kappa))

    # Clamp spread to reasonable range
    half_spread = max(MIN_SPREAD_TICKS, min(MAX_SPREAD_TICKS, half_spread))

    bid_raw = r - half_spread
    ask_raw = r + half_spread

    # Round conservatively (bid floors, ask ceils)
    bid_price = math.floor(bid_raw)
    ask_price = math.ceil(ask_raw)

    # Must not cross TOB (maker only)
    bid_price = min(bid_price, tob_bid)
    ask_price = max(ask_price, tob_ask)

    return bid_price, ask_price

# ═══════════════════════════════════════════════════════════════════
#  INVENTORY + PNL TRACKING
# ═══════════════════════════════════════════════════════════════════

class Inventory:
    def __init__(self):
        self.btc      = 0.0    # net BTC (+ long, - short)
        self.avg_cost = 0.0    # avg entry price of current position
        self.realized = 0.0
        self.daily    = 0.0
        self.equity   = float(CAPITAL)
        self.peak_eq  = float(CAPITAL)
        self.fill_log = []

    def apply_fill(self, side: str, price: int, size: float, ts: float):
        """Update inventory and realized PnL on fill."""
        pnl = 0.0
        rem = size

        if side == "bid":   # we buy
            if self.btc < 0:  # close short first
                close       = min(rem, -self.btc)
                pnl        += close * (self.avg_cost - price)
                self.btc   += close
                rem        -= close
            if rem > 1e-9:    # add to long
                total         = self.btc + rem
                self.avg_cost = (self.avg_cost * self.btc + price * rem) / total
                self.btc      = total

        else:               # we sell
            if self.btc > 0:  # close long first
                close       = min(rem, self.btc)
                pnl        += close * (price - self.avg_cost)
                self.btc   -= close
                rem        -= close
            if rem > 1e-9:    # add to short
                total         = abs(self.btc) + rem
                self.avg_cost = (self.avg_cost * abs(self.btc) + price * rem) / total
                self.btc     -= rem

        self.realized += pnl
        self.daily    += pnl
        self.fill_log.append({"ts": ts, "side": side, "price": price,
                              "size": size, "pnl": pnl, "inv": self.btc})
        return pnl

    def mtm(self, mid: float) -> float:
        return self.btc * (mid - self.avg_cost) if abs(self.btc) > 1e-9 else 0.0

    def update_equity(self, mid: float):
        self.equity  = CAPITAL + self.realized + self.mtm(mid)
        self.peak_eq = max(self.peak_eq, self.equity)

    def drawdown(self) -> float:
        return (self.peak_eq - self.equity) / self.peak_eq if self.peak_eq else 0.0

    def hard_stop(self) -> bool:
        return (self.daily < -CAPITAL * MAX_DAILY_LOSS_PCT
                or self.drawdown() > MAX_DRAWDOWN_PCT)

# ═══════════════════════════════════════════════════════════════════
#  BACKTEST LOOP
# ═══════════════════════════════════════════════════════════════════

def run_backtest(df_train: pd.DataFrame, df_test: pd.DataFrame):

    # ── Train model ──────────────────────────────────────────────
    log.info(f"Training on {len(df_train):,} rows ...")
    clean  = df_train[FEATURES + [TARGET]].dropna()
    scaler = RobustScaler(quantile_range=(5, 95))
    X_sc   = np.clip(scaler.fit_transform(clean[FEATURES].values), -10, 10)
    y_tr   = clean[TARGET].values
    model  = RidgeCV(alphas=np.logspace(-4, 4, 30), cv=3)
    model.fit(X_sc, y_tr)
    log.info(f"  alpha={model.alpha_:.4f}  train_R2={model.score(X_sc, y_tr):.3f}")

    # ── Pre-extract arrays ────────────────────────────────────────
    ts_arr   = np.array([t.timestamp() for t in df_test.index])
    mid_arr  = df_test["mid"].values.astype(float)
    vol_arr  = df_test["realized_vol_10s"].values.astype(float)
    tc_arr   = df_test["trade_count_5s"].values.astype(float)   # for kappa
    bid1_arr = df_test["bid_px_1"].values.astype(float)
    ask1_arr = df_test["ask_px_1"].values.astype(float)
    feat_arr = df_test[FEATURES].values.astype(float)

    inv      = Inventory()
    bid_ord  = None   # single active bid entry order
    ask_ord  = None   # single active ask entry order
    rows_out = []

    log.info(f"Running on {len(ts_arr):,} rows ...")

    for i in range(len(ts_arr)):
        ts  = ts_arr[i]
        mid = mid_arr[i]
        vol = vol_arr[i] if not np.isnan(vol_arr[i]) else 0.0
        if np.isnan(mid): continue

        tob_bid = int(bid1_arr[i]) if not np.isnan(bid1_arr[i]) else math.floor(mid)-1
        tob_ask = int(ask1_arr[i]) if not np.isnan(ask1_arr[i]) else math.ceil(mid)+1

        inv.update_equity(mid)
        if inv.hard_stop():
            log.warning(f"Hard stop at row {i}"); break

        # ── Activate pending orders ───────────────────────────────
        for o in [bid_ord, ask_ord]:
            if o and o["status"] == "pending" and ts >= o["arr_ts"]:
                o["status"] = "live"

        # ── Cancel arrivals ───────────────────────────────────────
        for o in [bid_ord, ask_ord]:
            if o and o["status"] == "cancel_pending" and o["can_arr_ts"] and ts >= o["can_arr_ts"]:
                o["status"] = "cancelled"

        # ── Fill check (sweep detection) ──────────────────────────
        if bid_ord and bid_ord["status"] in ("live", "cancel_pending"):
            if tob_bid < bid_ord["price"]:
                bid_ord["status"] = "filled"
                inv.apply_fill("bid", bid_ord["price"], ORDER_SIZE_BTC, ts)
                bid_ord = None

        if ask_ord and ask_ord["status"] in ("live", "cancel_pending"):
            if tob_ask > ask_ord["price"]:
                ask_ord["status"] = "filled"
                inv.apply_fill("ask", ask_ord["price"], ORDER_SIZE_BTC, ts)
                ask_ord = None

        # ── Model prediction ──────────────────────────────────────
        x = feat_arr[i]
        if np.any(np.isnan(x)): continue
        pred = float(model.predict(np.clip(scaler.transform(x.reshape(1,-1)), -10, 10))[0])

        # ── A-S quote calculation ─────────────────────────────────
        s           = mid + pred
        sigma_dollar = float(vol) * mid          # dollar vol from log-return vol × mid
        tc          = tc_arr[i] if not np.isnan(tc_arr[i]) else 1.0
        kappa       = tc / 5.0                   # trade_count_5s → trades per second

        target_bid, target_ask = as_quotes(s, inv.btc, sigma_dollar,
                                           kappa, tob_bid, tob_ask)

        # Suppress side if inventory limit hit
        if inv.btc >= MAX_INV_BTC:  target_bid = None
        if inv.btc <= -MAX_INV_BTC: target_ask = None

        # ── Cancel stale orders FIRST ─────────────────────────────
        if bid_ord and bid_ord["status"] == "live":
            if target_bid is None or abs(bid_ord["price"] - target_bid) > REPRICE_THRESHOLD:
                lat = sim_lat(vol)
                bid_ord["status"]      = "cancel_pending"
                bid_ord["can_arr_ts"]  = ts + lat
                bid_ord = None

        if ask_ord and ask_ord["status"] == "live":
            if target_ask is None or abs(ask_ord["price"] - target_ask) > REPRICE_THRESHOLD:
                lat = sim_lat(vol)
                ask_ord["status"]      = "cancel_pending"
                ask_ord["can_arr_ts"]  = ts + lat
                ask_ord = None

        # ── Place new orders ──────────────────────────────────────
        if target_bid is not None and (bid_ord is None or bid_ord["status"] in ("cancelled","filled")):
            bid_ord = make_order("bid", target_bid, ts, sim_lat(vol))

        if target_ask is not None and (ask_ord is None or ask_ord["status"] in ("cancelled","filled")):
            ask_ord = make_order("ask", target_ask, ts, sim_lat(vol))

        rows_out.append({
            "ts": ts, "mid": mid, "pred": pred,
            "s": s, "inv": inv.btc,
            "bid": target_bid, "ask": target_ask,
            "spread": (target_ask - target_bid) if target_bid and target_ask else None,
            "equity": inv.equity, "realized": inv.realized, "mtm": inv.mtm(mid),
        })

    # ═══════════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════════
    final_mid = mid_arr[-1]
    inv.update_equity(final_mid)

    log.info("═" * 58)
    log.info(f"  Rows processed:  {len(rows_out):,}")
    log.info(f"  Total fills:     {len(inv.fill_log)}")
    log.info(f"  Realized PnL:    ${inv.realized:+,.4f}")
    log.info(f"  MTM PnL:         ${inv.mtm(final_mid):+,.4f}")
    log.info(f"  Final equity:    ${inv.equity:,.2f}")
    log.info(f"  Final inventory: {inv.btc:+.4f} BTC")
    log.info(f"  Max drawdown:    {inv.drawdown()*100:.2f}%")
    log.info("═" * 58)

    if inv.fill_log:
        fills = pd.DataFrame(inv.fill_log)
        buys  = fills[fills["side"]=="bid"]
        sells = fills[fills["side"]=="ask"]
        log.info(f"  Buys: {len(buys)}   Sells: {len(sells)}")
        log.info(f"  Avg buy  price: ${buys['price'].mean():.1f}"  if len(buys)  else "")
        log.info(f"  Avg sell price: ${sells['price'].mean():.1f}" if len(sells) else "")

        # Round trips (paired fills)
        rts = []
        bi, si = 0, 0
        bl = buys.reset_index(drop=True)
        sl = sells.reset_index(drop=True)
        while bi < len(bl) and si < len(sl):
            b, s_ = bl.iloc[bi], sl.iloc[si]
            if b["ts"] < s_["ts"]:
                rts.append({"pnl": (s_["price"]-b["price"])*b["size"],
                            "hold": s_["ts"]-b["ts"], "dir": "long"})
                bi += 1; si += 1
            else:
                rts.append({"pnl": (s_["price"]-b["price"])*s_["size"],
                            "hold": b["ts"]-s_["ts"], "dir": "short"})
                bi += 1; si += 1

        if rts:
            rt = pd.DataFrame(rts)
            w  = rt[rt["pnl"] > 0]
            log.info(f"\n  Round trips:    {len(rt)}")
            log.info(f"  Win rate:       {len(w)/len(rt)*100:.1f}%")
            log.info(f"  Avg PnL/trip:   ${rt['pnl'].mean():+.4f}")
            log.info(f"  Avg hold:       {rt['hold'].mean():.1f}s")
        log.info("═" * 58)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame(rows_out).to_csv(OUT_PATH, index=False)
    if inv.fill_log:
        pd.DataFrame(inv.fill_log).to_csv(OUT_PATH.replace(".csv","_fills.csv"), index=False)
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