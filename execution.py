"""
execution.py — Backtest execution engine, BTC / Hyperliquid
Standalone module. Plug in any model predictions + strategy function.

Usage:
    from execution import BacktestEngine

    def my_strategy(state, positions):
        # state: dict with mid, tob_bid, tob_ask, pred, vol, ts
        # positions: list of open position dicts
        # return: list of order dicts
        if state["pred"] > 3.0 and not positions:
            return [{"action":"place","side":"bid","order_type":"limit",
                     "price": state["tob_bid"], "size": 0.5}]
        return []

    engine = BacktestEngine(df, capital=10_000)
    engine.run(predictions=pred_array, strategy_fn=my_strategy)
    engine.print_report()
"""

import math, logging, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS — adjust per account/venue
# ═══════════════════════════════════════════════════════════════════

FEE_MAKER   = 0.00000   # Hyperliquid base tier maker (zero)
FEE_TAKER   = 0.00045   # Hyperliquid base tier taker (0.045%)

# Latency model (submission and cancel, in seconds)
LAT_BASE_MS = 450
LAT_MAX_MS  = 1200
LAT_VOL_THR = 0.0002    # realized_vol_10s threshold for max latency
LAT_JITTER  = 0.10

# ═══════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Order:
    id:          int
    side:        str          # 'bid' | 'ask'
    order_type:  str          # 'limit' | 'market'
    price:       Optional[int]
    size:        float
    submitted_ts: float
    arrived_ts:  float
    queue_ahead: float = 0.0
    status:      str   = "pending"   # pending|live|filled|cancelled|cancel_pending
    fill_price:  Optional[float] = None
    fill_ts:     Optional[float] = None
    fee_type:    str   = "maker"     # maker | taker
    can_arr_ts:  Optional[float] = None

@dataclass
class Position:
    id:          int
    side:        str          # 'long' | 'short'
    entry_price: float
    size:        float
    entry_ts:    float
    entry_fee:   float
    status:      str   = "open"
    exit_price:  Optional[float] = None
    exit_ts:     Optional[float] = None
    exit_reason: str   = ""
    exit_fee:    float = 0.0
    pnl_gross:   float = 0.0   # before fees
    pnl_net:     float = 0.0   # after fees
    hold_sec:    float = 0.0

# ═══════════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════════

class BacktestEngine:

    def __init__(self, df: pd.DataFrame, capital: float = 10_000,
                 fee_maker: float = FEE_MAKER, fee_taker: float = FEE_TAKER):
        self.df         = df
        self.capital    = capital
        self.fee_maker  = fee_maker
        self.fee_taker  = fee_taker
        self._has_trade_px = "trade_price" in df.columns
        if not self._has_trade_px:
            log.warning("'trade_price' column missing — using TOB-sweep fill approximation. "
                        "Add trade_price to features parquet for accurate touch fills.")

        # State (populated during run)
        self.orders:    List[Order]    = []
        self.positions: List[Position] = []
        self.closed:    List[Position] = []
        self.equity_curve: List[float] = []
        self.realized_pnl = 0.0
        self.daily_pnl    = 0.0
        self.total_fees   = 0.0
        self._oid         = 0

    # ─────────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────

    def run(self, predictions: np.ndarray, strategy_fn: Callable,
            max_daily_loss_pct: float = 0.03, max_drawdown_pct: float = 0.10):
        """
        Main loop. predictions[i] = model output for row i of df.
        strategy_fn(state_dict, open_positions) -> list of order dicts
        """
        df = self.df
        ts_arr   = np.array([t.timestamp() for t in df.index])
        mid_arr  = df["mid"].values.astype(float)
        bid1_arr = df["bid_px_1"].values.astype(float)
        ask1_arr = df["ask_px_1"].values.astype(float)
        vol_arr  = df["realized_vol_10s"].values.astype(float)
        tpx_arr  = df["trade_price"].values.astype(float) if self._has_trade_px else None
        tsz_arr  = df["trade_size"].values.astype(float)  if "trade_size" in df.columns else None
        evt_arr  = df["event_type"].values if "event_type" in df.columns else None

        peak_equity = self.capital
        log.info(f"Running backtest on {len(ts_arr):,} rows ...")

        for i in range(len(ts_arr)):
            ts  = ts_arr[i]
            mid = mid_arr[i]
            if np.isnan(mid): continue

            tob_bid = int(bid1_arr[i]) if not np.isnan(bid1_arr[i]) else math.floor(mid) - 1
            tob_ask = int(ask1_arr[i]) if not np.isnan(ask1_arr[i]) else math.ceil(mid)  + 1
            vol     = float(vol_arr[i]) if not np.isnan(vol_arr[i]) else 0.0

            # ── Equity ───────────────────────────────────────────
            mtm    = self._mtm(mid)
            equity = self.capital + self.realized_pnl + mtm
            peak_equity = max(peak_equity, equity)
            drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
            self.equity_curve.append(equity)

            # ── Hard stop ─────────────────────────────────────────
            if (self.daily_pnl < -self.capital * max_daily_loss_pct
                    or drawdown > max_drawdown_pct):
                self._force_close_all(ts, tob_bid, tob_ask, mid, "hard_stop")
                log.warning(f"Hard stop triggered at row {i}")
                break

            # ── Activate pending orders ───────────────────────────
            for o in self.orders:
                if o.status == "pending" and ts >= o.arrived_ts:
                    o.queue_ahead = self._book_size_at(self.df.iloc[i], o.side, o.price)
                    o.status = "live"

            # ── Cancel arrivals ───────────────────────────────────
            for o in self.orders:
                if o.status == "cancel_pending" and o.can_arr_ts and ts >= o.can_arr_ts:
                    o.status = "cancelled"

            # ── Queue erosion (book updates) ──────────────────────
            is_book = (evt_arr is None or evt_arr[i] == "book")
            if is_book:
                for o in self.orders:
                    if o.status == "live" and o.price is not None:
                        cur = self._book_size_at(self.df.iloc[i], o.side, o.price)
                        if cur < o.queue_ahead:
                            o.queue_ahead = max(0.0, cur)

            # ── Fill simulation ───────────────────────────────────
            is_trade = (evt_arr is not None and evt_arr[i] == "trade")
            trade_px = float(tpx_arr[i]) if (tpx_arr is not None
                            and not np.isnan(tpx_arr[i])) else None
            trade_sz = float(tsz_arr[i]) if (tsz_arr is not None
                            and not np.isnan(tsz_arr[i])) else float("inf")

            for o in self.orders:
                if o.status not in ("live", "cancel_pending"):
                    continue
                if o.order_type == "market":
                    # Market orders fill immediately on activation
                    fill_px = tob_ask if o.side == "bid" else tob_bid
                    self._fill_order(o, fill_px, ts, "taker")
                    continue

                filled = False
                # ── With trade_price (accurate) ───────────────────
                if trade_px is not None and is_trade:
                    if o.side == "bid":
                        if trade_px < o.price:              # sweep below bid
                            filled = True
                        elif trade_px == o.price:           # touch at bid
                            if trade_sz > o.queue_ahead:
                                filled = True
                            else:
                                o.queue_ahead = max(0, o.queue_ahead - trade_sz)
                    else:  # ask
                        if trade_px > o.price:              # sweep above ask
                            filled = True
                        elif trade_px == o.price:           # touch at ask
                            if trade_sz > o.queue_ahead:
                                filled = True
                            else:
                                o.queue_ahead = max(0, o.queue_ahead - trade_sz)
                # ── Fallback: TOB sweep (approximate) ────────────
                else:
                    if o.side == "bid" and tob_bid < o.price:
                        filled = True
                    elif o.side == "ask" and tob_ask > o.price:
                        filled = True

                if filled:
                    # Limit order fills at OUR stated price, not trade price
                    self._fill_order(o, o.price, ts, "maker")

            # ── Process fills → update positions ──────────────────
            for o in [x for x in self.orders if x.status == "filled"
                      and x.fill_ts == ts]:
                self._update_position(o, ts)

            # ── Strategy call ─────────────────────────────────────
            pred = predictions[i] if i < len(predictions) else np.nan
            state = {
                "ts": ts, "mid": mid, "pred": pred,
                "tob_bid": tob_bid, "tob_ask": tob_ask,
                "vol": vol, "equity": equity, "drawdown": drawdown,
            }
            if not np.isnan(pred) if not isinstance(pred, float) or not np.isnan(pred) else False:
                pass
            try:
                intents = strategy_fn(state, [p for p in self.positions if p.status == "open"])
            except Exception:
                intents = []

            for intent in intents:
                self._process_intent(intent, ts, vol)

            # Prune dead orders
            self.orders = [o for o in self.orders
                           if o.status in ("pending","live","cancel_pending")]

        log.info("Backtest complete.")

    def print_report(self):
        """Print full performance report."""
        metrics = self.compute_metrics()
        print("\n" + "═" * 60)
        print("  EXECUTION REPORT")
        print("═" * 60)

        print("\n── OVERVIEW ──")
        print(f"  Capital:          ${metrics['capital']:,.2f}")
        print(f"  Final equity:     ${metrics['final_equity']:,.2f}")
        print(f"  Realized PnL:     ${metrics['realized_pnl']:+,.4f}")
        print(f"  MTM PnL:          ${metrics['mtm_pnl']:+,.4f}")
        print(f"  Total trades:     {metrics['total_trades']}")

        print("\n── RISK METRICS ──")
        print(f"  Max drawdown:     {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe ratio:     {metrics['sharpe']:.3f}")
        print(f"  Profit factor:    {metrics['profit_factor']:.3f}")
        print(f"  Exposure time:    {metrics['exposure_pct']:.1f}%")
        print(f"  Max consec loss:  {metrics['max_consec_loss']}")

        print("\n── TRADE DECOMPOSITION ──")
        print(f"  Win rate:         {metrics['win_rate']*100:.1f}%")
        print(f"  Avg PnL/trade:    ${metrics['avg_pnl']:+.4f}")
        print(f"  Avg winner:       ${metrics['avg_winner']:+.4f}")
        print(f"  Avg loser:        ${metrics['avg_loser']:+.4f}")
        print(f"  Largest winner:   ${metrics['largest_winner']:+.4f}")
        print(f"  Largest loser:    ${metrics['largest_loser']:+.4f}")
        print(f"  Median PnL:       ${metrics['median_pnl']:+.4f}")
        print(f"  Avg hold time:    {metrics['avg_hold_sec']:.1f}s")
        print(f"  Long PnL:         ${metrics['long_pnl']:+.4f}  ({metrics['long_trades']} trades)")
        print(f"  Short PnL:        ${metrics['short_pnl']:+.4f}  ({metrics['short_trades']} trades)")

        print("\n── COST ANALYSIS ──")
        print(f"  Total fees:       ${metrics['total_fees']:+.4f}")
        print(f"  Avg fee/trade:    ${metrics['avg_fee_per_trade']:.4f}")
        print(f"  Maker fills:      {metrics['maker_fills']}")
        print(f"  Taker fills:      {metrics['taker_fills']}")
        print(f"  Fill mode:        {'trade_price (accurate)' if self._has_trade_px else 'TOB sweep (approximate)'}")

        print("\n── PNL ATTRIBUTION ──")
        print(f"  Gross PnL:        ${metrics['gross_pnl']:+.4f}")
        print(f"  Fees:             ${-metrics['total_fees']:.4f}")
        print(f"  Net PnL:          ${metrics['realized_pnl']:+.4f}")
        print("═" * 60 + "\n")

    def compute_metrics(self) -> dict:
        trades = self.closed
        n      = len(trades)
        ec     = np.array(self.equity_curve) if self.equity_curve else np.array([self.capital])
        final_mid = self.df["mid"].iloc[-1]

        if n == 0:
            return {"total_trades": 0, "capital": self.capital,
                    "final_equity": ec[-1], "realized_pnl": self.realized_pnl,
                    "mtm_pnl": self._mtm(final_mid)}

        pnls   = np.array([p.pnl_net for p in trades])
        gross  = np.array([p.pnl_gross for p in trades])
        fees   = np.array([p.entry_fee + p.exit_fee for p in trades])
        holds  = np.array([p.hold_sec for p in trades])
        sides  = np.array([p.side for p in trades])

        wins   = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        # Sharpe: annualised from per-trade returns
        returns = pnls / self.capital
        sharpe  = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 24 * 3600 / (holds.mean() + 1))

        # Max drawdown
        running_max = np.maximum.accumulate(ec)
        dd          = (running_max - ec) / running_max
        max_dd      = dd.max() * 100

        # Profit factor
        gross_wins   = gross[gross > 0].sum()
        gross_losses = abs(gross[gross <= 0].sum())
        pf           = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Consecutive losses
        consec, max_consec, cur = 0, 0, 0
        for p in pnls:
            if p <= 0: cur += 1; max_consec = max(max_consec, cur)
            else:      cur = 0

        # Exposure: fraction of rows where position was open
        # Approximate: total hold time / total test duration
        total_secs = (self.df.index[-1] - self.df.index[0]).total_seconds()
        exposure   = min(100.0, holds.sum() / total_secs * 100)

        longs  = [p for p in trades if p.side == "long"]
        shorts = [p for p in trades if p.side == "short"]

        maker_fills = sum(1 for o in self.orders if getattr(o,"fee_type","maker")=="maker")
        taker_fills = sum(1 for o in self.orders if getattr(o,"fee_type","taker")=="taker")

        return {
            "capital":          self.capital,
            "final_equity":     ec[-1],
            "realized_pnl":     self.realized_pnl,
            "mtm_pnl":          self._mtm(final_mid),
            "gross_pnl":        gross.sum(),
            "total_fees":       self.total_fees,
            "total_trades":     n,
            # Risk
            "max_drawdown_pct": max_dd,
            "sharpe":           sharpe,
            "profit_factor":    pf,
            "exposure_pct":     exposure,
            "max_consec_loss":  max_consec,
            # Trade
            "win_rate":         len(wins) / n,
            "avg_pnl":          pnls.mean(),
            "avg_winner":       wins.mean()   if len(wins)   else 0.0,
            "avg_loser":        losses.mean() if len(losses) else 0.0,
            "largest_winner":   wins.max()    if len(wins)   else 0.0,
            "largest_loser":    losses.min()  if len(losses) else 0.0,
            "median_pnl":       np.median(pnls),
            "avg_hold_sec":     holds.mean(),
            "long_pnl":         sum(p.pnl_net for p in longs),
            "short_pnl":        sum(p.pnl_net for p in shorts),
            "long_trades":      len(longs),
            "short_trades":     len(shorts),
            # Costs
            "avg_fee_per_trade": fees.mean(),
            "maker_fills":      maker_fills,
            "taker_fills":      taker_fills,
        }

    # ─────────────────────────────────────────────────────────────
    #  INTERNAL
    # ─────────────────────────────────────────────────────────────

    def _next_id(self):
        self._oid += 1; return self._oid

    def _lat(self, vol: float) -> float:
        v  = min(1.0, vol / max(LAT_VOL_THR, 1e-9))
        ms = LAT_BASE_MS + (LAT_MAX_MS - LAT_BASE_MS) * (1 - (1-v)**2)
        return max(LAT_BASE_MS, ms + np.random.normal(0, ms * LAT_JITTER)) / 1000.0

    def _book_size_at(self, row, side: str, price: Optional[int]) -> float:
        if price is None: return 0.0
        prefix = "bid" if side == "bid" else "ask"
        for i in range(1, 21):
            px = row.get(f"{prefix}_px_{i}", np.nan)
            if pd.isna(px): break
            if int(px) == price:
                return float(row.get(f"{prefix}_sz_{i}", 0.0))
        return 0.0

    def _fill_order(self, order: Order, fill_px: float, ts: float, fee_type: str):
        order.status     = "filled"
        order.fill_price = fill_px
        order.fill_ts    = ts
        order.fee_type   = fee_type
        fee_rate         = self.fee_maker if fee_type == "maker" else self.fee_taker
        order._fee       = fill_px * order.size * fee_rate

    def _update_position(self, order: Order, ts: float):
        """Match fill to open position or open new one."""
        fill_px = order.fill_price
        fee_amt = getattr(order, "_fee", 0.0)
        self.total_fees += fee_amt

        direction = "long" if order.side == "bid" else "short"

        # Check if this closes an existing position
        opposite = "short" if direction == "long" else "long"
        for pos in self.positions:
            if pos.status == "open" and pos.side == opposite:
                # Close this position
                if direction == "long":   # closing short
                    gross = (pos.entry_price - fill_px) * pos.size
                else:                     # closing long
                    gross = (fill_px - pos.entry_price) * pos.size
                net = gross - pos.entry_fee - fee_amt
                pos.exit_price  = fill_px
                pos.exit_ts     = ts
                pos.exit_fee    = fee_amt
                pos.pnl_gross   = gross
                pos.pnl_net     = net
                pos.hold_sec    = ts - pos.entry_ts
                pos.status      = "closed"
                self.realized_pnl += net
                self.daily_pnl    += net
                self.closed.append(pos)
                self.positions.remove(pos)
                return

        # No opposite position — open new
        pos = Position(
            id=self._next_id(), side=direction,
            entry_price=fill_px, size=order.size,
            entry_ts=ts, entry_fee=fee_amt,
        )
        self.positions.append(pos)

    def _mtm(self, mid: float) -> float:
        total = 0.0
        for p in self.positions:
            if p.status == "open":
                if p.side == "long":
                    total += (mid - p.entry_price) * p.size
                else:
                    total += (p.entry_price - mid) * p.size
        return total

    def _force_close_all(self, ts, tob_bid, tob_ask, mid, reason):
        for pos in self.positions:
            if pos.status != "open": continue
            exit_px = tob_bid if pos.side == "long" else tob_ask
            fee_amt = exit_px * pos.size * self.fee_taker
            if pos.side == "long":
                gross = (exit_px - pos.entry_price) * pos.size
            else:
                gross = (pos.entry_price - exit_px) * pos.size
            net = gross - pos.entry_fee - fee_amt
            pos.update(exit_price=exit_px, exit_ts=ts, exit_reason=reason,
                       exit_fee=fee_amt, pnl_gross=gross, pnl_net=net,
                       hold_sec=ts-pos.entry_ts, status="closed")
            self.realized_pnl += net
            self.total_fees   += fee_amt
            self.closed.append(pos)
        self.positions.clear()

    def _process_intent(self, intent: dict, ts: float, vol: float):
        action = intent.get("action")
        if action == "cancel":
            oid = intent.get("order_id")
            for o in self.orders:
                if o.id == oid and o.status == "live":
                    lat = self._lat(vol)
                    o.status      = "cancel_pending"
                    o.can_arr_ts  = ts + lat
            return

        if action != "place":
            return

        lat   = self._lat(vol)
        price = intent.get("price")
        if price is not None:
            price = int(price)

        o = Order(
            id=self._next_id(),
            side=intent["side"],
            order_type=intent.get("order_type", "limit"),
            price=price,
            size=intent["size"],
            submitted_ts=ts,
            arrived_ts=ts + lat,
        )
        if o.order_type == "market":
            o.status = "live"  # market orders skip pending
        self.orders.append(o)