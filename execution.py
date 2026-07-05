"""
execution.py — Backtest execution engine, BTC / Hyperliquid

Fill model:
  Entry orders  : post-only limit. If crosses TOB on arrival → cancelled.
  Touch fill    : trade_price == order_price → probabilistic (TOUCH_FILL_PROB)
  Trade-through : trade_price past order_price → full fill at order_price
  TP            : position metadata. Checked each trade row. Fill at tp_price (no slippage).
  SL            : trigger detected → latency delay → market fill at tob ± slippage ticks
"""

import math
import logging
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
FEE_MAKER        = 0.00000
FEE_TAKER        = 0.00045
LAT_BASE_MS      = 300
LAT_MAX_MS       = 1200
LAT_VOL_THR      = 0.0002
LAT_JITTER       = 0.10
TOUCH_FILL_PROB  = 0.5     # probability of fill when trade touches our price  [tune]
SL_SLIPPAGE_TICKS = 2      # extra ticks of slippage on SL market close        [tune]

# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class Order:
    id:           int
    side:         str            # 'bid' | 'ask'
    price:        int
    size:         float
    tp_offset:    int            # ticks above/below entry for TP
    sl_offset:    int            # ticks above/below entry for SL
    submitted_ts: float
    arrived_ts:   float
    queue_ahead:  float = 0.0
    status:       str   = "pending"   # pending|live|filled|cancelled|cancel_pending
    fill_price:   Optional[float] = None
    fill_ts:      Optional[float] = None
    can_arr_ts:   Optional[float] = None
    _fee:         float = 0.0

@dataclass
class Position:
    id:           int
    side:         str            # 'long' | 'short'
    entry_price:  float
    size:         float
    entry_ts:     float
    entry_fee:    float
    tp_price:     float          # fixed at open
    sl_price:     float          # fixed at open
    # SL pending state
    sl_pending:   bool  = False
    sl_arrives_ts: Optional[float] = None
    status:       str   = "open"
    exit_price:   Optional[float] = None
    exit_ts:      Optional[float] = None
    exit_reason:  str   = ""     # 'tp' | 'sl'
    exit_fee:     float = 0.0
    pnl_gross:    float = 0.0
    pnl_net:      float = 0.0
    hold_sec:     float = 0.0

# ── Engine ─────────────────────────────────────────────────────────────

class BacktestEngine:

    def __init__(self, df: pd.DataFrame, capital: float = 10_000,
                 fee_maker: float = FEE_MAKER, fee_taker: float = FEE_TAKER):
        self.df            = df
        self.capital       = capital
        self.fee_maker     = fee_maker
        self.fee_taker     = fee_taker
        self._has_trade_px = "trade_price" in df.columns
        if not self._has_trade_px:
            log.warning("trade_price missing — fill simulation degraded")

        self.orders:       List[Order]    = []
        self.positions:    List[Position] = []
        self.closed:       List[Position] = []
        self.equity_curve: List[float]   = []
        self.realized_pnl  = 0.0
        self.daily_pnl     = 0.0
        self.total_fees    = 0.0
        self._oid          = 0
        self._cancelled_post_only = 0

    # ── Public ────────────────────────────────────────────────────────

    def run(self, predictions: np.ndarray, strategy_fn: Callable,
            max_daily_loss_pct: float = 0.03, max_drawdown_pct: float = 0.10):

        df       = self.df
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

            # ── Equity + hard stop ────────────────────────────────
            mtm    = self._mtm(mid)
            equity = self.capital + self.realized_pnl + mtm
            peak_equity = max(peak_equity, equity)
            dd     = (peak_equity - equity) / peak_equity if peak_equity else 0.0
            self.equity_curve.append(equity)

            if self.daily_pnl < -self.capital * max_daily_loss_pct or dd > max_drawdown_pct:
                self._force_close_all(ts, tob_bid, tob_ask, vol, "hard_stop")
                log.warning(f"Hard stop at row {i}")
                break

            # ── Activate pending entry orders (post-only) ─────────
            # Post-only: if order crosses TOB on arrival → cancelled
            for o in self.orders:
                if o.status != "pending" or ts < o.arrived_ts:
                    continue
                if o.side == "bid" and tob_ask <= o.price:
                    # Would cross → cancel (post-only rejected)
                    o.status = "cancelled"
                    self._cancelled_post_only += 1
                elif o.side == "ask" and tob_bid >= o.price:
                    # Would cross → cancel (post-only rejected)
                    o.status = "cancelled"
                    self._cancelled_post_only += 1
                else:
                    # Safe to rest on book
                    o.queue_ahead = self._book_size_at(df.iloc[i], o.side, o.price)
                    o.status = "live"

            # ── Cancel arrivals ───────────────────────────────────
            for o in self.orders:
                if o.status == "cancel_pending" and o.can_arr_ts and ts >= o.can_arr_ts:
                    o.status = "cancelled"

            # ── Queue erosion (book updates only) ─────────────────
            is_book  = (evt_arr is None or evt_arr[i] == "book")
            is_trade = (evt_arr is not None and evt_arr[i] == "trade")
            if is_book:
                for o in self.orders:
                    if o.status == "live":
                        cur = self._book_size_at(df.iloc[i], o.side, o.price)
                        if cur < o.queue_ahead:
                            o.queue_ahead = max(0.0, cur)

            # ── Fill simulation for entry orders ──────────────────
            trade_px = float(tpx_arr[i]) if tpx_arr is not None and not np.isnan(tpx_arr[i]) else None
            trade_sz = float(tsz_arr[i]) if tsz_arr is not None and not np.isnan(tsz_arr[i]) else 0.0

            newly_filled = []
            for o in self.orders:
                if o.status not in ("live", "cancel_pending"):
                    continue

                filled = False
                if trade_px is not None and is_trade:
                    if o.side == "bid":
                        if trade_px < o.price:
                            # Trade-through → full fill
                            filled = True
                        elif int(trade_px) == o.price:
                            # Touch → probabilistic
                            filled = (np.random.random() < TOUCH_FILL_PROB)
                    else:  # ask
                        if trade_px > o.price:
                            filled = True
                        elif int(trade_px) == o.price:
                            filled = (np.random.random() < TOUCH_FILL_PROB)
                else:
                    # Fallback: sweep detection from book
                    if o.side == "bid" and tob_bid < o.price:
                        filled = True
                    elif o.side == "ask" and tob_ask > o.price:
                        filled = True

                if filled:
                    o.status     = "filled"
                    o.fill_price = float(o.price)   # limit fill at our price
                    o.fill_ts    = ts
                    o._fee       = o.fill_price * o.size * self.fee_maker
                    self.total_fees += o._fee
                    newly_filled.append(o)

            # ── Open positions from fills ─────────────────────────
            for o in newly_filled:
                self._open_position(o, ts)

            # ── TP check (position metadata, no separate order) ───
            # TP fills at tp_price exactly — limit order semantics
            if trade_px is not None and is_trade:
                for pos in [p for p in self.positions if p.status == "open"
                            and not p.sl_pending]:
                    tp_hit = False
                    if pos.side == "long":
                        if trade_px > pos.tp_price:          # trade-through TP
                            tp_hit = True
                        elif int(trade_px) == int(pos.tp_price):  # touch TP
                            tp_hit = (np.random.random() < TOUCH_FILL_PROB)
                    else:
                        if trade_px < pos.tp_price:
                            tp_hit = True
                        elif int(trade_px) == int(pos.tp_price):
                            tp_hit = (np.random.random() < TOUCH_FILL_PROB)
                    if tp_hit:
                        self._close_position(pos, pos.tp_price, ts, "tp", 0.0)

            # ── SL trigger → schedule latency-delayed market close ─
            for pos in [p for p in self.positions
                        if p.status == "open" and not p.sl_pending]:
                sl_triggered = ((pos.side == "long"  and mid <= pos.sl_price) or
                                (pos.side == "short" and mid >= pos.sl_price))
                if sl_triggered:
                    lat = self._lat(vol)
                    pos.sl_pending    = True
                    pos.sl_arrives_ts = ts + lat

            # ── Execute pending SL orders when latency expires ─────
            for pos in [p for p in self.positions
                        if p.status == "open" and p.sl_pending
                        and p.sl_arrives_ts is not None and ts >= p.sl_arrives_ts]:
                # Market close with slippage
                if pos.side == "long":
                    exit_px = float(tob_bid - SL_SLIPPAGE_TICKS)
                else:
                    exit_px = float(tob_ask + SL_SLIPPAGE_TICKS)
                fee_amt = exit_px * pos.size * self.fee_taker
                self._close_position(pos, exit_px, ts, "sl", fee_amt)

            # ── Strategy call ─────────────────────────────────────
            pred  = float(predictions[i]) if i < len(predictions) else np.nan
            state = {"ts": ts, "mid": mid, "pred": pred,
                     "tob_bid": tob_bid, "tob_ask": tob_ask,
                     "vol": vol, "equity": equity, "drawdown": dd}
            try:
                intents = strategy_fn(
                    state, [p for p in self.positions if p.status == "open"])
            except Exception as e:
                log.debug(f"Strategy error: {e}")
                intents = []

            for intent in (intents or []):
                self._process_intent(intent, ts, vol, tob_bid, tob_ask)

            # ── Prune dead orders ─────────────────────────────────
            self.orders = [o for o in self.orders
                           if o.status in ("pending","live","cancel_pending")]

        log.info("Backtest complete.")

    # ── Position management ───────────────────────────────────────────

    def _open_position(self, order: Order, ts: float):
        direction = "long" if order.side == "bid" else "short"
        fill_px   = order.fill_price
        tp_price  = (fill_px + order.tp_offset if direction == "long"
                     else fill_px - order.tp_offset)
        sl_price  = (fill_px - order.sl_offset if direction == "long"
                     else fill_px + order.sl_offset)
        pos = Position(
            id=self._next_id(), side=direction,
            entry_price=fill_px, size=order.size,
            entry_ts=ts, entry_fee=order._fee,
            tp_price=tp_price, sl_price=sl_price,
        )
        self.positions.append(pos)
        log.debug(f"  OPEN {direction}@{fill_px:.0f} TP={tp_price:.0f} SL={sl_price:.0f}")

    def _close_position(self, pos: Position, exit_px: float,
                        ts: float, reason: str, fee_amt: float):
        if pos.side == "long":
            gross = (exit_px - pos.entry_price) * pos.size
        else:
            gross = (pos.entry_price - exit_px) * pos.size
        net = gross - pos.entry_fee - fee_amt

        pos.exit_price  = exit_px
        pos.exit_ts     = ts
        pos.exit_reason = reason
        pos.exit_fee    = fee_amt
        pos.pnl_gross   = gross
        pos.pnl_net     = net
        pos.hold_sec    = ts - pos.entry_ts
        pos.status      = "closed"
        self.realized_pnl += net
        self.daily_pnl    += net
        self.total_fees   += fee_amt
        self.closed.append(pos)
        self.positions.remove(pos)
        log.debug(f"  CLOSE {pos.side}@{pos.entry_price:.0f} exit={exit_px:.0f} "
                  f"reason={reason} pnl={net:+.4f}")

    # ── Intent processing ─────────────────────────────────────────────

    def _process_intent(self, intent: dict, ts: float, vol: float,
                        tob_bid: int, tob_ask: int):
        action = intent.get("action")
        if action == "cancel":
            oid = intent.get("order_id")
            for o in self.orders:
                if o.id == oid and o.status == "live":
                    o.status     = "cancel_pending"
                    o.can_arr_ts = ts + self._lat(vol)
            return
        if action != "place":
            return

        side  = intent["side"]
        price = int(intent["price"])

        # Auto-cancel existing live entry on same side (repricing)
        for o in self.orders:
            if o.side == side and o.status == "live":
                o.status     = "cancel_pending"
                o.can_arr_ts = ts + self._lat(vol)

        lat = self._lat(vol)
        o = Order(
            id=self._next_id(),
            side=side,
            price=price,
            size=intent["size"],
            tp_offset=intent.get("tp_offset", 5),
            sl_offset=intent.get("sl_offset", 5),
            submitted_ts=ts,
            arrived_ts=ts + lat,
        )
        self.orders.append(o)

    # ── Helpers ───────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._oid += 1; return self._oid

    def _lat(self, vol: float) -> float:
        v  = min(1.0, vol / max(LAT_VOL_THR, 1e-9))
        ms = LAT_BASE_MS + (LAT_MAX_MS - LAT_BASE_MS) * (1 - (1 - v) ** 2)
        return max(LAT_BASE_MS, ms + np.random.normal(0, ms * LAT_JITTER)) / 1000.0

    def _book_size_at(self, row, side: str, price: int) -> float:
        prefix = "bid" if side == "bid" else "ask"
        for i in range(1, 21):
            px = row.get(f"{prefix}_px_{i}", np.nan)
            if pd.isna(px): break
            if int(px) == price:
                return float(row.get(f"{prefix}_sz_{i}", 0.0))
        return 0.0

    def _mtm(self, mid: float) -> float:
        return sum(
            (mid - p.entry_price) * p.size if p.side == "long"
            else (p.entry_price - mid) * p.size
            for p in self.positions if p.status == "open"
        )

    def _force_close_all(self, ts: float, tob_bid: int,
                         tob_ask: int, vol: float, reason: str):
        for pos in list(self.positions):
            if pos.status != "open": continue
            exit_px = float(tob_bid - SL_SLIPPAGE_TICKS if pos.side == "long"
                            else tob_ask + SL_SLIPPAGE_TICKS)
            fee_amt = exit_px * pos.size * self.fee_taker
            self._close_position(pos, exit_px, ts, reason, fee_amt)

    # ── Metrics ───────────────────────────────────────────────────────

    def compute_metrics(self) -> dict:
        trades    = self.closed
        n         = len(trades)
        ec        = np.array(self.equity_curve) if self.equity_curve else np.array([self.capital])
        final_mid = float(self.df["mid"].iloc[-1])

        if n == 0:
            return {"total_trades": 0, "capital": self.capital,
                    "final_equity": ec[-1] if len(ec) else self.capital,
                    "realized_pnl": self.realized_pnl,
                    "mtm_pnl": self._mtm(final_mid),
                    "total_fees": self.total_fees,
                    "cancelled_post_only": self._cancelled_post_only}

        pnls  = np.array([p.pnl_net   for p in trades])
        gross = np.array([p.pnl_gross for p in trades])
        fees  = np.array([p.entry_fee + p.exit_fee for p in trades])
        holds = np.array([p.hold_sec  for p in trades])

        wins   = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        returns = pnls / self.capital
        sharpe  = ((returns.mean() / (returns.std() + 1e-9))
                   * np.sqrt(252 * 24 * 3600 / max(holds.mean(), 1)))

        running_max = np.maximum.accumulate(ec)
        dd          = (running_max - ec) / (running_max + 1e-9)
        max_dd      = dd.max() * 100

        gw = gross[gross > 0].sum()
        gl = abs(gross[gross <= 0].sum())
        pf = gw / gl if gl > 0 else float("inf")

        cur = 0; max_consec = 0
        for p in pnls:
            cur = cur + 1 if p <= 0 else 0
            max_consec = max(max_consec, cur)

        total_secs = (self.df.index[-1] - self.df.index[0]).total_seconds()
        exposure   = min(100.0, holds.sum() / max(total_secs, 1) * 100)

        longs  = [p for p in trades if p.side == "long"]
        shorts = [p for p in trades if p.side == "short"]
        tp_cls = sum(1 for p in trades if p.exit_reason == "tp")
        sl_cls = sum(1 for p in trades if p.exit_reason == "sl")

        return {
            "capital":             self.capital,
            "final_equity":        ec[-1],
            "realized_pnl":        self.realized_pnl,
            "mtm_pnl":             self._mtm(final_mid),
            "gross_pnl":           gross.sum(),
            "total_fees":          self.total_fees,
            "total_trades":        n,
            "tp_closes":           tp_cls,
            "sl_closes":           sl_cls,
            "cancelled_post_only": self._cancelled_post_only,
            "max_drawdown_pct":    max_dd,
            "sharpe":              sharpe,
            "profit_factor":       pf,
            "exposure_pct":        exposure,
            "max_consec_loss":     max_consec,
            "win_rate":            len(wins) / n,
            "avg_pnl":             pnls.mean(),
            "avg_winner":          wins.mean()   if len(wins)   else 0.0,
            "avg_loser":           losses.mean() if len(losses) else 0.0,
            "largest_winner":      wins.max()    if len(wins)   else 0.0,
            "largest_loser":       losses.min()  if len(losses) else 0.0,
            "median_pnl":          float(np.median(pnls)),
            "avg_hold_sec":        holds.mean(),
            "long_pnl":            sum(p.pnl_net for p in longs),
            "short_pnl":           sum(p.pnl_net for p in shorts),
            "long_trades":         len(longs),
            "short_trades":        len(shorts),
            "avg_fee_per_trade":   fees.mean(),
            "fill_mode":           "trade_price" if self._has_trade_px else "tob_sweep",
        }

    def print_report(self):
        m = self.compute_metrics()
        print("\n" + "═" * 60)
        print("  EXECUTION REPORT")
        print("═" * 60)
        print(f"\n── OVERVIEW ──")
        print(f"  Capital:            ${m['capital']:,.2f}")
        print(f"  Final equity:       ${m['final_equity']:,.2f}")
        print(f"  Realized PnL:       ${m['realized_pnl']:+,.4f}")
        print(f"  MTM PnL:            ${m['mtm_pnl']:+,.4f}")
        print(f"  Total trades:       {m['total_trades']}  (TP={m.get('tp_closes',0)}  SL={m.get('sl_closes',0)})")
        print(f"  Cancelled post-only:{m.get('cancelled_post_only',0)}")
        print(f"  Fill mode:          {m['fill_mode']}")
        print(f"\n── RISK ──")
        print(f"  Max drawdown:       {m['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe:             {m['sharpe']:.3f}")
        print(f"  Profit factor:      {m['profit_factor']:.3f}")
        print(f"  Exposure:           {m['exposure_pct']:.1f}%")
        print(f"  Max consec loss:    {m['max_consec_loss']}")
        print(f"\n── TRADES ──")
        print(f"  Win rate:           {m['win_rate']*100:.1f}%")
        print(f"  Avg PnL/trade:      ${m['avg_pnl']:+.4f}")
        print(f"  Avg winner:         ${m['avg_winner']:+.4f}")
        print(f"  Avg loser:          ${m['avg_loser']:+.4f}")
        print(f"  Largest winner:     ${m['largest_winner']:+.4f}")
        print(f"  Largest loser:      ${m['largest_loser']:+.4f}")
        print(f"  Median PnL:         ${m['median_pnl']:+.4f}")
        print(f"  Avg hold:           {m['avg_hold_sec']:.1f}s")
        print(f"  Long PnL:           ${m['long_pnl']:+.4f}  ({m['long_trades']} trades)")
        print(f"  Short PnL:          ${m['short_pnl']:+.4f}  ({m['short_trades']} trades)")
        print(f"\n── COSTS ──")
        print(f"  Total fees:         ${m['total_fees']:+.4f}")
        print(f"  Avg fee/trade:      ${m['avg_fee_per_trade']:.4f}")
        print(f"\n── PNL ATTRIBUTION ──")
        print(f"  Gross PnL:          ${m['gross_pnl']:+.4f}")
        print(f"  Total fees:         ${-m['total_fees']:.4f}")
        print(f"  Net PnL:            ${m['realized_pnl']:+.4f}")
        print("═" * 60 + "\n")