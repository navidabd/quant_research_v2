"""
Hyperliquid WebSocket collector.

Subscribes to l2Book and trades for every coin in config.COINS.
Buffers rows in memory and flushes to rotating parquet files every
config.SAVE_INTERVAL seconds.
"""

import asyncio
import json
import logging
import time

import websockets

import config
from collect.writer import RotatingParquetWriter

logger = logging.getLogger(__name__)


class Collector:
    def __init__(self):
        # one writer per (coin, channel) pair
        self._book_writers  = {c: RotatingParquetWriter(config.DATA_DIR, c, "books",  config.ROTATE_MINUTES) for c in config.COINS}
        self._trade_writers = {c: RotatingParquetWriter(config.DATA_DIR, c, "trades", config.ROTATE_MINUTES) for c in config.COINS}

        # in-memory buffers; cleared after each flush
        self._book_buf:  dict[str, list] = {c: [] for c in config.COINS}
        self._trade_buf: dict[str, list] = {c: [] for c in config.COINS}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        start = time.time()

        flush_task = asyncio.create_task(self._periodic_flush())

        try:
            while time.time() - start < config.RUN_SECONDS:
                try:
                    await self._session(start)
                except Exception as exc:
                    logger.error("Session ended: %s — flushing and reconnecting in 5s", exc)
                    self._flush_all()
                    await asyncio.sleep(5)
        finally:
            self._flush_all()
            self._close_writers()
            flush_task.cancel()
            logger.info("Collection finished")

    # ------------------------------------------------------------------
    # Single WebSocket session (reconnects on drop)
    # ------------------------------------------------------------------

    async def _session(self, start: float) -> None:
        async with websockets.connect(
            config.WS_URL,
            ping_interval=None,   # we send our own heartbeat
            max_size=None,
        ) as ws:
            await self._subscribe(ws)
            logger.info("Connected — subscribed to %s", config.COINS)

            asyncio.create_task(self._heartbeat(ws))

            while time.time() - start < config.RUN_SECONDS:
                raw = await ws.recv()
                local_ts = int(time.time() * 1000)
                msg = json.loads(raw)

                channel = msg.get("channel")

                if channel == "l2Book":
                    self._handle_book(msg, local_ts)

                elif channel == "trades":
                    self._handle_trades(msg, local_ts)

                # "pong" responses from our heartbeat — ignore

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def _subscribe(self, ws) -> None:
        for coin in config.COINS:
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "l2Book", "coin": coin},
            }))
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": coin},
            }))

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_book(self, msg: dict, local_ts: int) -> None:
        book = msg["data"]
        coin = book["coin"]

        if coin not in config.COINS:
            return

        bids = book["levels"][0]
        asks = book["levels"][1]

        row: dict = {
            "exchange_ts": book["time"],
            "local_ts":    local_ts,
        }

        for i, level in enumerate(bids[: config.BOOK_DEPTH], start=1):
            row[f"bid_px_{i}"] = float(level["px"])
            row[f"bid_sz_{i}"] = float(level["sz"])

        for i, level in enumerate(asks[: config.BOOK_DEPTH], start=1):
            row[f"ask_px_{i}"] = float(level["px"])
            row[f"ask_sz_{i}"] = float(level["sz"])

        self._book_buf[coin].append(row)

    def _handle_trades(self, msg: dict, local_ts: int) -> None:
        for trade in msg["data"]:
            coin = trade["coin"]

            if coin not in config.COINS:
                continue

            row = {
                "exchange_ts": trade["time"],
                "local_ts":    local_ts,
                "price":       float(trade["px"]),
                "size":        float(trade["sz"]),
                "side":        trade["side"],   # "B" or "A"
                "trade_id":    trade["tid"],
            }

            self._trade_buf[coin].append(row)

    # ------------------------------------------------------------------
    # Flush helpers
    # ------------------------------------------------------------------

    def _flush_all(self) -> None:
        for coin in config.COINS:
            if self._book_buf[coin]:
                self._book_writers[coin].write(self._book_buf[coin])
                logger.info("Flushed %d book rows for %s", len(self._book_buf[coin]), coin)
                self._book_buf[coin].clear()

            if self._trade_buf[coin]:
                self._trade_writers[coin].write(self._trade_buf[coin])
                logger.info("Flushed %d trade rows for %s", len(self._trade_buf[coin]), coin)
                self._trade_buf[coin].clear()

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(config.SAVE_INTERVAL)
            self._flush_all()

    def _close_writers(self) -> None:
        for coin in config.COINS:
            self._book_writers[coin].close()
            self._trade_writers[coin].close()

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(config.PING_INTERVAL)
            try:
                await ws.send(json.dumps({"method": "ping"}))
            except Exception:
                break