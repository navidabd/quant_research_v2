"""
Rotating Parquet writer.

Opens one ParquetWriter per (symbol, channel) pair. Rotates to a new file
every ROTATE_MINUTES minutes so individual files stay small and easy to load.
"""

import os
import logging
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class RotatingParquetWriter:
    """
    Writes rows to parquet files that rotate on a fixed time boundary.

    Files are stored at:
        {base_dir}/{symbol}/{channel}/{YYYYMMDD_HHMM}.parquet

    where HHMM is rounded down to the nearest ROTATE_MINUTES boundary.
    """

    def __init__(self, base_dir: str, symbol: str, channel: str, rotate_minutes: int = 60):
        self.base_dir = base_dir
        self.symbol = symbol
        self.channel = channel
        self.rotate_minutes = rotate_minutes

        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self._current_slot: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, rows: list[dict]) -> None:
        """Convert a list of row dicts to an Arrow table and flush to disk."""
        if not rows:
            return

        table = pa.Table.from_pylist(rows)
        self._maybe_rotate(table.schema)
        self._writer.write_table(table)

        logger.debug("%s/%s: wrote %d rows to %s", self.symbol, self.channel, len(rows), self._current_slot)

    def close(self) -> None:
        """Flush and close the current writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info("%s/%s: closed parquet writer", self.symbol, self.channel)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _time_slot(self) -> str:
        """Return e.g. '20250619_1400' for the current rotate-boundary."""
        now = datetime.now(timezone.utc)
        # floor minute to nearest rotate_minutes boundary
        floored = (now.minute // self.rotate_minutes) * self.rotate_minutes
        return now.strftime(f"%Y%m%d_%H{floored:02d}")

    def _maybe_rotate(self, schema: pa.Schema) -> None:
        """Open a new writer if the time slot has changed."""
        slot = self._time_slot()

        if slot == self._current_slot:
            return  # still in the same window

        # close the previous writer before opening a new file
        self.close()

        self._current_slot = slot
        self._schema = schema

        dir_path = os.path.join(self.base_dir, self.symbol, self.channel)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"{slot}.parquet")

        self._writer = pq.ParquetWriter(file_path, schema, compression="snappy")
        logger.info("%s/%s: opened new file %s", self.symbol, self.channel, file_path)