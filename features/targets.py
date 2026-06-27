"""
Target generation: fwd_change_{h}s = mid(t+h) - mid(t)

For each row at time t, predict how much the midprice will change
over the next h seconds. Backward lookup: finds the last known
book update at or before t+h (never uses future data).

Horizons: 1, 3, 5, 10, 15 seconds
"""

import numpy as np
import pandas as pd

HORIZONS_SEC = [1, 3, 5, 10, 15]


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    idx_ns    = df.index.astype("datetime64[ns, UTC]").view("int64")
    tol_ns    = int(2 * 1e9)   # 2s tolerance for sparse data

    # Lookup table: book updates only (mid is defined at book events)
    book_mask       = df["event_type"] == "book"
    book_ns         = idx_ns[book_mask]
    book_mid        = df.loc[book_mask, "mid"].values
    sort_order      = np.argsort(book_ns)
    book_ns_sorted  = book_ns[sort_order]
    book_mid_sorted = book_mid[sort_order]

    for h in HORIZONS_SEC:
        h_ns         = int(h * 1e9)
        target_times = idx_ns + h_ns
        insert_pos   = np.searchsorted(book_ns_sorted, target_times, side="right")

        future_mid = np.full(len(df), np.nan)
        for i, pos in enumerate(insert_pos):
            if pos == 0:
                continue
            last_pos = pos - 1
            last_ts  = book_ns_sorted[last_pos]
            # Reject if the last book update is too stale
            if last_ts >= (target_times[i] - tol_ns):
                future_mid[i] = book_mid_sorted[last_pos]

        # Target = future price change in dollars
        df[f"fwd_change_{h}s"] = (future_mid - df["mid"].values).astype("float32")

    return df