"""
check_data2.py — Quality check on raw2 data
Run: python check_data2.py
"""

import os, glob, sys
import numpy as np
import pandas as pd

RAW2_DIR = "data/raw2"

def get_ts(df):
    """Find timestamp column and return as UTC datetime Series."""
    for col in ["local_ts", "exchange_ts", "timestamp", "time", "ts"]:
        if col not in df.columns:
            continue
        raw = df[col]
        if pd.api.types.is_datetime64_any_dtype(raw):
            return raw.dt.tz_localize("UTC") if raw.dt.tz is None else raw
        if pd.api.types.is_integer_dtype(raw) or pd.api.types.is_float_dtype(raw):
            med = float(raw.median())
            # Determine unit from magnitude
            if med > 1e16:    unit = "ns"
            elif med > 1e13:  unit = "us"
            elif med > 1e10:  unit = "ms"
            else:             unit = "s"
            s = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
            if s.notna().sum() > 0:
                return s.sort_values()
    return None

def load_parquets(files):
    dfs, bad = [], []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            bad.append((os.path.basename(f), str(e)[:80]))
    return dfs, bad

def check_asset(asset):
    print(f"\n{'='*60}\n  {asset}\n{'='*60}")

    for kind in ["books", "trades"]:
        fdir  = os.path.join(RAW2_DIR, asset, kind)
        files = sorted(glob.glob(os.path.join(fdir, "*.parquet")))
        print(f"\n  [{kind}]  files: {len(files)}")
        if not files:
            continue

        dfs, bad = load_parquets(files)
        if bad:
            print(f"  Corrupted ({len(bad)}): {[b[0] for b in bad]}")

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)[:8]}...")

        ts = get_ts(df)
        if ts is None:
            print("  WARNING: no timestamp column found — check column names above")
            continue

        ts = ts.dropna().sort_values().reset_index(drop=True)
        print(f"  Start:    {ts.iloc[0]}")
        print(f"  End:      {ts.iloc[-1]}")
        dur_h = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600
        print(f"  Duration: {dur_h:.1f}h  ({dur_h/24:.1f} days)")

        diffs = ts.diff().dropna().dt.total_seconds()
        thresh = 60 if kind == "books" else 120
        gaps   = diffs[diffs > thresh]
        print(f"  Gaps > {thresh}s: {len(gaps)}")
        for pos, g in gaps.nlargest(5).items():
            print(f"    {g:.0f}s gap ending at {ts.iloc[pos]}")

        if kind == "books":
            for col in ["bid_px_1", "ask_px_1"]:
                if col in df.columns:
                    print(f"  {col}: ${df[col].min():,.0f} - ${df[col].max():,.0f}")
        if kind == "trades" and "side" in df.columns:
            print(f"  Buy/Sell: {df['side'].value_counts().to_dict()}")

def main():
    print("DATA QUALITY CHECK - raw2")
    if not os.path.exists(RAW2_DIR):
        print(f"ERROR: {RAW2_DIR} not found"); return
    assets = [d for d in os.listdir(RAW2_DIR)
              if os.path.isdir(os.path.join(RAW2_DIR, d))]
    print(f"Assets: {assets}")
    for a in sorted(assets):
        check_asset(a)

if __name__ == "__main__":
    main()