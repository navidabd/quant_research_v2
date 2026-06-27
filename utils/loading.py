"""
Load raw book/trade parquet files into clean, time-indexed DataFrames.
"""

import glob
import os

import pandas as pd

import config


def _load(symbol: str, channel: str) -> pd.DataFrame:
    pattern = os.path.join(config.DATA_DIR, symbol, channel, "*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No parquet files found at {pattern}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("local_ts").reset_index(drop=True)
    df.index = pd.to_datetime(df["local_ts"], unit="ms", utc=True)
    return df


def load_book(symbol: str) -> pd.DataFrame:
    df = _load(symbol, "books")
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_trades(symbol: str) -> pd.DataFrame:
    df = _load(symbol, "trades")
    df = df.drop_duplicates(subset="trade_id", keep="first")
    return df