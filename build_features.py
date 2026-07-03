"""
Build full feature dataset and save to disk.
Run once — no train/test split here, that happens in train.py.

Output: data/features/BTC_features.parquet

Run:
    python build_features.py
"""

import gc
import logging
import os

import pandas as pd

from utils.loading import load_book, load_trades
from features.engineer import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs("data/features", exist_ok=True)

    logger.info("Loading raw BTC data ...")
    book   = load_book("BTC")
    trades = load_trades("BTC")
    logger.info("Book rows: %d   Trade rows: %d", len(book), len(trades))

    logger.info("Building features ...")
    df = build_features(book, trades)
    del book, trades
    gc.collect()

    # Downcast to save memory
    for col in df.select_dtypes("float32").columns:
        df[col] = df[col].astype("float32")

    out = "data/features/BTC_features.parquet"
    df.to_parquet(out)
    logger.info("Saved: %d rows, %d columns -> %s", len(df), df.shape[1], out)


if __name__ == "__main__":
    main()