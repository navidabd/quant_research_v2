"""
Build the BTC feature + target dataset.

ETH data is collected but excluded from this pipeline entirely.

Run:
    python build_dataset.py

Output:
    data/processed/BTC_dataset.parquet
"""

import gc
import logging
import os

import config
from utils.loading import load_book, load_trades
from features.engineer import build_features
from features.targets import build_targets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

OUT_DIR = "data/processed"


def downcast(df):
    float_cols = df.select_dtypes("float32").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Loading raw BTC data")
    book   = load_book("BTC")
    trades = load_trades("BTC")
    logger.info("BTC: %d book rows, %d trades", len(book), len(trades))

    feats = build_features(book, trades)
    del book, trades
    gc.collect()
    logger.info("BTC: union grid built — %d rows, %d feature columns",
                len(feats), feats.shape[1])

    logger.info("Building targets for BTC")
    df = build_targets(feats)
    del feats
    gc.collect()

    df = downcast(df)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    out    = os.path.join(OUT_DIR, "BTC_dataset.parquet")
    df.to_parquet(out)
    logger.info("Saved BTC: %d rows, %d columns, %.0f MB -> %s",
                len(df), df.shape[1], mem_mb, out)

    del df
    gc.collect()
    logger.info("Done")


if __name__ == "__main__":
    main()