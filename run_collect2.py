"""
Data collection run 2 — saves to data/raw2/ (BTC2 and ETH2 folders).

Identical to run_collect.py but uses a different output directory
so original 3-day dataset is never overwritten.

Run:
    python run_collect2.py
"""

import asyncio
import logging
import config

# Override data directory before importing collector
config.DATA_DIR = "data/raw2"
config.RUN_SECONDS = 3 * 24 * 3600   # 3 days

from collect.collector import Collector


def setup_logging():
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("collect2.log"),
        ],
    )


async def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting collector 2 — saving to data/raw2/ (BTC + ETH)")
    logger.info("Output: data/raw2/BTC/books/, data/raw2/BTC/trades/, "
                "data/raw2/ETH/books/, data/raw2/ETH/trades/")
    collector = Collector()
    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())