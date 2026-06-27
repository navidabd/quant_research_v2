"""
Entry point.  Run with:

    python run_collect.py

Logs go to both stdout and collect.log so you can tail the file
or check it after the fact.
"""

import asyncio
import logging

from collect.collector import Collector


def setup_logging() -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("collect.log"),
        ],
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting collector")

    collector = Collector()
    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())