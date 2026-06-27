WS_URL = "wss://api.hyperliquid.xyz/ws"

COINS = ["BTC", "ETH"]

RUN_SECONDS = 3 * 24 * 3600  # 3 days; set to a smaller number to test
SAVE_INTERVAL = 60             # flush buffers to disk every N seconds
PING_INTERVAL = 30             # WebSocket heartbeat
BOOK_DEPTH = 20                # top N bid/ask levels to store
ROTATE_MINUTES = 60            # start a new parquet file every N minutes

DATA_DIR = "data/raw"