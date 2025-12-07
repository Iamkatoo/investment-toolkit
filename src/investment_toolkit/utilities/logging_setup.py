import logging, inspect
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

ACTIVE_DIR = Path(
    "/Users/HOME/Library/Mobile Documents/com~apple~CloudDocs/InvestmentLogs/active"
    .replace(" ", r"\ ")            # ← logrotate と同じ表記で一貫
)
ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

def _handler(name: str) -> logging.Handler:
    h = TimedRotatingFileHandler(
        ACTIVE_DIR / f"{name}.log",
        when="midnight",
        interval=1,
        backupCount=14,              # ← 14 本（＝ 14 日）保持
        encoding="utf-8",
    )
    h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    return h

def setup_logger(name=None, level=logging.INFO, to_console=True):
    if name is None:
        frame = inspect.stack()[1]
        name = inspect.getmodule(frame[0]).__name__ or "main"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(_handler(name))
