import logging, inspect
import os
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

# 環境変数でログディレクトリを指定可能 (デフォルト: ./logs)
LOG_DIR_STR = os.getenv("LOG_DIR", "./logs")
ACTIVE_DIR = Path(LOG_DIR_STR).expanduser().resolve()
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
