from dotenv import load_dotenv
import os
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import RealDictCursor

# プロジェクトのルートを相対パスで取得して.envを読み込む
# __file__ = src/investment_toolkit/utilities/config.py
# parent.parent.parent.parent = utilities/ -> investment_toolkit/ -> src/ -> プロジェクトルート
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# ロギングの設定
log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 環境変数を取得（デフォルト値付き）
FMP_API_KEY_PRIMARY = os.getenv("FMP_API_KEY_PRIMARY", "")
FMP_API_KEY_SECONDARY = os.getenv("FMP_API_KEY_SECONDARY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "investment")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")

# J-Quants API認証情報
JQUANTS_EMAIL = os.getenv("JQUANTS_EMAIL", "")
JQUANTS_PASSWORD = os.getenv("JQUANTS_PASSWORD", "")

@contextmanager
def get_connection() -> Generator[Connection, None, None]:
    """
    PostgreSQLデータベース接続を提供するコンテキストマネージャー。
    
    環境変数から接続情報を取得します。
    
    Yields:
        Connection: PostgreSQLデータベース接続
        
    Raises:
        psycopg2.Error: 接続エラーが発生した場合
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST or "localhost",
            port=int(str(DB_PORT) if DB_PORT and DB_PORT != "None" else "5432"),
            user=DB_USER or "",
            password=DB_PASSWORD or "",
            database=DB_NAME or "investment",
            cursor_factory=RealDictCursor
        )
        logger.debug("データベース接続を確立しました")
        yield conn
    except Exception as e:
        logger.error(f"データベース接続エラー: {str(e)}")
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.debug("データベース接続を閉じました")

