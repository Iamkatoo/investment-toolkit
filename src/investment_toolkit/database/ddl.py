#!/usr/bin/env python
"""
データベーススキーマ定義と作成スクリプトを提供するモジュール。
"""
import logging
from typing import List, Dict, Any

from sqlalchemy import MetaData, Table, Column, Integer, BigInteger, Numeric, Text, Date, Index
from sqlalchemy.schema import CreateTable

from ..utilities.config import get_connection

logger = logging.getLogger(__name__)

# メタデータ定義
metadata = MetaData()

# セクターとインダストリーの価格テーブル定義
sector_daily_prices = Table(
    'sector_daily_prices',
    metadata,
    Column('trade_date', Date, primary_key=True),
    Column('group_name', Text, primary_key=True, nullable=False),
    Column('currency', Text, primary_key=True, nullable=False),
    Column('avg_open', Numeric(18, 4)),
    Column('avg_high', Numeric(18, 4)),
    Column('avg_low', Numeric(18, 4)),
    Column('avg_close', Numeric(18, 4)),
    Column('avg_volume', BigInteger),
    Column('symbol_count', Integer),
    schema='calculated_metrics'
)
Index('idx_sector_daily_prices_date', sector_daily_prices.c.trade_date)

sector_weekly_prices = Table(
    'sector_weekly_prices',
    metadata,
    Column('trade_date', Date, primary_key=True),
    Column('group_name', Text, primary_key=True, nullable=False),
    Column('currency', Text, primary_key=True, nullable=False),
    Column('avg_open', Numeric(18, 4)),
    Column('avg_high', Numeric(18, 4)),
    Column('avg_low', Numeric(18, 4)),
    Column('avg_close', Numeric(18, 4)),
    Column('avg_volume', BigInteger),
    Column('symbol_count', Integer),
    schema='calculated_metrics'
)
Index('idx_sector_weekly_prices_date', sector_weekly_prices.c.trade_date)

sector_monthly_prices = Table(
    'sector_monthly_prices',
    metadata,
    Column('trade_date', Date, primary_key=True),
    Column('group_name', Text, primary_key=True, nullable=False),
    Column('currency', Text, primary_key=True, nullable=False),
    Column('avg_open', Numeric(18, 4)),
    Column('avg_high', Numeric(18, 4)),
    Column('avg_low', Numeric(18, 4)),
    Column('avg_close', Numeric(18, 4)),
    Column('avg_volume', BigInteger),
    Column('symbol_count', Integer),
    schema='calculated_metrics'
)
Index('idx_sector_monthly_prices_date', sector_monthly_prices.c.trade_date)

industry_daily_prices = Table(
    'industry_daily_prices',
    metadata,
    Column('trade_date', Date, primary_key=True),
    Column('group_name', Text, primary_key=True, nullable=False),
    Column('currency', Text, primary_key=True, nullable=False),
    Column('avg_open', Numeric(18, 4)),
    Column('avg_high', Numeric(18, 4)),
    Column('avg_low', Numeric(18, 4)),
    Column('avg_close', Numeric(18, 4)),
    Column('avg_volume', BigInteger),
    Column('symbol_count', Integer),
    schema='calculated_metrics'
)
Index('idx_industry_daily_prices_date', industry_daily_prices.c.trade_date)

industry_weekly_prices = Table(
    'industry_weekly_prices',
    metadata,
    Column('trade_date', Date, primary_key=True),
    Column('group_name', Text, primary_key=True, nullable=False),
    Column('currency', Text, primary_key=True, nullable=False),
    Column('avg_open', Numeric(18, 4)),
    Column('avg_high', Numeric(18, 4)),
    Column('avg_low', Numeric(18, 4)),
    Column('avg_close', Numeric(18, 4)),
    Column('avg_volume', BigInteger),
    Column('symbol_count', Integer),
    schema='calculated_metrics'
)
Index('idx_industry_weekly_prices_date', industry_weekly_prices.c.trade_date)

industry_monthly_prices = Table(
    'industry_monthly_prices',
    metadata,
    Column('trade_date', Date, primary_key=True),
    Column('group_name', Text, primary_key=True, nullable=False),
    Column('currency', Text, primary_key=True, nullable=False),
    Column('avg_open', Numeric(18, 4)),
    Column('avg_high', Numeric(18, 4)),
    Column('avg_low', Numeric(18, 4)),
    Column('avg_close', Numeric(18, 4)),
    Column('avg_volume', BigInteger),
    Column('symbol_count', Integer),
    schema='calculated_metrics'
)
Index('idx_industry_monthly_prices_date', industry_monthly_prices.c.trade_date)

# 全テーブルリスト
all_tables = [
    sector_daily_prices, sector_weekly_prices, sector_monthly_prices,
    industry_daily_prices, industry_weekly_prices, industry_monthly_prices
]

# DDL SQL文の生成
def get_create_tables_sql() -> str:
    """
    テーブル作成用のSQL文を生成します。
    
    Returns:
        str: スキーマとテーブル作成用のSQL文
    """
    sql = """
    CREATE SCHEMA IF NOT EXISTS calculated_metrics;
    
    -- セクター日次価格テーブル
    CREATE TABLE IF NOT EXISTS calculated_metrics.sector_daily_prices (
        trade_date DATE,
        group_name TEXT NOT NULL,
        currency TEXT NOT NULL,
        avg_open NUMERIC(18,4),
        avg_high NUMERIC(18,4),
        avg_low NUMERIC(18,4),
        avg_close NUMERIC(18,4),
        avg_volume BIGINT,
        symbol_count INTEGER,
        PRIMARY KEY (trade_date, group_name, currency)
    );
    CREATE INDEX IF NOT EXISTS idx_sector_daily_prices_date ON calculated_metrics.sector_daily_prices (trade_date);
    
    -- セクター週次価格テーブル
    CREATE TABLE IF NOT EXISTS calculated_metrics.sector_weekly_prices (
        trade_date DATE,
        group_name TEXT NOT NULL,
        currency TEXT NOT NULL,
        avg_open NUMERIC(18,4),
        avg_high NUMERIC(18,4),
        avg_low NUMERIC(18,4),
        avg_close NUMERIC(18,4),
        avg_volume BIGINT,
        symbol_count INTEGER,
        PRIMARY KEY (trade_date, group_name, currency)
    );
    CREATE INDEX IF NOT EXISTS idx_sector_weekly_prices_date ON calculated_metrics.sector_weekly_prices (trade_date);
    
    -- セクター月次価格テーブル
    CREATE TABLE IF NOT EXISTS calculated_metrics.sector_monthly_prices (
        trade_date DATE,
        group_name TEXT NOT NULL,
        currency TEXT NOT NULL,
        avg_open NUMERIC(18,4),
        avg_high NUMERIC(18,4),
        avg_low NUMERIC(18,4),
        avg_close NUMERIC(18,4),
        avg_volume BIGINT,
        symbol_count INTEGER,
        PRIMARY KEY (trade_date, group_name, currency)
    );
    CREATE INDEX IF NOT EXISTS idx_sector_monthly_prices_date ON calculated_metrics.sector_monthly_prices (trade_date);
    
    -- インダストリー日次価格テーブル
    CREATE TABLE IF NOT EXISTS calculated_metrics.industry_daily_prices (
        trade_date DATE,
        group_name TEXT NOT NULL,
        currency TEXT NOT NULL,
        avg_open NUMERIC(18,4),
        avg_high NUMERIC(18,4),
        avg_low NUMERIC(18,4),
        avg_close NUMERIC(18,4),
        avg_volume BIGINT,
        symbol_count INTEGER,
        PRIMARY KEY (trade_date, group_name, currency)
    );
    CREATE INDEX IF NOT EXISTS idx_industry_daily_prices_date ON calculated_metrics.industry_daily_prices (trade_date);
    
    -- インダストリー週次価格テーブル
    CREATE TABLE IF NOT EXISTS calculated_metrics.industry_weekly_prices (
        trade_date DATE,
        group_name TEXT NOT NULL,
        currency TEXT NOT NULL,
        avg_open NUMERIC(18,4),
        avg_high NUMERIC(18,4),
        avg_low NUMERIC(18,4),
        avg_close NUMERIC(18,4),
        avg_volume BIGINT,
        symbol_count INTEGER,
        PRIMARY KEY (trade_date, group_name, currency)
    );
    CREATE INDEX IF NOT EXISTS idx_industry_weekly_prices_date ON calculated_metrics.industry_weekly_prices (trade_date);
    
    -- インダストリー月次価格テーブル
    CREATE TABLE IF NOT EXISTS calculated_metrics.industry_monthly_prices (
        trade_date DATE,
        group_name TEXT NOT NULL,
        currency TEXT NOT NULL,
        avg_open NUMERIC(18,4),
        avg_high NUMERIC(18,4),
        avg_low NUMERIC(18,4),
        avg_close NUMERIC(18,4),
        avg_volume BIGINT,
        symbol_count INTEGER,
        PRIMARY KEY (trade_date, group_name, currency)
    );
    CREATE INDEX IF NOT EXISTS idx_industry_monthly_prices_date ON calculated_metrics.industry_monthly_prices (trade_date);
    """
    return sql

def create_tables() -> None:
    """
    データベースにテーブルを作成します。
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(get_create_tables_sql())
            conn.commit()
        logger.info("テーブルが正常に作成されました")
    except Exception as e:
        logger.error(f"テーブル作成エラー: {str(e)}")
        raise


if __name__ == "__main__":
    create_tables() 