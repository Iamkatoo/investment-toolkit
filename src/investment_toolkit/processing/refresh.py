#!/usr/bin/env python
"""
セクターとインダストリーの価格データを更新するモジュールです。
"""
import logging
from datetime import date
from typing import Dict

from ..utilities.config import get_connection
from ..utils.helpers import normalize_name

logger = logging.getLogger(__name__)

# 更新クエリ定義
SECTOR_DAILY_UPSERT_QUERY = """
INSERT INTO calculated_metrics.sector_daily_prices
    (trade_date, group_name, currency, avg_open, avg_high, avg_low, avg_close, avg_volume, symbol_count)
SELECT 
    trade_date,
    group_name,
    currency,
    AVG(open) AS avg_open,
    AVG(high) AS avg_high,
    AVG(low) AS avg_low,
    AVG(close) AS avg_close,
    AVG(volume)::bigint AS avg_volume,
    SUM(cnt) AS symbol_count
FROM (
    SELECT DISTINCT
        dp.date AS trade_date,
        COALESCE(normalize_name_sql(gs.sector_name), 'Unclassified') AS group_name,
        COALESCE(cp.currency, 'USD') AS currency,
        dp.open,
        dp.high,
        dp.low,
        dp.close,
        dp.volume,
        1 AS cnt
    FROM fmp_data.daily_prices dp
    LEFT JOIN reference.company_gics cg ON dp.symbol = cg.symbol
    LEFT JOIN reference.gics_sector gs ON cg.sector_id = gs.sector_id
    LEFT JOIN fmp_data.company_profile cp ON dp.symbol = cp.symbol
    WHERE dp.date >= %(from_date)s
) t
GROUP BY trade_date, group_name, currency
ON CONFLICT (trade_date, group_name, currency) DO UPDATE SET
    avg_open = EXCLUDED.avg_open,
    avg_high = EXCLUDED.avg_high,
    avg_low = EXCLUDED.avg_low,
    avg_close = EXCLUDED.avg_close,
    avg_volume = EXCLUDED.avg_volume,
    symbol_count = EXCLUDED.symbol_count
"""

INDUSTRY_DAILY_UPSERT_QUERY = """
INSERT INTO calculated_metrics.industry_daily_prices
    (trade_date, group_name, currency, avg_open, avg_high, avg_low, avg_close, avg_volume, symbol_count)
SELECT 
    trade_date,
    group_name,
    currency,
    AVG(open) AS avg_open,
    AVG(high) AS avg_high,
    AVG(low) AS avg_low,
    AVG(close) AS avg_close,
    AVG(volume)::bigint AS avg_volume,
    SUM(cnt) AS symbol_count
FROM (
    SELECT DISTINCT
        dp.date AS trade_date,
        COALESCE(normalize_name_sql(gi.industry_name), 'Unclassified') AS group_name,
        COALESCE(cp.currency, 'USD') AS currency,
        dp.open,
        dp.high,
        dp.low,
        dp.close,
        dp.volume,
        1 AS cnt
    FROM fmp_data.daily_prices dp
    LEFT JOIN reference.company_gics cg ON dp.symbol = cg.symbol
    LEFT JOIN reference.gics_industry gi ON cg.industry_id = gi.industry_id
    LEFT JOIN fmp_data.company_profile cp ON dp.symbol = cp.symbol
    WHERE dp.date >= %(from_date)s
) t
GROUP BY trade_date, group_name, currency
ON CONFLICT (trade_date, group_name, currency) DO UPDATE SET
    avg_open = EXCLUDED.avg_open,
    avg_high = EXCLUDED.avg_high,
    avg_low = EXCLUDED.avg_low,
    avg_close = EXCLUDED.avg_close,
    avg_volume = EXCLUDED.avg_volume,
    symbol_count = EXCLUDED.symbol_count
"""

SECTOR_WEEKLY_UPSERT_QUERY = """
INSERT INTO calculated_metrics.sector_weekly_prices
    (trade_date, group_name, currency, avg_open, avg_high, avg_low, avg_close, avg_volume, symbol_count)
SELECT 
    trade_date,
    group_name,
    currency,
    AVG(open) AS avg_open,
    AVG(high) AS avg_high,
    AVG(low) AS avg_low,
    AVG(close) AS avg_close,
    AVG(volume)::bigint AS avg_volume,
    SUM(cnt) AS symbol_count
FROM (
    SELECT DISTINCT
        wp.week_start_date AS trade_date,
        COALESCE(normalize_name_sql(gs.sector_name), 'Unclassified') AS group_name,
        COALESCE(cp.currency, 'USD') AS currency,
        wp.open,
        wp.high,
        wp.low,
        wp.close,
        wp.volume,
        1 AS cnt
    FROM calculated_metrics.weekly_prices wp
    LEFT JOIN reference.company_gics cg ON wp.symbol = cg.symbol
    LEFT JOIN reference.gics_sector gs ON cg.sector_id = gs.sector_id
    LEFT JOIN fmp_data.company_profile cp ON wp.symbol = cp.symbol
    WHERE wp.week_start_date >= %(from_date)s
) t
GROUP BY trade_date, group_name, currency
ON CONFLICT (trade_date, group_name, currency) DO UPDATE SET
    avg_open = EXCLUDED.avg_open,
    avg_high = EXCLUDED.avg_high,
    avg_low = EXCLUDED.avg_low,
    avg_close = EXCLUDED.avg_close,
    avg_volume = EXCLUDED.avg_volume,
    symbol_count = EXCLUDED.symbol_count
"""

INDUSTRY_WEEKLY_UPSERT_QUERY = """
INSERT INTO calculated_metrics.industry_weekly_prices
    (trade_date, group_name, currency, avg_open, avg_high, avg_low, avg_close, avg_volume, symbol_count)
SELECT 
    trade_date,
    group_name,
    currency,
    AVG(open) AS avg_open,
    AVG(high) AS avg_high,
    AVG(low) AS avg_low,
    AVG(close) AS avg_close,
    AVG(volume)::bigint AS avg_volume,
    SUM(cnt) AS symbol_count
FROM (
    SELECT DISTINCT
        wp.week_start_date AS trade_date,
        COALESCE(normalize_name_sql(gi.industry_name), 'Unclassified') AS group_name,
        COALESCE(cp.currency, 'USD') AS currency,
        wp.open,
        wp.high,
        wp.low,
        wp.close,
        wp.volume,
        1 AS cnt
    FROM calculated_metrics.weekly_prices wp
    LEFT JOIN reference.company_gics cg ON wp.symbol = cg.symbol
    LEFT JOIN reference.gics_industry gi ON cg.industry_id = gi.industry_id
    LEFT JOIN fmp_data.company_profile cp ON wp.symbol = cp.symbol
    WHERE wp.week_start_date >= %(from_date)s
) t
GROUP BY trade_date, group_name, currency
ON CONFLICT (trade_date, group_name, currency) DO UPDATE SET
    avg_open = EXCLUDED.avg_open,
    avg_high = EXCLUDED.avg_high,
    avg_low = EXCLUDED.avg_low,
    avg_close = EXCLUDED.avg_close,
    avg_volume = EXCLUDED.avg_volume,
    symbol_count = EXCLUDED.symbol_count
"""

SECTOR_MONTHLY_UPSERT_QUERY = """
INSERT INTO calculated_metrics.sector_monthly_prices
    (trade_date, group_name, currency, avg_open, avg_high, avg_low, avg_close, avg_volume, symbol_count)
SELECT 
    trade_date,
    group_name,
    currency,
    AVG(open) AS avg_open,
    AVG(high) AS avg_high,
    AVG(low) AS avg_low,
    AVG(close) AS avg_close,
    AVG(volume)::bigint AS avg_volume,
    SUM(cnt) AS symbol_count
FROM (
    SELECT DISTINCT
        mp.month_start_date AS trade_date,
        COALESCE(normalize_name_sql(gs.sector_name), 'Unclassified') AS group_name,
        COALESCE(cp.currency, 'USD') AS currency,
        mp.open,
        mp.high,
        mp.low,
        mp.close,
        mp.volume,
        1 AS cnt
    FROM calculated_metrics.monthly_prices mp
    LEFT JOIN reference.company_gics cg ON mp.symbol = cg.symbol
    LEFT JOIN reference.gics_sector gs ON cg.sector_id = gs.sector_id
    LEFT JOIN fmp_data.company_profile cp ON mp.symbol = cp.symbol
    WHERE mp.month_start_date >= %(from_date)s
) t
GROUP BY trade_date, group_name, currency
ON CONFLICT (trade_date, group_name, currency) DO UPDATE SET
    avg_open = EXCLUDED.avg_open,
    avg_high = EXCLUDED.avg_high,
    avg_low = EXCLUDED.avg_low,
    avg_close = EXCLUDED.avg_close,
    avg_volume = EXCLUDED.avg_volume,
    symbol_count = EXCLUDED.symbol_count
"""

INDUSTRY_MONTHLY_UPSERT_QUERY = """
INSERT INTO calculated_metrics.industry_monthly_prices
    (trade_date, group_name, currency, avg_open, avg_high, avg_low, avg_close, avg_volume, symbol_count)
SELECT 
    trade_date,
    group_name,
    currency,
    AVG(open) AS avg_open,
    AVG(high) AS avg_high,
    AVG(low) AS avg_low,
    AVG(close) AS avg_close,
    AVG(volume)::bigint AS avg_volume,
    SUM(cnt) AS symbol_count
FROM (
    SELECT DISTINCT
        mp.month_start_date AS trade_date,
        COALESCE(normalize_name_sql(gi.industry_name), 'Unclassified') AS group_name,
        COALESCE(cp.currency, 'USD') AS currency,
        mp.open,
        mp.high,
        mp.low,
        mp.close,
        mp.volume,
        1 AS cnt
    FROM calculated_metrics.monthly_prices mp
    LEFT JOIN reference.company_gics cg ON mp.symbol = cg.symbol
    LEFT JOIN reference.gics_industry gi ON cg.industry_id = gi.industry_id
    LEFT JOIN fmp_data.company_profile cp ON mp.symbol = cp.symbol
    WHERE mp.month_start_date >= %(from_date)s
) t
GROUP BY trade_date, group_name, currency
ON CONFLICT (trade_date, group_name, currency) DO UPDATE SET
    avg_open = EXCLUDED.avg_open,
    avg_high = EXCLUDED.avg_high,
    avg_low = EXCLUDED.avg_low,
    avg_close = EXCLUDED.avg_close,
    avg_volume = EXCLUDED.avg_volume,
    symbol_count = EXCLUDED.symbol_count
"""

# PostgreSQL正規化関数を定義
NORMALIZE_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION normalize_name_sql(s text) RETURNS text AS $$
BEGIN
    IF s IS NULL OR trim(s) = '' THEN
        RETURN 'Unclassified';
    END IF;
    
    -- 空白のトリムと連続する空白の置換
    s := regexp_replace(trim(s), '\s+', ' ', 'g');
    
    -- タイトルケースに変換
    RETURN initcap(s);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""

def ensure_normalize_function() -> None:
    """
    正規化関数がデータベースに存在することを確認します。
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(NORMALIZE_FUNCTION_SQL)
            conn.commit()
        logger.debug("正規化関数が確認されました")
    except Exception as e:
        logger.error(f"正規化関数の作成エラー: {str(e)}")
        raise

def refresh_sector_industry_daily(from_date: date) -> Dict[str, int]:
    """
    指定された日付以降のセクターとインダストリーの日次価格データを更新します。
    
    Args:
        from_date: 更新する開始日
        
    Returns:
        Dict[str, int]: 更新されたテーブルと行数の辞書
    """
    results = {}
    
    try:
        ensure_normalize_function()
        
        with get_connection() as conn:
            # セクター日次更新
            with conn.cursor() as cur:
                cur.execute(SECTOR_DAILY_UPSERT_QUERY, {'from_date': from_date})
                results['sector_daily'] = cur.rowcount
                logger.info(f"セクター日次価格を更新しました: {cur.rowcount} 行")
            
            # インダストリー日次更新
            with conn.cursor() as cur:
                cur.execute(INDUSTRY_DAILY_UPSERT_QUERY, {'from_date': from_date})
                results['industry_daily'] = cur.rowcount
                logger.info(f"インダストリー日次価格を更新しました: {cur.rowcount} 行")
            
            conn.commit()
    
    except Exception as e:
        logger.error(f"日次更新エラー: {str(e)}")
        raise
    
    return results

def refresh_sector_industry_weekly(from_date: date) -> Dict[str, int]:
    """
    指定された日付以降のセクターとインダストリーの週次価格データを更新します。
    
    Args:
        from_date: 更新する開始日
        
    Returns:
        Dict[str, int]: 更新されたテーブルと行数の辞書
    """
    results = {}
    
    try:
        ensure_normalize_function()
        
        with get_connection() as conn:
            # セクター週次更新
            with conn.cursor() as cur:
                cur.execute(SECTOR_WEEKLY_UPSERT_QUERY, {'from_date': from_date})
                results['sector_weekly'] = cur.rowcount
                logger.info(f"セクター週次価格を更新しました: {cur.rowcount} 行")
            
            # インダストリー週次更新
            with conn.cursor() as cur:
                cur.execute(INDUSTRY_WEEKLY_UPSERT_QUERY, {'from_date': from_date})
                results['industry_weekly'] = cur.rowcount
                logger.info(f"インダストリー週次価格を更新しました: {cur.rowcount} 行")
            
            conn.commit()
    
    except Exception as e:
        logger.error(f"週次更新エラー: {str(e)}")
        raise
    
    return results

def refresh_sector_industry_monthly(from_date: date) -> Dict[str, int]:
    """
    指定された日付以降のセクターとインダストリーの月次価格データを更新します。
    
    Args:
        from_date: 更新する開始日
        
    Returns:
        Dict[str, int]: 更新されたテーブルと行数の辞書
    """
    results = {}
    
    try:
        ensure_normalize_function()
        
        with get_connection() as conn:
            # セクター月次更新
            with conn.cursor() as cur:
                cur.execute(SECTOR_MONTHLY_UPSERT_QUERY, {'from_date': from_date})
                results['sector_monthly'] = cur.rowcount
                logger.info(f"セクター月次価格を更新しました: {cur.rowcount} 行")
            
            # インダストリー月次更新
            with conn.cursor() as cur:
                cur.execute(INDUSTRY_MONTHLY_UPSERT_QUERY, {'from_date': from_date})
                results['industry_monthly'] = cur.rowcount
                logger.info(f"インダストリー月次価格を更新しました: {cur.rowcount} 行")
            
            conn.commit()
    
    except Exception as e:
        logger.error(f"月次更新エラー: {str(e)}")
        raise
    
    return results 