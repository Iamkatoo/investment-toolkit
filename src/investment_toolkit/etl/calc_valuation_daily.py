#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日次バリュエーション指標計算ETL
- 日次価格データを取得
- 指標計算（return_6m, return_12m, high_52w_gap, vol_30d, beta_1y, amihud_illiquidity）
- 結果をvaluation_dailyテーブルにUPSERT
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 相対インポートをサポートするためにプロジェクトルートをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from investment_toolkit.database.db_manager import get_db_connection, DatabaseManager
from investment_toolkit.metrics.valuation_daily import compute_valuation_indicators
from investment_toolkit.utilities.logging_setup import setup_logger

# ログ設定
logger = setup_logger('calc_valuation_daily', logging.INFO)


def fetch_market_index_data(db_manager, start_date, market_symbol='^GSPC'):
    """
    市場指数のデータを取得する（デフォルトはS&P 500）
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        start_date (str): データ取得開始日 (YYYY-MM-DD)
        market_symbol (str): 市場指数のシンボル
        
    Returns:
        pd.DataFrame: 市場指数データ ['date', 'close']
    """
    query = """
    SELECT date, close 
    FROM fmp_data.daily_prices 
    WHERE symbol = %s 
      AND date >= %s
    ORDER BY date
    """
    
    logger.info(f"市場指数データ取得中: {market_symbol} (開始日: {start_date})")
    
    try:
        market_data = db_manager.fetchall(query, (market_symbol, start_date))
        
        if not market_data:
            logger.warning(f"市場指数データが見つかりません: {market_symbol}")
            return pd.DataFrame(columns=['date', 'close'])
        
        # リスト形式の結果をDataFrameに変換
        df_market = pd.DataFrame(market_data, columns=['date', 'close'])
        logger.info(f"市場指数データ取得完了: {len(df_market)}行")
        
        return df_market
        
    except Exception as e:
        logger.error(f"市場指数データ取得エラー: {e}")
        return pd.DataFrame(columns=['date', 'close'])


def fetch_daily_prices(db_manager, start_date, symbols=None):
    """
    日次価格データを取得
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        start_date (str): データ取得開始日 (YYYY-MM-DD)
        symbols (list): 対象銘柄のリスト（Noneの場合は全銘柄）
        
    Returns:
        pd.DataFrame: 日次価格データ ['symbol', 'date', 'close', 'volume']
    """
    if symbols:
        # 指定銘柄のみを取得
        placeholder = ','.join(['%s'] * len(symbols))
        query = f"""
        SELECT symbol, date, close, volume 
        FROM fmp_data.daily_prices 
        WHERE symbol IN ({placeholder}) 
          AND date >= %s
        ORDER BY symbol, date
        """
        params = symbols + [start_date]
    else:
        # 全銘柄を取得
        query = """
        SELECT symbol, date, close, volume 
        FROM fmp_data.daily_prices 
        WHERE date >= %s
        ORDER BY symbol, date
        """
        params = (start_date,)
    
    logger.info(f"日次価格データ取得中 (開始日: {start_date}, "
                f"対象銘柄: {'全銘柄' if not symbols else len(symbols)}銘柄)")
    
    try:
        # データ取得と変換
        price_data = db_manager.fetchall(query, params)
        
        if not price_data:
            logger.warning("対象期間の価格データが見つかりません")
            return pd.DataFrame(columns=['symbol', 'date', 'close', 'volume'])
        
        df_prices = pd.DataFrame(price_data, columns=['symbol', 'date', 'close', 'volume'])
        logger.info(f"日次価格データ取得完了: {len(df_prices)}行, {df_prices['symbol'].nunique()}銘柄")
        
        return df_prices
        
    except Exception as e:
        logger.error(f"日次価格データ取得エラー: {e}")
        return pd.DataFrame(columns=['symbol', 'date', 'close', 'volume'])


def upsert_valuation_daily(db_manager, df):
    """
    計算した指標をvaluation_dailyテーブルにUPSERT
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        df (pd.DataFrame): 計算済み指標データフレーム
        
    Returns:
        bool: 成功した場合はTrue
    """
    if df.empty:
        logger.warning("データがないため、UPSERTをスキップします")
        return False
    
    # 必要なカラムの存在確認
    required_columns = ['symbol', 'date', 'price', 'return_6m', 'return_12m', 
                        'high_52w_gap', 'vol_30d', 'beta_1y', 'amihud_illiquidity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"必要なカラムが不足しています: {', '.join(missing_columns)}")
        return False
    
    # NaN値をNoneに変換
    for col in df.columns:
        if col not in ['symbol', 'date']:
            df[col] = df[col].where(pd.notna(df[col]), None)
    
    # トランザクション開始
    try:
        logger.info(f"valuation_dailyテーブルへのUPSERT開始: {len(df)}行")
        cursor = db_manager.conn.cursor()
        
        # バッチサイズ
        batch_size = 1000
        batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
        
        total_processed = 0
        
        for batch_df in batches:
            # データ作成
            values = []
            for _, row in batch_df.iterrows():
                values.append((
                    row['symbol'],
                    row['date'],
                    row.get('price'),
                    row.get('return_6m'),
                    row.get('return_12m'),
                    row.get('high_52w_gap'),
                    row.get('vol_30d'),
                    row.get('beta_1y'),
                    row.get('amihud_illiquidity')
                ))
            
            # UPSERT SQL
            upsert_sql = """
            INSERT INTO calculated_metrics.valuation_daily
              (symbol, date, price, return_6m, return_12m, high_52w_gap, 
               vol_30d, beta_1y, amihud_illiquidity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date)
            DO UPDATE SET
              price = EXCLUDED.price,
              return_6m = EXCLUDED.return_6m,
              return_12m = EXCLUDED.return_12m,
              high_52w_gap = EXCLUDED.high_52w_gap,
              vol_30d = EXCLUDED.vol_30d,
              beta_1y = EXCLUDED.beta_1y,
              amihud_illiquidity = EXCLUDED.amihud_illiquidity;
            """
            
            cursor.executemany(upsert_sql, values)
            total_processed += len(batch_df)
            logger.info(f"バッチ処理完了: {total_processed}/{len(df)}行")
        
        # コミット
        db_manager.commit()
        logger.info(f"valuation_dailyテーブルへのUPSERT完了: {total_processed}行")
        return True
        
    except Exception as e:
        db_manager.rollback()
        logger.error(f"valuation_dailyテーブルへのUPSERTエラー: {e}")
        return False


def main(backfill=False, days_back=None, symbols=None):
    """
    メイン処理
    
    Args:
        backfill (bool): バックフィルモードかどうか
        days_back (int): 計算対象日数（Noneの場合はデフォルト値を使用）
        symbols (list): 対象銘柄のリスト（Noneの場合は全銘柄）
    
    Returns:
        bool: 成功した場合はTrue
    """
    # 計算対象期間の設定
    if backfill:
        # バックフィルモード: 過去2年
        lookback_days = 730
        logger.info(f"バックフィルモード: 過去{lookback_days}日間のデータを処理します")
    elif days_back:
        # 指定日数
        lookback_days = days_back
        logger.info(f"指定日数モード: 過去{lookback_days}日間のデータを処理します")
    else:
        # デフォルト: 前日分のみ
        lookback_days = 1
        logger.info("日次更新モード: 前日分のみ処理します")
    
    # 開始日を計算
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    # 指標計算に必要な過去データを含めるため、さらに1年分遡る
    fetch_start_date = (datetime.now() - timedelta(days=lookback_days + 365)).strftime('%Y-%m-%d')
    
    # データベース接続
    try:
        with DatabaseManager() as db_manager:
            # 市場指数データ取得
            df_market = fetch_market_index_data(db_manager, fetch_start_date)
            
            # 日次価格データ取得
            df_prices = fetch_daily_prices(db_manager, fetch_start_date, symbols)
            
            if df_prices.empty:
                logger.error("取得した価格データが空のため、処理を中止します")
                return False
            
            # バリュエーション指標計算
            df_valuation = compute_valuation_indicators(df_prices, df_market)
            
            # 計算対象期間のデータだけにフィルタリング
            # (過去1年分は計算用に取得したが、結果はstart_date以降のみを保存)
            df_valuation = df_valuation[df_valuation['date'] >= start_date]
            
            if df_valuation.empty:
                logger.warning(f"{start_date}以降の計算結果が空のため、UPSERTをスキップします")
                return False
            
            # 計算結果をDBに保存
            result = upsert_valuation_daily(db_manager, df_valuation)
            
            return result
            
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='日次バリュエーション指標計算ETL')
    parser.add_argument('--backfill', action='store_true',
                        help='過去2年分のデータを処理する（バックフィルモード）')
    parser.add_argument('--days-back', type=int,
                        help='指定した日数分の過去データを処理する')
    parser.add_argument('--symbols', nargs='+',
                        help='対象とする銘柄のリスト（例: AAPL MSFT GOOGL）')
    
    args = parser.parse_args()
    
    success = main(
        backfill=args.backfill,
        days_back=args.days_back,
        symbols=args.symbols
    )
    
    if success:
        logger.info("処理が正常に完了しました")
        sys.exit(0)
    else:
        logger.error("処理が異常終了しました")
        sys.exit(1) 