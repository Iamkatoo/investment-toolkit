#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
時価総額（market_cap）計算ETL
- 日次価格データを取得
- 株式数データを取得
- 時価総額を計算（価格 × 発行済み株式数）
- 結果をbasic_metricsテーブルにUPDATE
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

from investment_analysis.database.db_manager import get_db_connection, DatabaseManager
from investment_analysis.utilities.logging_setup import setup_logger

# ログ設定
logger = setup_logger('calc_market_cap', logging.INFO)


def fetch_daily_prices(db_manager, as_of_date, symbols=None):
    """
    指定日の日次価格データを取得
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        as_of_date (str): 指定日付 (YYYY-MM-DD)
        symbols (list): 対象銘柄のリスト（Noneの場合は全銘柄）
        
    Returns:
        pd.DataFrame: 日次価格データ ['symbol', 'date', 'close']
    """
    if symbols:
        # 指定銘柄のみを取得
        placeholder = ','.join(['%s'] * len(symbols))
        query = f"""
        SELECT symbol, date, close 
        FROM fmp_data.daily_prices 
        WHERE symbol IN ({placeholder}) 
          AND date = %s
        """
        params = symbols + [as_of_date]
    else:
        # 全銘柄を取得
        query = """
        SELECT symbol, date, close 
        FROM fmp_data.daily_prices 
        WHERE date = %s
        """
        params = (as_of_date,)
    
    logger.info(f"日次価格データ取得中 (指定日: {as_of_date}, "
                f"対象銘柄: {'全銘柄' if not symbols else len(symbols)}銘柄)")
    
    try:
        # データ取得と変換
        price_data = db_manager.fetchall(query, params)
        
        if not price_data:
            logger.warning(f"指定日 {as_of_date} の価格データが見つかりません")
            return pd.DataFrame(columns=['symbol', 'date', 'close'])
        
        df_prices = pd.DataFrame(price_data, columns=['symbol', 'date', 'close'])
        logger.info(f"日次価格データ取得完了: {len(df_prices)}行, {df_prices['symbol'].nunique()}銘柄")
        
        return df_prices
        
    except Exception as e:
        logger.error(f"日次価格データ取得エラー: {e}")
        return pd.DataFrame(columns=['symbol', 'date', 'close'])


def fetch_shares_data(db_manager, symbols=None):
    """
    最新の株式数データを取得
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        symbols (list): 対象銘柄のリスト（Noneの場合は全銘柄）
        
    Returns:
        pd.DataFrame: 株式数データ ['symbol', 'outstanding_shares']
    """
    if symbols:
        # 指定銘柄のみを取得
        placeholder = ','.join(['%s'] * len(symbols))
        query = f"""
        WITH latest_shares AS (
            SELECT symbol, MAX(date) as latest_date
            FROM fmp_data.shares
            WHERE symbol IN ({placeholder})
            GROUP BY symbol
        )
        SELECT s.symbol, s.outstanding_shares
        FROM fmp_data.shares s
        JOIN latest_shares ls ON s.symbol = ls.symbol AND s.date = ls.latest_date
        """
        params = symbols
    else:
        # 全銘柄を取得
        query = """
        WITH latest_shares AS (
            SELECT symbol, MAX(date) as latest_date
            FROM fmp_data.shares
            GROUP BY symbol
        )
        SELECT s.symbol, s.outstanding_shares
        FROM fmp_data.shares s
        JOIN latest_shares ls ON s.symbol = ls.symbol AND s.date = ls.latest_date
        """
        params = None
    
    logger.info(f"株式数データ取得中 (対象銘柄: {'全銘柄' if not symbols else len(symbols)}銘柄)")
    
    try:
        # データ取得と変換
        if params:
            shares_data = db_manager.fetchall(query, params)
        else:
            shares_data = db_manager.fetchall(query)
        
        if not shares_data:
            logger.warning("株式数データが見つかりません")
            return pd.DataFrame(columns=['symbol', 'outstanding_shares'])
        
        df_shares = pd.DataFrame(shares_data, columns=['symbol', 'outstanding_shares'])
        logger.info(f"株式数データ取得完了: {len(df_shares)}行, {df_shares['symbol'].nunique()}銘柄")
        
        return df_shares
        
    except Exception as e:
        logger.error(f"株式数データ取得エラー: {e}")
        return pd.DataFrame(columns=['symbol', 'outstanding_shares'])


def calculate_market_cap(df_prices, df_shares):
    """
    時価総額を計算
    
    Args:
        df_prices (pd.DataFrame): 価格データ ['symbol', 'date', 'close']
        df_shares (pd.DataFrame): 株式数データ ['symbol', 'outstanding_shares']
        
    Returns:
        pd.DataFrame: 時価総額データ ['symbol', 'date', 'market_cap']
    """
    logger.info("時価総額計算を開始します")
    
    try:
        # 価格データと株式数データをマージ
        df = pd.merge(df_prices, df_shares, on='symbol', how='inner')
        logger.info(f"価格と株式数のマージ結果: {len(df)}行")
        
        # 時価総額を計算（0での除算を避ける）
        mask = (df['close'] > 0) & (df['outstanding_shares'] > 0)
        df['market_cap'] = np.nan  # 初期値をNaNに設定
        df.loc[mask, 'market_cap'] = df.loc[mask, 'close'] * df.loc[mask, 'outstanding_shares']
        
        logger.info(f"時価総額計算完了: {df['market_cap'].notna().sum()}銘柄で有効な値")
        
        # 必要なカラムだけを返す
        result_df = df[['symbol', 'date', 'market_cap']]
        
        return result_df
        
    except Exception as e:
        logger.error(f"時価総額計算エラー: {e}")
        return pd.DataFrame(columns=['symbol', 'date', 'market_cap'])


def update_basic_metrics(db_manager, market_cap_df):
    """
    calculated_metrics.basic_metricsテーブルのmarket_capを更新
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        market_cap_df (pd.DataFrame): 時価総額データ ['symbol', 'date', 'market_cap']
        
    Returns:
        bool: 成功した場合はTrue
    """
    if market_cap_df.empty:
        logger.warning("データがないため、更新をスキップします")
        return False
    
    logger.info(f"basic_metricsテーブルのmarket_cap更新を開始: {len(market_cap_df)}行")
    
    try:
        cursor = db_manager.conn.cursor()
        
        # バッチサイズ
        batch_size = 500
        batches = [market_cap_df[i:i + batch_size] for i in range(0, len(market_cap_df), batch_size)]
        
        total_updated = 0
        
        for batch_df in batches:
            for _, row in batch_df.iterrows():
                symbol = row['symbol']
                as_of_date = row['date']
                market_cap = row['market_cap'] if pd.notna(row['market_cap']) else None
                
                if market_cap is None:
                    continue
                
                # market_capを更新
                update_query = """
                UPDATE calculated_metrics.basic_metrics
                SET market_cap = %s
                WHERE symbol = %s AND as_of_date = %s
                """
                
                cursor.execute(update_query, (market_cap, symbol, as_of_date))
                total_updated += cursor.rowcount
            
            # 各バッチ後にコミット
            db_manager.commit()
            logger.info(f"バッチ処理完了: 合計{total_updated}行を更新")
        
        logger.info(f"basic_metricsテーブルの更新完了: {total_updated}行を更新")
        return True
        
    except Exception as e:
        db_manager.rollback()
        logger.error(f"basic_metricsテーブルの更新エラー: {e}")
        return False


def process_single_symbol(db_manager, symbol, as_of_date):
    """
    1銘柄の時価総額を計算して更新
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        symbol (str): 銘柄コード
        as_of_date (str): 指定日付 (YYYY-MM-DD)
        
    Returns:
        bool: 成功した場合はTrue
    """
    logger.info(f"銘柄 {symbol} の時価総額計算を開始 (対象日: {as_of_date})")
    
    # 日次価格データを取得
    df_prices = fetch_daily_prices(db_manager, as_of_date, [symbol])
    if df_prices.empty:
        logger.error(f"銘柄 {symbol} の価格データが取得できませんでした")
        return False
    
    # 株式数データを取得
    df_shares = fetch_shares_data(db_manager, [symbol])
    if df_shares.empty:
        logger.error(f"銘柄 {symbol} の株式数データが取得できませんでした")
        return False
    
    # 時価総額を計算
    market_cap_df = calculate_market_cap(df_prices, df_shares)
    if market_cap_df.empty:
        logger.error(f"銘柄 {symbol} の時価総額計算に失敗しました")
        return False
    
    # basic_metricsテーブルを更新
    success = update_basic_metrics(db_manager, market_cap_df)
    
    if success:
        logger.info(f"銘柄 {symbol} の時価総額更新が完了しました")
    else:
        logger.error(f"銘柄 {symbol} の時価総額更新に失敗しました")
    
    return success


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='時価総額（market_cap）を計算してbasic_metricsテーブルを更新します')
    parser.add_argument('--symbol', type=str, help='特定の銘柄コード (指定しない場合は全銘柄を処理)')
    parser.add_argument('--date', type=str, help='対象日 (YYYY-MM-DD形式, 指定しない場合はbasic_metricsの最新日)')
    args = parser.parse_args()
    
    try:
        with DatabaseManager() as db_manager:
            # 対象日を決定
            if args.date:
                as_of_date = args.date
            else:
                # basic_metricsテーブルの最新日を取得
                latest_date_query = """
                SELECT MAX(as_of_date) as latest_date
                FROM calculated_metrics.basic_metrics
                """
                latest_date = db_manager.fetchone(latest_date_query)[0]
                
                if not latest_date:
                    logger.error("basic_metricsテーブルに日付が見つかりません")
                    return False
                
                as_of_date = latest_date.strftime('%Y-%m-%d')
            
            logger.info(f"対象日: {as_of_date}")
            
            # 特定銘柄のみ処理
            if args.symbol:
                return process_single_symbol(db_manager, args.symbol, as_of_date)
            
            # 全銘柄の処理
            # basic_metricsテーブルの対象日のレコードを取得
            symbols_query = """
            SELECT symbol
            FROM calculated_metrics.basic_metrics
            WHERE as_of_date = %s
            """
            symbols_data = db_manager.fetchall(symbols_query, (as_of_date,))
            
            if not symbols_data:
                logger.error(f"日付 {as_of_date} のbasic_metricsレコードが見つかりません")
                return False
            
            # シンボルのリストを抽出
            symbols = [row[0] for row in symbols_data]
            logger.info(f"対象銘柄数: {len(symbols)}")
            
            # 日次価格データを取得
            df_prices = fetch_daily_prices(db_manager, as_of_date)
            if df_prices.empty:
                logger.error("価格データが取得できませんでした")
                return False
            
            # 株式数データを取得
            df_shares = fetch_shares_data(db_manager)
            if df_shares.empty:
                logger.error("株式数データが取得できませんでした")
                return False
            
            # 時価総額を計算
            market_cap_df = calculate_market_cap(df_prices, df_shares)
            if market_cap_df.empty:
                logger.error("時価総額計算に失敗しました")
                return False
            
            # basic_metricsテーブルを更新
            success = update_basic_metrics(db_manager, market_cap_df)
            
            if success:
                logger.info("時価総額更新が完了しました")
            else:
                logger.error("時価総額更新に失敗しました")
            
            return success
            
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 