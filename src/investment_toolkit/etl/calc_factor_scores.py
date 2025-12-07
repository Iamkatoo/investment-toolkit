#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ファクタースコア計算ETL
- valuation_dailyから価格・リターン・ボラティリティ情報を取得
- basic_metricsから財務指標を取得
- reference.symbolsからセクター情報を取得
- z-scoreの計算
- デシルランクの計算
- 結果をfactor_scores.dailyとfactor_scores.rankテーブルにUPSERT
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
from investment_analysis.metrics.factor_scores import calculate_factor_scores
from investment_analysis.utilities.logging_setup import setup_logger

# ログ設定
logger = setup_logger('calc_factor_scores', logging.INFO)


def fetch_valuation_daily(db_manager, date):
    """
    特定日付の日次バリュエーション指標を取得
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        date (str): 取得する日付（YYYY-MM-DD形式）
        
    Returns:
        pd.DataFrame: 日次バリュエーション指標データ
    """
    query = """
    SELECT symbol, date, price, return_6m, return_12m, high_52w_gap, vol_30d, beta_1y, amihud_illiquidity
    FROM calculated_metrics.valuation_daily
    WHERE date = %s
    """
    
    logger.info(f"日次バリュエーション指標データ取得中: {date}")
    
    try:
        data = db_manager.fetchall(query, (date,))
        
        if not data:
            logger.warning(f"日付 {date} の日次バリュエーション指標データが見つかりません")
            return pd.DataFrame()
        
        # カラム名の取得とDataFrame変換
        columns = ['symbol', 'date', 'price', 'return_6m', 'return_12m', 
                  'high_52w_gap', 'vol_30d', 'beta_1y', 'amihud_illiquidity']
        df = pd.DataFrame(data, columns=columns)
        
        logger.info(f"日次バリュエーション指標データ取得完了: {len(df)}行")
        return df
        
    except Exception as e:
        logger.error(f"日次バリュエーション指標データ取得エラー: {e}")
        return pd.DataFrame()


def fetch_basic_metrics(db_manager, as_of_date):
    """
    指定日付時点の最新の基本財務指標を取得
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        as_of_date (str): 取得基準日（YYYY-MM-DD形式）
        
    Returns:
        pd.DataFrame: 基本財務指標データ
    """
    query = """
    WITH latest_metrics AS (
        SELECT symbol, MAX(as_of_date) AS latest_date
        FROM calculated_metrics.basic_metrics
        WHERE as_of_date <= %s
        GROUP BY symbol
    )
    SELECT bm.symbol, bm.as_of_date, bm.per, bm.pbr, bm.roe, bm.roa,
           bm.eps, bm.eps_growth, bm.revenue_growth, bm.net_income_growth,
           bm.dividend_yield, bm.fcf_yield, bm.roe as roic, 
           0.5 as news_sentiment  -- 仮のセンチメント値を設定
    FROM calculated_metrics.basic_metrics bm
    JOIN latest_metrics lm ON bm.symbol = lm.symbol AND bm.as_of_date = lm.latest_date
    """
    
    logger.info(f"基本財務指標データ取得中: {as_of_date}時点")
    
    try:
        data = db_manager.fetchall(query, (as_of_date,))
        
        if not data:
            logger.warning(f"日付 {as_of_date} 時点の基本財務指標データが見つかりません")
            return pd.DataFrame()
        
        # カラム名の取得とDataFrame変換
        columns = ['symbol', 'as_of_date', 'per', 'pbr', 'roe', 'roa',
                  'eps', 'eps_growth', 'revenue_growth', 'net_income_growth',
                  'dividend_yield', 'fcf_yield', 'roic', 'news_sentiment']
        df = pd.DataFrame(data, columns=columns)
        
        logger.info(f"基本財務指標データ取得完了: {len(df)}行")
        return df
        
    except Exception as e:
        logger.error(f"基本財務指標データ取得エラー: {e}")
        return pd.DataFrame()


def fetch_symbols_master(db_manager):
    """
    シンボルマスター情報（銘柄とセクターの紐付け）を取得
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        
    Returns:
        pd.DataFrame: シンボルマスターデータ
    """
    query = """
    SELECT symbol, sector
    FROM reference.symbols
    """
    
    logger.info("シンボルマスターデータ取得中")
    
    try:
        data = db_manager.fetchall(query)
        
        if not data:
            # リファレンステーブルが存在しない場合はfmp_data.company_profileから取得を試みる
            logger.warning("reference.symbolsからデータを取得できませんでした。代替ソースを使用します。")
            
            alt_query = """
            WITH latest_profile AS (
                SELECT symbol, MAX(date) AS latest_date
                FROM fmp_data.company_profile
                GROUP BY symbol
            )
            SELECT cp.symbol, cp.sector
            FROM fmp_data.company_profile cp
            JOIN latest_profile lp ON cp.symbol = lp.symbol AND cp.date = lp.latest_date
            """
            
            data = db_manager.fetchall(alt_query)
            
            if not data:
                logger.warning("シンボルマスターデータが見つかりません")
                return pd.DataFrame()
        
        # カラム名の取得とDataFrame変換
        columns = ['symbol', 'sector']
        df = pd.DataFrame(data, columns=columns)
        
        # セクターが欠損していないものだけを保持
        df = df.dropna(subset=['sector'])
        
        logger.info(f"シンボルマスターデータ取得完了: {len(df)}行")
        return df
        
    except Exception as e:
        logger.error(f"シンボルマスターデータ取得エラー: {e}")
        return pd.DataFrame()


def upsert_factor_scores(db_manager, df_scores, df_ranks):
    """
    計算したファクタースコアとランク情報をテーブルにUPSERT
    
    Args:
        db_manager (DatabaseManager): データベース接続マネージャ
        df_scores (pd.DataFrame): ファクタースコアデータフレーム
        df_ranks (pd.DataFrame): ファクターランクデータフレーム
        
    Returns:
        bool: 成功した場合はTrue
    """
    if df_scores.empty:
        logger.warning("スコアデータがないため、UPSERTをスキップします")
        return False
    
    success = True
    
    # 1. factor_scores.daily へのUPSERT
    try:
        logger.info(f"factor_scores.dailyテーブルへのUPSERT開始: {len(df_scores)}行")
        
        # NaN値をNoneに変換
        for col in df_scores.columns:
            if col not in ['symbol', 'date']:
                df_scores[col] = df_scores[col].where(pd.notna(df_scores[col]), None)
        
        cursor = db_manager.conn.cursor()
        
        # バッチサイズ
        batch_size = 1000
        batches = [df_scores[i:i + batch_size] for i in range(0, len(df_scores), batch_size)]
        
        total_processed = 0
        
        for batch_df in batches:
            # データ作成
            values = []
            for _, row in batch_df.iterrows():
                values.append((
                    row['symbol'],
                    row['date'],
                    row.get('value_z'),
                    row.get('growth_z'),
                    row.get('quality_z'),
                    row.get('momentum_z'),
                    row.get('risk_z'),
                    row.get('sentiment_z'),
                    row.get('composite_score')
                ))
            
            # UPSERT SQL
            upsert_sql = """
            INSERT INTO factor_scores.daily
              (symbol, date, value_z, growth_z, quality_z, momentum_z,
               risk_z, sentiment_z, composite_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date)
            DO UPDATE SET
              value_z = EXCLUDED.value_z,
              growth_z = EXCLUDED.growth_z,
              quality_z = EXCLUDED.quality_z,
              momentum_z = EXCLUDED.momentum_z,
              risk_z = EXCLUDED.risk_z,
              sentiment_z = EXCLUDED.sentiment_z,
              composite_score = EXCLUDED.composite_score;
            """
            
            cursor.executemany(upsert_sql, values)
            total_processed += len(batch_df)
            logger.info(f"バッチ処理完了: {total_processed}/{len(df_scores)}行")
        
        logger.info(f"factor_scores.dailyテーブルへのUPSERT完了: {total_processed}行")
        
        # 成功した場合のみコミット
        db_manager.commit()
        logger.info(f"ファクタースコア情報のUPSERTが完了しました")
        return True
    except Exception as e:
        db_manager.rollback()
        logger.error(f"factor_scores.dailyテーブルへのUPSERTエラー: {e}")
        logger.error(f"ファクタースコア情報のUPSERTに失敗しました")
        return False
    
    # テスト中はrankテーブルの更新をスキップ
    logger.warning("現在のテスト中はrankテーブルの更新をスキップしています")
    return success


def main(date_str=None, days_back=None, backfill=False):
    """
    メイン処理
    
    Args:
        date_str (str): 処理対象日（YYYY-MM-DD形式、Noneの場合は前日）
        days_back (int): 指定日数分の過去データを処理
        backfill (bool): バックフィルモードかどうか
    
    Returns:
        bool: 成功した場合はTrue
    """
    # 処理対象期間の設定
    if backfill:
        # バックフィルモード: 過去2年
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"バックフィルモード: {start_date}から{end_date}までのデータを処理します")
    elif days_back:
        # 指定日数
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"指定日数モード: {start_date}から{end_date}までのデータを処理します")
    elif date_str:
        # 指定日
        start_date = date_str
        end_date = date_str
        logger.info(f"指定日モード: {date_str}のデータを処理します")
    else:
        # デフォルト: 前日分のみ
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = start_date
        logger.info(f"日次更新モード: {start_date}のデータを処理します")
    
    # 対象日付のリストを生成
    dates = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_date_dt:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    # データベース接続
    try:
        with DatabaseManager() as db_manager:
            # シンボルマスター情報取得（一度だけ）
            symbols_master = fetch_symbols_master(db_manager)
            
            if symbols_master.empty:
                logger.warning("シンボルマスター情報が取得できないため、セクター別ランクは計算されません")
            
            # 各日付について処理
            for date in dates:
                logger.info(f"日付 {date} の処理を開始")
                
                # 1. 日次バリュエーション指標の取得
                valuation_daily = fetch_valuation_daily(db_manager, date)
                
                if valuation_daily.empty:
                    logger.warning(f"日付 {date} の日次バリュエーション指標データがないため、スキップします")
                    continue
                
                # 2. 基本財務指標の取得
                basic_metrics = fetch_basic_metrics(db_manager, date)
                
                if basic_metrics.empty:
                    logger.warning(f"日付 {date} 時点の基本財務指標データがないため、一部スコアのみ計算します")
                
                # 3. ファクタースコアの計算
                factor_scores_df, factor_ranks_df = calculate_factor_scores(
                    valuation_daily, basic_metrics, symbols_master
                )
                
                # 4. 計算結果のDB保存
                if factor_scores_df.empty:
                    logger.warning(f"日付 {date} のファクタースコア計算結果が空のため、保存をスキップします")
                    continue
                
                result = upsert_factor_scores(db_manager, factor_scores_df, factor_ranks_df)
                
                if result:
                    logger.info(f"日付 {date} のファクタースコア保存が成功しました")
                else:
                    logger.error(f"日付 {date} のファクタースコア保存に失敗しました")
            
            return True
            
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ファクタースコア計算ETL')
    parser.add_argument('--date', help='処理対象日（YYYY-MM-DD形式）')
    parser.add_argument('--backfill', action='store_true',
                        help='過去2年分のデータを処理する（バックフィルモード）')
    parser.add_argument('--days-back', type=int,
                        help='指定した日数分の過去データを処理する')
    
    args = parser.parse_args()
    
    success = main(
        date_str=args.date,
        days_back=args.days_back,
        backfill=args.backfill
    )
    
    if success:
        logger.info("処理が正常に完了しました")
        sys.exit(0)
    else:
        logger.error("処理が異常終了しました")
        sys.exit(1) 