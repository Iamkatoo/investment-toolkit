#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
オリジナル総合スコア計算スクリプト（最適化版）

処理速度とメモリ効率を大幅に改善したバージョン：
- バッチ処理による段階的データ読み込み
- パーセンタイル計算の簡素化
- 定期的なデータベース保存
- メモリ使用量の削減
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from sqlalchemy import create_engine, text
from scipy import stats

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backtest_optimized_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OptimizedScoreCalculator:
    """最適化されたオリジナル総合スコア計算クラス"""
    
    def __init__(self, config_path: str = "config/score_weights.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.engine = self._create_db_engine()
        self.batch_size = 100  # 1000から100に削減
        self.save_interval = 10  # 何日ごとに保存するか
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            raise
    
    def _create_db_engine(self) -> Any:
        """データベース接続エンジンを作成"""
        db_name = os.environ.get("DB_NAME", "investment")
        db_user = os.environ.get("DB_USER", "HOME")  # macOSのデフォルトユーザー
        db_password = os.environ.get("DB_PASSWORD", "")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        
        if db_password:
            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            connection_string = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"
        
        return create_engine(connection_string)
    
    def get_date_range(self, start_date: str = "2015-01-01", end_date: Optional[str] = None) -> List[str]:
        """
        処理対象の日付リストを取得
        
        Args:
            start_date: 開始日
            end_date: 終了日（Noneの場合は最新日まで）
            
        Returns:
            日付文字列のリスト
        """
        query = """
        SELECT DISTINCT date 
        FROM backtest_results.vw_daily_master
        WHERE date >= %(start_date)s
        """
        params = {'start_date': start_date}
        
        if end_date:
            query += " AND date <= %(end_date)s"
            params['end_date'] = end_date
            
        query += " ORDER BY date"
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            # 日付列が既に文字列の場合とdatetimeの場合を処理
            if df['date'].dtype == 'object':
                # 既に文字列の場合
                return df['date'].tolist()
            else:
                # datetimeの場合
                return df['date'].dt.strftime('%Y-%m-%d').tolist()
        except Exception as e:
            logger.error(f"日付範囲取得エラー: {e}")
            raise
    
    def load_batch_data(self, dates: List[str]) -> pd.DataFrame:
        """
        指定された日付範囲のデータを読み込み（最適化版）
        
        Args:
            dates: 対象日付のリスト
            
        Returns:
            バッチデータのDataFrame
        """
        if not dates:
            return pd.DataFrame()
        
        # 必要な列のみ選択してメモリ使用量を削減
        query = """
        SELECT 
            symbol, date, close, market_cap,
            per, roic, roe, eps_cagr_3y, eps_cagr_5y, revenue_cagr_3y, revenue_cagr_5y,
            debt_to_equity, cfo_to_net_income,
            sma_20, sma_40, rsi_14, atr_14, macd_hist,
            raw_industry, sector_id,
            vix_value, dgs10_value, dgs2_value, cpi_value, ffr_value
        FROM backtest_results.vw_daily_master
        WHERE date = ANY(%(dates)s)
        ORDER BY date, symbol
        """
        
        try:
            df = pd.read_sql(query, self.engine, params={'dates': dates})
            logger.info(f"バッチデータ読み込み完了: {len(df):,}行, 期間: {dates[0]} ～ {dates[-1]}")
            
            # デバッグ: 日付列の情報を出力
            if len(df) > 0:
                logger.info(f"日付列の型: {df['date'].dtype}")
                logger.info(f"日付の例: {df['date'].head().tolist()}")
                logger.info(f"ユニークな日付: {sorted(df['date'].unique())}")
            
            return df
        except Exception as e:
            logger.error(f"バッチデータ読み込みエラー: {e}")
            raise
    
    def calculate_simple_percentiles(self, df: pd.DataFrame, date: str) -> Dict[str, float]:
        """
        簡素化されたパーセンタイル計算（速度重視）
        
        Args:
            df: データフレーム
            date: 計算対象日
            
        Returns:
            各指標の平均パーセンタイル
        """
        # 当日のデータを取得
        if df['date'].dtype == 'object':
            # datetime.dateオブジェクトの場合は文字列に変換して比較
            target_df = df[df['date'].astype(str) == date].copy()
        else:
            target_df = df[df['date'].dt.strftime('%Y-%m-%d') == date].copy()
        
        if len(target_df) == 0:
            logger.warning(f"パーセンタイル計算: 日付 {date} のデータが見つかりません")
            return {}
        
        # 過去30日分のデータのみ使用（252日から大幅短縮）
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=30)
        
        # 日付列の型を統一
        if df['date'].dtype == 'object':
            # datetime.dateオブジェクトの場合はdatetimeに変換
            df_dates = pd.to_datetime(df['date'])
        else:
            df_dates = df['date']
        
        period_df = df[
            (df_dates >= start_date) & 
            (df_dates <= end_date)
        ].copy()
        
        percentiles = {}
        
        # 簡素化されたパーセンタイル計算（業界別は省略し、全体のみ）
        for col in ['per', 'roic', 'roe', 'eps_cagr_3y', 'revenue_cagr_3y', 'rsi_14']:
            if col in target_df.columns and col in period_df.columns:
                period_data = period_df[col].dropna()
                target_data = target_df[col].dropna()
                
                if len(period_data) > 10 and len(target_data) > 0:
                    # 平均パーセンタイルを計算（個別計算は省略）
                    target_median = target_data.median()
                    percentile = (period_data <= target_median).mean()
                    percentiles[f'{col}_percentile'] = percentile
                else:
                    percentiles[f'{col}_percentile'] = 0.5  # デフォルト値
        
        return percentiles
    
    def calculate_simplified_scores(self, row: pd.Series, percentiles: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        簡素化されたスコア計算（速度重視）
        
        Args:
            row: データ行
            percentiles: パーセンタイル情報
            
        Returns:
            (総合スコア, 詳細スコア辞書)
        """
        weights = self.config['scoring_weights']
        scores = {}
        
        # Value Score (20点)
        per_percentile = percentiles.get('per_percentile', 0.5)
        roic_percentile = percentiles.get('roic_percentile', 0.5)
        scores['value_score'] = (
            (1 - per_percentile) * 8 +  # PER反転
            roic_percentile * 12  # ROIC
        )
        
        # Growth Score (20点)
        eps_3y = row.get('eps_cagr_3y', 0) or 0
        eps_5y = row.get('eps_cagr_5y', 0) or 0
        rev_3y = row.get('revenue_cagr_3y', 0) or 0
        
        eps_score = min(10, max(0, eps_3y * 0.5)) if eps_3y > 0 else 0
        rev_score = min(5, max(0, rev_3y * 0.33)) if rev_3y > 0 else 0
        consistency_score = 5 if abs(eps_5y - eps_3y) < 5 else 0
        
        scores['growth_score'] = eps_score + rev_score + consistency_score
        
        # Quality Score (25点)
        roic = row.get('roic', 0) or 0
        roe = row.get('roe', 0) or 0
        debt_equity = row.get('debt_to_equity', 0) or 0
        
        roic_score = min(10, max(0, roic * 0.33))
        roe_score = min(5, max(0, roe * 0.2))
        debt_score = max(0, 5 * (1 - min(1, debt_equity / 0.5))) if debt_equity > 0 else 5
        quality_bonus = 5 if roic > 15 and roe > 15 else 0
        
        scores['quality_score'] = roic_score + roe_score + debt_score + quality_bonus
        
        # Momentum Score (20点)
        sma_20 = row.get('sma_20', 0) or 0
        sma_40 = row.get('sma_40', 0) or 0
        rsi = row.get('rsi_14', 50) or 50
        close = row.get('close', 0) or 0
        
        golden_cross = 5 if sma_20 > sma_40 and sma_20 > 0 else 0
        rsi_score = 3 if 50 <= rsi <= 70 else (1 if 70 < rsi <= 80 else 0)
        momentum_score = min(7, max(0, (close / sma_20 - 1) * 100)) if sma_20 > 0 else 0
        
        scores['momentum_score'] = golden_cross + rsi_score + momentum_score
        
        # Macro & Sector Score (15点)
        vix = row.get('vix_value', 25) or 25
        dgs10 = row.get('dgs10_value', 0) or 0
        dgs2 = row.get('dgs2_value', 0) or 0
        
        vix_score = 5 if vix < 18 else 0
        yield_score = 5 if (dgs10 - dgs2) > 0.8 else 0
        macro_bonus = 5  # 簡素化
        
        scores['macro_sector_score'] = vix_score + yield_score + macro_bonus
        
        # 総合スコア
        total_score = sum(scores.values())
        total_score = max(0, min(100, total_score))  # 0-100にクリップ
        
        return total_score, scores
    
    def process_batch(self, dates: List[str]) -> pd.DataFrame:
        """
        バッチ処理でスコア計算
        
        Args:
            dates: 処理対象日付のリスト
            
        Returns:
            計算済みスコアのDataFrame
        """
        logger.info(f"バッチ処理開始: {len(dates)}日分 ({dates[0]} ～ {dates[-1]})")
        
        # バッチデータ読み込み
        df = self.load_batch_data(dates)
        if len(df) == 0:
            logger.warning("バッチデータが空です")
            return pd.DataFrame()
        
        results = []
        
        for date in dates:
            date_str = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
            
            # その日のデータを取得（日付の型に関係なく比較）
            if df['date'].dtype == 'object':
                # datetime.dateオブジェクトの場合は文字列に変換して比較
                daily_df = df[df['date'].astype(str) == date_str].copy()
            else:
                daily_df = df[df['date'].dt.strftime('%Y-%m-%d') == date_str].copy()
            
            if len(daily_df) == 0:
                logger.warning(f"日付 {date_str} のデータが見つかりません")
                continue
            
            logger.info(f"日付 {date_str}: {len(daily_df)}銘柄を処理中")
            
            # 簡素化されたパーセンタイル計算
            percentiles = self.calculate_simple_percentiles(df, date_str)
            logger.debug(f"パーセンタイル計算完了: {len(percentiles)}個の指標")
            
            # 各銘柄のスコア計算
            processed_count = 0
            for idx, row in daily_df.iterrows():
                try:
                    total_score, score_details = self.calculate_simplified_scores(row, percentiles)
                    
                    # 結果記録
                    score_record = {
                        'symbol': row['symbol'],
                        'date': date_str,
                        'total_score': total_score,
                        **score_details
                    }
                    
                    results.append(score_record)
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"スコア計算エラー {row['symbol']} {date_str}: {e}")
                    continue
            
            logger.info(f"日付 {date_str}: {processed_count}件のスコアを計算")
        
        logger.info(f"バッチ処理完了: {len(results)}件のスコアを計算")
        
        # データベースに保存（小さなバッチで保存）
        if len(results) > 0:
            df = pd.DataFrame(results)
            
            # 重複削除（同じsymbol, dateの組み合わせ）
            df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
            
            # UPSERT処理（PostgreSQL用）
            try:
                # 小さなバッチに分けて保存（SQLパラメータ制限対応）
                save_batch_size = 500  # 1000から500に削減
                for i in range(0, len(df), save_batch_size):
                    batch_df = df.iloc[i:i + save_batch_size]
                    
                    # 一時テーブルに挿入してからUPSERT
                    temp_table = 'temp_daily_scores'
                    batch_df.to_sql(
                        temp_table,
                        self.engine,
                        if_exists='replace',
                        index=False,
                        method='multi'
                    )
                    
                    # UPSERTクエリ実行（日付の型キャスト追加）
                    upsert_query = f"""
                    INSERT INTO backtest_results.daily_scores (symbol, date, total_score, value_score, growth_score, quality_score, momentum_score, macro_sector_score)
                    SELECT symbol, date::date, total_score, value_score, growth_score, quality_score, momentum_score, macro_sector_score 
                    FROM {temp_table}
                    ON CONFLICT (symbol, date) 
                    DO UPDATE SET
                        total_score = EXCLUDED.total_score,
                        value_score = EXCLUDED.value_score,
                        growth_score = EXCLUDED.growth_score,
                        quality_score = EXCLUDED.quality_score,
                        momentum_score = EXCLUDED.momentum_score,
                        macro_sector_score = EXCLUDED.macro_sector_score
                    """
                    
                    with self.engine.connect() as conn:
                        conn.execute(text(upsert_query))
                        conn.commit()
                    
                    # 一時テーブル削除
                    with self.engine.connect() as conn:
                        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
                        conn.commit()
                
                logger.info(f"バッチ保存完了: {len(df)}行")
            except Exception as e:
                logger.error(f"UPSERT保存エラー: {e}")
                raise
        
        return pd.DataFrame(results)
    
    def calculate_scores_optimized(self, start_date: str = "2015-01-01", end_date: Optional[str] = None, limit_days: Optional[int] = None) -> None:
        """
        最適化されたスコア計算のメイン処理
        
        Args:
            start_date: 開始日
            end_date: 終了日
            limit_days: 処理する日数制限
        """
        logger.info("最適化されたスコア計算を開始")
        
        # 処理対象日付を取得
        all_dates = self.get_date_range(start_date, end_date)
        
        if limit_days and len(all_dates) > limit_days:
            all_dates = all_dates[-limit_days:]
            logger.info(f"期間制限により最新 {limit_days} 日分のみ処理")
        
        logger.info(f"処理対象: {len(all_dates)}日分 ({all_dates[0]} ～ {all_dates[-1]})")
        
        # バッチサイズを大幅に削減（SQLパラメータ制限対応）
        batch_size = 100  # 1000から100に削減
        
        # 日付をバッチに分割
        batches = [all_dates[i:i + batch_size] for i in range(0, len(all_dates), batch_size)]
        
        # バッチ処理
        all_results = []
        
        for i, batch in enumerate(batches):
            batch_dates = batch
            logger.info(f"バッチ {i+1}/{len(batches)} を処理中...")
            
            try:
                # バッチ処理実行
                batch_results = self.process_batch(batch_dates)
                all_results.extend(batch_results.to_dict('records'))
                
                logger.info(f"バッチ {i+1}/{len(batches)} 完了: {len(batch_results)}件処理")
                
            except Exception as e:
                logger.error(f"バッチ {i+1} でエラー: {e}")
                continue
        
        logger.info(f"最適化スコア計算完了: 合計 {len(all_results)} 件処理")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="最適化されたオリジナル総合スコア計算")
    parser.add_argument("--start-date", default="2015-01-01", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/score_weights.yaml", help="設定ファイルパス")
    parser.add_argument("--limit-days", type=int, help="処理する日数制限（テスト用）")
    
    args = parser.parse_args()
    
    try:
        # 最適化スコア計算器を初期化
        calculator = OptimizedScoreCalculator(args.config)
        
        # スコア計算実行
        calculator.calculate_scores_optimized(args.start_date, args.end_date, args.limit_days)
        
        logger.info("最適化スコア計算処理完了")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 