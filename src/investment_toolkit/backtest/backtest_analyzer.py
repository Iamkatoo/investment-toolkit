#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
スコアベースバックテスト分析システム
スコアと株価の相関、エントリータイミング、パフォーマンス分析
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text

# 警告を抑制
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
sns.set_style("whitegrid")

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestAnalyzer:
    """スコアベースバックテスト分析器"""
    
    def __init__(self):
        """初期化"""
        logger.info("バックテスト分析器を初期化中...")
        self.engine = self._create_db_engine()
        
    def _create_db_engine(self):
        """データベースエンジン作成"""
        db_name = os.environ.get("DB_NAME", "investment")
        db_user = os.environ.get("DB_USER", "HOME")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        
        if db_password:
            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            connection_string = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"
        
        try:
            engine = create_engine(connection_string, pool_pre_ping=True)
            logger.info("データベース接続完了")
            return engine
        except Exception as e:
            logger.error(f"データベース接続エラー: {e}")
            raise
            
    def get_score_distribution(self, start_date: str = "2015-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
        """スコア分布を取得"""
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            date as score_date,
            total_score,
            value_score,
            growth_score,
            quality_score,
            momentum_score,
            macro_sector_score,
            symbol
        FROM backtest_results.daily_scores
        WHERE date BETWEEN %s AND %s
        AND total_score IS NOT NULL
        ORDER BY score_date, total_score DESC
        """
        
        df = pd.read_sql(query, self.engine, params=(start_date, end_date))
        logger.info(f"スコアデータ取得: {len(df)}件")
        return df
        
    def analyze_score_correlation(self, start_date: str = "2024-01-01", end_date: Optional[str] = None, 
                                 holding_periods: List[int] = [1, 5, 10, 20]) -> Dict:
        """スコアと株価の相関分析"""
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"相関分析開始: {start_date} ～ {end_date}")
        
        # スコアと将来リターンのデータを取得
        query = """
        WITH score_data AS (
            SELECT 
                symbol,
                date as score_date,
                total_score,
                value_score,
                growth_score,
                quality_score,
                momentum_score,
                macro_sector_score
            FROM backtest_results.daily_scores
            WHERE date BETWEEN %s AND %s
            AND total_score IS NOT NULL
        ),
        price_data AS (
            SELECT 
                symbol,
                date as price_date,
                close,
                LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) as prev_1d_close,
                LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date) as prev_5d_close,
                LAG(close, 10) OVER (PARTITION BY symbol ORDER BY date) as prev_10d_close,
                LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date) as prev_20d_close
            FROM fmp_data.daily_prices
            WHERE date BETWEEN %s AND %s
        )
        SELECT 
            s.symbol,
            s.score_date,
            s.total_score,
            s.value_score,
            s.growth_score,
            s.quality_score,
            s.momentum_score,
            s.macro_sector_score,
            p.close as entry_price,
            p.prev_1d_close,
            p.prev_5d_close,
            p.prev_10d_close,
            p.prev_20d_close,
            CASE WHEN p.prev_1d_close > 0 THEN (p.close - p.prev_1d_close) / p.prev_1d_close ELSE NULL END as return_1d,
            CASE WHEN p.prev_5d_close > 0 THEN (p.close - p.prev_5d_close) / p.prev_5d_close ELSE NULL END as return_5d,
            CASE WHEN p.prev_10d_close > 0 THEN (p.close - p.prev_10d_close) / p.prev_10d_close ELSE NULL END as return_10d,
            CASE WHEN p.prev_20d_close > 0 THEN (p.close - p.prev_20d_close) / p.prev_20d_close ELSE NULL END as return_20d
        FROM score_data s
        LEFT JOIN price_data p ON s.symbol = p.symbol AND s.score_date = p.price_date
        WHERE p.close IS NOT NULL
        """
        
        df = pd.read_sql(query, self.engine, params=(start_date, end_date, start_date, end_date))
        
        if len(df) == 0:
            logger.warning("相関分析用データが見つかりません")
            return {}
        
        logger.info(f"相関分析データ: {len(df)}件")
        
        # 相関分析
        correlation_results = {}
        score_columns = ['total_score', 'value_score', 'growth_score', 'quality_score', 'momentum_score', 'macro_sector_score']
        return_columns = ['return_1d', 'return_5d', 'return_10d', 'return_20d']
        
        for score_col in score_columns:
            correlation_results[score_col] = {}
            for return_col in return_columns:
                # 有効なデータのみで相関計算
                valid_data = df[[score_col, return_col]].dropna()
                if len(valid_data) > 10:
                    corr = valid_data[score_col].corr(valid_data[return_col])
                    correlation_results[score_col][return_col] = corr
                else:
                    correlation_results[score_col][return_col] = None
        
        return correlation_results
        
    def analyze_threshold_performance(self, start_date: str = "2024-01-01", end_date: Optional[str] = None,
                                    thresholds: List[float] = [40, 45, 50, 60], holding_period: int = 130) -> pd.DataFrame:
        """閾値別パフォーマンス分析（半年以内の最大値）"""
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"閾値別パフォーマンス分析開始: 閾値 {thresholds}, 保有期間 {holding_period}日（半年以内の最大値）")
        
        # 高スコア銘柄とそのパフォーマンスを取得（半年以内の最大値）
        query = """
        WITH score_data AS (
            SELECT 
                symbol,
                date as score_date,
                total_score,
                ROW_NUMBER() OVER (PARTITION BY date ORDER BY total_score DESC) as daily_rank
            FROM backtest_results.daily_scores
            WHERE date BETWEEN %s AND %s
            AND total_score IS NOT NULL
        ),
        price_returns AS (
            SELECT 
                symbol,
                date as entry_date,
                close as entry_price,
                -- 半年以内の最大値を取得
                MAX(close) OVER (
                    PARTITION BY symbol 
                    ORDER BY date 
                    ROWS BETWEEN 1 FOLLOWING AND %s FOLLOWING
                ) as max_price_6months
            FROM fmp_data.daily_prices
            WHERE date BETWEEN %s AND %s
        )
        SELECT 
            s.score_date,
            s.symbol,
            s.total_score,
            s.daily_rank,
            p.entry_price,
            p.max_price_6months as exit_price,
            CASE 
                WHEN p.entry_price > 0 AND p.max_price_6months IS NOT NULL 
                THEN (p.max_price_6months - p.entry_price) / p.entry_price 
                ELSE NULL 
            END as return_pct
        FROM score_data s
        LEFT JOIN price_returns p ON s.symbol = p.symbol AND s.score_date = p.entry_date
        WHERE p.entry_price IS NOT NULL
        ORDER BY s.score_date, s.total_score DESC
        """
        
        df = pd.read_sql(query, self.engine, params=(start_date, end_date, holding_period, start_date, end_date))
        
        if len(df) == 0:
            logger.warning("パフォーマンス分析用データが見つかりません")
            return pd.DataFrame()
        
        logger.info(f"パフォーマンス分析データ: {len(df)}件")
        
        # 閾値別の分析
        results = []
        
        for threshold in thresholds:
            threshold_data = df[df['total_score'] >= threshold].copy()
            
            if len(threshold_data) == 0:
                continue
                
            # 有効なリターンデータのみを使用
            valid_returns = threshold_data['return_pct'].dropna()
            
            if len(valid_returns) > 0:
                result = {
                    'threshold': threshold,
                    'count': len(threshold_data),
                    'valid_returns_count': len(valid_returns),
                    'mean_return': valid_returns.mean(),
                    'median_return': valid_returns.median(),
                    'std_return': valid_returns.std(),
                    'positive_ratio': (valid_returns > 0).mean(),
                    'best_return': valid_returns.max(),
                    'worst_return': valid_returns.min(),
                    'percentile_75': valid_returns.quantile(0.75),
                    'percentile_25': valid_returns.quantile(0.25)
                }
                results.append(result)
        
        return pd.DataFrame(results)
        
    def analyze_timing_effect(self, start_date: str = "2024-01-01", end_date: Optional[str] = None,
                            threshold: float = 50, lag_days: List[int] = [-30, -10, 0, 10]) -> pd.DataFrame:
        """エントリータイミング効果分析（半年以内の最大値）"""
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"タイミング効果分析開始: 閾値 {threshold}, ラグ {lag_days}日（半年以内の最大値）")
        
        results = []
        
        for lag in lag_days:
            # 負の値は「何日前」、正の値は「何日後」を意味する
            lag_description = f"{abs(lag)}日前" if lag < 0 else f"{lag}日後" if lag > 0 else "即日"
            
            query = """
            WITH score_data AS (
                SELECT 
                    symbol,
                    date as score_date,
                    total_score
                FROM backtest_results.daily_scores
                WHERE date BETWEEN %s AND %s
                AND total_score >= %s
            ),
            price_returns AS (
                SELECT 
                    symbol,
                    date as entry_date,
                    close as entry_price,
                    -- 半年以内の最大値を取得
                    MAX(close) OVER (
                        PARTITION BY symbol 
                        ORDER BY date 
                        ROWS BETWEEN 1 FOLLOWING AND 130 FOLLOWING
                    ) as max_price_6months
                FROM fmp_data.daily_prices
                WHERE date BETWEEN %s AND %s
            )
            SELECT 
                s.symbol,
                s.score_date,
                s.total_score,
                p.entry_price,
                p.max_price_6months as exit_price,
                CASE 
                    WHEN p.entry_price > 0 AND p.max_price_6months IS NOT NULL 
                    THEN (p.max_price_6months - p.entry_price) / p.entry_price 
                    ELSE NULL 
                END as return_pct
            FROM score_data s
            LEFT JOIN price_returns p ON s.symbol = p.symbol 
                AND p.entry_date = s.score_date + INTERVAL '%s days'
            WHERE p.entry_price IS NOT NULL
            """
            
            df = pd.read_sql(query, self.engine, params=(start_date, end_date, threshold, start_date, end_date, lag))
            
            if len(df) > 0:
                valid_returns = df['return_pct'].dropna()
                if len(valid_returns) > 0:
                    result = {
                        'lag_days': lag,
                        'lag_description': lag_description,
                        'count': len(valid_returns),
                        'mean_return': valid_returns.mean(),
                        'median_return': valid_returns.median(),
                        'std_return': valid_returns.std(),
                        'positive_ratio': (valid_returns > 0).mean(),
                        'sharpe_ratio': valid_returns.mean() / valid_returns.std() if valid_returns.std() > 0 else 0,
                        'best_return': valid_returns.max(),
                        'worst_return': valid_returns.min(),
                        'percentile_75': valid_returns.quantile(0.75),
                        'percentile_25': valid_returns.quantile(0.25)
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
        
    def generate_performance_report(self, start_date: str = "2024-01-01", end_date: Optional[str] = None,
                                  output_dir: str = "reports") -> str:
        """総合パフォーマンスレポート生成（修正版）"""
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"修正版パフォーマンスレポート生成開始: {start_date} ～ {end_date}")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 各種分析実行
        logger.info("1. スコア分布分析")
        score_dist = self.get_score_distribution(start_date, end_date)
        
        logger.info("2. 相関分析")
        correlation = self.analyze_score_correlation(start_date, end_date)
        
        logger.info("3. 閾値別パフォーマンス（半年以内の最大値）")
        threshold_perf = self.analyze_threshold_performance(start_date, end_date)
        
        logger.info("4. タイミング効果分析（半年以内の最大値）")
        timing_effect = self.analyze_timing_effect(start_date, end_date)
        
        # レポート作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{output_dir}/backtest_performance_report_v2_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("バックテストパフォーマンスレポート（修正版）\n")
            f.write("="*80 + "\n")
            f.write(f"分析期間: {start_date} ～ {end_date}\n")
            f.write(f"利益確定: 半年以内の最大値\n")
            f.write(f"レポート生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # スコア分布サマリー
            f.write("1. スコア分布サマリー\n")
            f.write("-"*40 + "\n")
            if not score_dist.empty:
                f.write(f"総データ件数: {len(score_dist):,}件\n")
                f.write(f"平均スコア: {score_dist['total_score'].mean():.2f}\n")
                f.write(f"最高スコア: {score_dist['total_score'].max():.2f}\n")
                f.write(f"最低スコア: {score_dist['total_score'].min():.2f}\n")
                f.write(f"標準偏差: {score_dist['total_score'].std():.2f}\n\n")
            
            # 相関分析結果
            f.write("2. スコアと株価リターンの相関\n")
            f.write("-"*40 + "\n")
            for score_type, correlations in correlation.items():
                f.write(f"{score_type}:\n")
                for period, corr in correlations.items():
                    if corr is not None:
                        f.write(f"  {period}: {corr:.4f}\n")
                f.write("\n")
            
            # 閾値別パフォーマンス
            f.write("3. 閾値別パフォーマンス（半年以内の最大値）\n")
            f.write("-"*50 + "\n")
            if not threshold_perf.empty:
                for _, row in threshold_perf.iterrows():
                    f.write(f"閾値 {row['threshold']}以上:\n")
                    f.write(f"  選定銘柄数: {row['count']:,}件\n")
                    f.write(f"  有効リターン数: {row['valid_returns_count']:,}件\n")
                    f.write(f"  平均リターン: {row['mean_return']:.2%}\n")
                    f.write(f"  中央値リターン: {row['median_return']:.2%}\n")
                    f.write(f"  勝率: {row['positive_ratio']:.1%}\n")
                    f.write(f"  75%タイル: {row['percentile_75']:.2%}\n")
                    f.write(f"  25%タイル: {row['percentile_25']:.2%}\n")
                    f.write(f"  最高リターン: {row['best_return']:.2%}\n")
                    f.write(f"  最低リターン: {row['worst_return']:.2%}\n\n")
            
            # タイミング効果
            f.write("4. エントリータイミング効果（閾値50以上、半年以内の最大値）\n")
            f.write("-"*60 + "\n")
            if not timing_effect.empty:
                for _, row in timing_effect.iterrows():
                    f.write(f"{row['lag_description']}エントリー:\n")
                    f.write(f"  対象銘柄数: {row['count']:,}件\n")
                    f.write(f"  平均リターン: {row['mean_return']:.2%}\n")
                    f.write(f"  中央値リターン: {row['median_return']:.2%}\n")
                    f.write(f"  勝率: {row['positive_ratio']:.1%}\n")
                    f.write(f"  75%タイル: {row['percentile_75']:.2%}\n")
                    f.write(f"  25%タイル: {row['percentile_25']:.2%}\n")
                    f.write(f"  シャープ比: {row['sharpe_ratio']:.3f}\n")
                    f.write(f"  最高リターン: {row['best_return']:.2%}\n")
                    f.write(f"  最低リターン: {row['worst_return']:.2%}\n\n")
        
        logger.info(f"修正版レポート生成完了: {report_file}")
        return report_file


def main():
    """メイン処理"""
    analyzer = BacktestAnalyzer()
    
    # 2024年のデータでレポート生成
    report_file = analyzer.generate_performance_report(
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    print(f"レポートが生成されました: {report_file}")


if __name__ == "__main__":
    main() 