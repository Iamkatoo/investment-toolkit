#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
オリジナル総合スコア計算スクリプト（改良版）

指定された仕様に基づいて、日次でのスコア計算を実行します。
改良点：
- Z-score標準化による極端値の抑制
- Winsorize処理による外れ値対応
- Quality/Growthフィルターによる「安いだけ」銘柄の排除
- バランスの取れたファクター配分

- Value 20点（z-score化で暴走抑制）
- Growth 25点（成長性重視で+5）
- Quality 25点（維持）
- Momentum 15点（リバウンドのみで上位に来ないよう-5）
- Macro & Sector Overlay 15点（維持）
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
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

from investment_toolkit.database.db_manager import DatabaseManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backtest_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ImprovedScoreCalculator:
    """改良版オリジナル総合スコア計算クラス"""
    
    def __init__(self, config_path: str = "config/score_weights.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.engine = self._create_db_engine()
        
        # 改良版ウエイト設定
        self.factor_weights = {
            'value': 20,      # z-score化で暴走抑制
            'growth': 25,     # 成長性重視で+5
            'quality': 25,    # 維持
            'momentum': 15,   # リバウンドのみ抑制で-5
            'macro_sector': 15  # 維持
        }
        
        # Quality/Growthフィルター設定
        self.quality_filters = {
            'min_altman_z_proxy': 5.0,    # Altman Z代理指標最低値
            'min_piotroski_proxy': 3.0,   # Piotroski F代理指標最低値
            'min_eps_cagr_3y': -10.0,     # 3年EPS成長率最低値（%）
            'max_debt_equity': 2.0,       # 最大Debt/Equity比率
            'min_roic': -5.0,             # 最低ROIC（%）
        }
        
        # バリュートラップ対策設定
        self.value_trap_filters = {
            'min_per': 3.0,     # 最低PER
            'max_per': 50.0,    # 最高PER
            'min_pbr': 0.3,     # 最低PBR
            'optimal_per_range': (8.0, 15.0),  # 最適PER範囲
        }
    
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
        db_user = os.environ.get("DB_USER", "HOME") 
        db_password = os.environ.get("DB_PASSWORD", "")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        
        if db_password:
            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            connection_string = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"
        return create_engine(connection_string)
    
    def winsorize_data(self, data: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        """
        データのWinsorize処理（5%-95%）
        
        Args:
            data: 処理対象データ
            lower: 下位パーセンタイル
            upper: 上位パーセンタイル
            
        Returns:
            Winsorize処理済みデータ
        """
        if len(data.dropna()) == 0:
            return data
        
        lower_bound = data.quantile(lower)
        upper_bound = data.quantile(upper)
        
        return data.clip(lower=lower_bound, upper=upper_bound)
    
    def calculate_z_score(self, data: pd.Series, winsorize: bool = True) -> pd.Series:
        """
        Z-score標準化（オプションでWinsorize処理）
        
        Args:
            data: 処理対象データ
            winsorize: Winsorize処理の有無
            
        Returns:
            Z-score標準化済みデータ
        """
        if len(data.dropna()) < 2:
            return pd.Series(0, index=data.index)
        
        # Winsorize処理
        if winsorize:
            processed_data = self.winsorize_data(data)
        else:
            processed_data = data.copy()
        
        # Z-score計算
        mean_val = processed_data.mean()
        std_val = processed_data.std()
        
        if std_val == 0 or pd.isna(std_val):
            return pd.Series(0, index=data.index)
        
        z_scores = (processed_data - mean_val) / std_val
        
        # 極端な値を-3〜3にクリップ
        return z_scores.clip(-3, 3)
    
    def apply_quality_growth_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quality/Growthフィルターを適用して「安いだけ」銘柄を除外（フィルター無効化版）
        
        Args:
            df: 対象データフレーム
            
        Returns:
            フィルター適用後のデータフレーム
        """
        logger.info("Quality/Growthフィルター（無効化版）を適用中...")
        initial_count = len(df)
        
        # フィルターを無効化：すべての銘柄を通す
        filtered_df = df.copy()
        
        filtered_count = len(filtered_df)
        logger.info(f"Quality/Growthフィルター結果（フィルター無効）: {initial_count}銘柄 → {filtered_count}銘柄 ({filtered_count/initial_count*100:.1f}%残存)")
        
        return filtered_df
    
    def apply_value_trap_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        バリュートラップフィルターを適用（フィルター無効化版）
        
        Args:
            df: 対象データフレーム
            
        Returns:
            フィルター適用後のデータフレーム
        """
        logger.info("バリュートラップフィルター（無効化版）を適用中...")
        initial_count = len(df)
        
        # フィルターを無効化：すべての銘柄を通す
        filtered_df = df.copy()
        
        filtered_count = len(filtered_df)
        logger.info(f"バリュートラップフィルター結果（フィルター無効）: {initial_count}銘柄 → {filtered_count}銘柄 ({filtered_count/initial_count*100:.1f}%残存)")
        
        return filtered_df
    
    def load_master_data(self, start_date: str = "2015-01-01", end_date: Optional[str] = None, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        マスターデータを読み込み
        
        Args:
            start_date: 開始日
            end_date: 終了日（Noneの場合は最新日まで）
            target_date: 特定日のみ読み込み（指定時は他の日付は無視）
            
        Returns:
            マスターデータのDataFrame
        """
        logger.info("マスターデータを読み込み中...")
        
        if target_date:
            # 特定日のみの場合
            query = """
            SELECT * FROM backtest_results.vw_daily_master
            WHERE date = %(target_date)s
            ORDER BY symbol, date
            """
            params = {'target_date': target_date}
        else:
            # 期間指定の場合
            query = """
            SELECT * FROM backtest_results.vw_daily_master
            WHERE date >= %(start_date)s
            """
            params = {'start_date': start_date}
            
            if end_date:
                query += " AND date <= %(end_date)s"
                params['end_date'] = end_date
                
            query += " ORDER BY symbol, date"
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            logger.info(f"データ読み込み完了: {len(df):,}行")
            return df
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise
    
    def calculate_percentiles_optimized(self, df: pd.DataFrame, date: str, lookback_days: int = 252) -> Dict[str, Dict]:
        """
        最適化されたパーセンタイル計算
        
        Args:
            df: データフレーム
            date: 計算対象日
            lookback_days: ルックバック期間
            
        Returns:
            各指標のパーセンタイル辞書
        """
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=lookback_days)
        
        # 当日のデータを取得
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return {}
        
        # 期間内データを取得（効率化のため、必要な列のみ）
        period_df = df[
            (pd.to_datetime(df['date']) >= start_date) & 
            (pd.to_datetime(df['date']) <= end_date)
        ][['symbol', 'date', 'per', 'fcf_yield', 'roic', 'roe', 'eps_cagr_3y', 'revenue_cagr_3y', 'rsi_14', 'raw_industry']].copy()
        
        percentiles = {}
        
        # PER（業界別パーセンタイル）- 最適化版
        if 'per' in target_df.columns and 'raw_industry' in target_df.columns:
            industry_per = {}
            
            # 業界ごとに一括処理
            for industry in target_df['raw_industry'].unique():
                if pd.isna(industry):
                    continue
                
                # 期間内の業界データ
                industry_period = period_df[period_df['raw_industry'] == industry]['per'].dropna()
                # 当日の業界データ
                industry_target = target_df[target_df['raw_industry'] == industry]
                
                if len(industry_period) > 1:
                    # numpy配列に変換して高速化
                    period_values = industry_period.values
                    
                    for idx, row in industry_target.iterrows():
                        per_value = row['per']
                        if not pd.isna(per_value):
                            percentile = np.mean(period_values <= per_value)
                            industry_per[idx] = percentile
            
            percentiles['per_industry_percentile'] = industry_per
        
        # その他の指標（全体パーセンタイル）- 最適化版
        for col in ['fcf_yield', 'roic', 'roe', 'eps_cagr_3y', 'revenue_cagr_3y', 'rsi_14']:
            if col in target_df.columns:
                period_data = period_df[col].dropna().values
                target_data = target_df[col].dropna()
                
                if len(period_data) > 1:
                    col_percentiles = {}
                    for idx, value in target_data.items():
                        if not pd.isna(value):
                            percentile = np.mean(period_data <= value)
                            col_percentiles[idx] = percentile
                    
                    percentiles[f'{col}_percentile'] = col_percentiles
        
        return percentiles
    
    def calculate_improved_value_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Valueスコア計算（Z-score標準化ベース）
        
        Args:
            df: 全期間データ
            date: 計算対象日
            
        Returns:
            Valueスコア付きデータフレーム
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return target_df
        
        # 期間データ（252日）を取得
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=252)
        period_df = df[
            (pd.to_datetime(df['date']) >= start_date) & 
            (pd.to_datetime(df['date']) <= end_date)
        ].copy()
        
        value_scores = []
        
        # 各Value指標のZ-score計算
        value_indicators = {
            'per_z': ('per', True),      # PERは反転（低い方が良い）
            'fcf_yield_z': ('fcf_yield', False),  # FCF Yieldは高い方が良い
            'pbr_z': ('pbr', True),      # PBRは反転（低い方が良い）
        }
        
        for z_col, (orig_col, invert) in value_indicators.items():
            if orig_col in period_df.columns:
                # 業界別でPER、全体でその他
                if orig_col == 'per' and 'raw_industry' in period_df.columns:
                    # 業界別Z-score計算
                    target_df[z_col] = 0.0
                    for industry in target_df['raw_industry'].unique():
                        if pd.isna(industry):
                            continue
                        
                        industry_period = period_df[period_df['raw_industry'] == industry][orig_col]
                        industry_target = target_df[target_df['raw_industry'] == industry]
                        
                        if len(industry_period.dropna()) > 10:  # 最低データ数確保
                            z_scores = self.calculate_z_score(industry_period)
                            # 当日データのZ-score
                            for idx, row in industry_target.iterrows():
                                if not pd.isna(row[orig_col]):
                                    target_val = row[orig_col]
                                    z_score = (target_val - industry_period.mean()) / industry_period.std()
                                    z_score = max(-3, min(3, z_score))  # クリップ
                                    target_df.loc[idx, z_col] = -z_score if invert else z_score
                else:
                    # 全体でZ-score計算
                    period_data = period_df[orig_col]
                    if len(period_data.dropna()) > 10:
                        for idx, row in target_df.iterrows():
                            if not pd.isna(row[orig_col]):
                                target_val = row[orig_col]
                                z_score = (target_val - period_data.mean()) / period_data.std()
                                z_score = max(-3, min(3, z_score))  # クリップ
                                target_df.loc[idx, z_col] = -z_score if invert else z_score
                            else:
                                target_df.loc[idx, z_col] = 0.0
        
        # PERロジスティックスコア（最適範囲で最高点）
        if 'per' in target_df.columns:
            optimal_min, optimal_max = self.value_trap_filters['optimal_per_range']
            target_df['per_logistic'] = target_df['per'].apply(
                lambda x: self._calculate_logistic_per_score(x, optimal_min, optimal_max) if not pd.isna(x) else 0
            )
        else:
            target_df['per_logistic'] = 0.0
        
        # Valueスコア合成（各指標の重み）
        value_weights = {
            'per_z': 0.3,
            'per_logistic': 0.3,  # PERロジスティック
            'fcf_yield_z': 0.25,
            'pbr_z': 0.15,
        }
        
        target_df['value_component_score'] = 0.0
        for indicator, weight in value_weights.items():
            if indicator in target_df.columns:
                # Z-scoreを0-1スケールに正規化
                normalized = (target_df[indicator] + 3) / 6  # -3~3 → 0~1
                target_df['value_component_score'] += normalized * weight
        
        # 最終Valueスコア（20点満点）
        target_df['value_score'] = target_df['value_component_score'] * self.factor_weights['value']
        target_df['value_score'] = target_df['value_score'].clip(0, self.factor_weights['value'])
        
        return target_df
    
    def _calculate_logistic_per_score(self, per_value: float, optimal_min: float, optimal_max: float) -> float:
        """
        PERのロジスティックスコア計算
        
        Args:
            per_value: PER値
            optimal_min: 最適範囲の最小値
            optimal_max: 最適範囲の最大値
            
        Returns:
            ロジスティックスコア（0-1）
        """
        if pd.isna(per_value):
            return 0
        
        # 最適範囲内なら高スコア
        if optimal_min <= per_value <= optimal_max:
            return 1.0
        
        # 最適範囲から離れるほど低スコア
        if per_value < optimal_min:
            # 低すぎる場合（バリュートラップの可能性）
            distance = optimal_min - per_value
            return max(0, 1 - distance / optimal_min)
        else:
            # 高すぎる場合
            distance = per_value - optimal_max
            return max(0, 1 - distance / (optimal_max * 2))
    
    def calculate_improved_growth_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Growthスコア計算（Z-score標準化ベース、25点満点）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return target_df
        
        # 期間データを取得
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=252)
        period_df = df[
            (pd.to_datetime(df['date']) >= start_date) & 
            (pd.to_datetime(df['date']) <= end_date)
        ].copy()
        
        # Growth指標のZ-score計算
        growth_indicators = {
            'eps_cagr_3y_z': 'eps_cagr_3y',
            'eps_cagr_5y_z': 'eps_cagr_5y',
            'revenue_cagr_3y_z': 'revenue_cagr_3y',
            'revenue_cagr_5y_z': 'revenue_cagr_5y',
        }
        
        for z_col, orig_col in growth_indicators.items():
            if orig_col in period_df.columns:
                period_data = period_df[orig_col]
                if len(period_data.dropna()) > 10:
                    for idx, row in target_df.iterrows():
                        if not pd.isna(row[orig_col]):
                            target_val = row[orig_col]
                            z_score = (target_val - period_data.mean()) / period_data.std()
                            target_df.loc[idx, z_col] = max(-3, min(3, z_score))
                        else:
                            target_df.loc[idx, z_col] = 0.0
                else:
                    target_df[z_col] = 0.0
            else:
                target_df[z_col] = 0.0
        
        # Growth Consistency（EPS CAGR安定性）
        if 'eps_cagr_3y' in target_df.columns and 'eps_cagr_5y' in target_df.columns:
            consistency = np.abs(target_df['eps_cagr_5y'] - target_df['eps_cagr_3y'])
            # 差が小さいほど高スコア
            target_df['growth_consistency_z'] = -consistency / 5.0  # 5%差で-1.0
            target_df['growth_consistency_z'] = target_df['growth_consistency_z'].clip(-3, 3)
        else:
            target_df['growth_consistency_z'] = 0.0
        
        # Growthスコア合成（重み調整でEPSを重視）
        growth_weights = {
            'eps_cagr_3y_z': 0.3,
            'eps_cagr_5y_z': 0.3,
            'revenue_cagr_3y_z': 0.15,
            'revenue_cagr_5y_z': 0.15,
            'growth_consistency_z': 0.1,
        }
        
        target_df['growth_component_score'] = 0.0
        for indicator, weight in growth_weights.items():
            if indicator in target_df.columns:
                normalized = (target_df[indicator] + 3) / 6  # -3~3 → 0~1
                target_df['growth_component_score'] += normalized * weight
        
        # 最終Growthスコア（25点満点）
        target_df['growth_score'] = target_df['growth_component_score'] * self.factor_weights['growth']
        target_df['growth_score'] = target_df['growth_score'].clip(0, self.factor_weights['growth'])
        
        return target_df
    
    def calculate_improved_quality_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Qualityスコア計算（Z-score標準化ベース、25点満点）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return target_df
        
        # 期間データを取得
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=252)
        period_df = df[
            (pd.to_datetime(df['date']) >= start_date) & 
            (pd.to_datetime(df['date']) <= end_date)
        ].copy()
        
        # Quality指標のZ-score計算
        quality_indicators = {
            'roic_z': 'roic',
            'roe_z': 'roe',
            'debt_equity_z': ('debt_to_equity', True),  # 低い方が良い（反転）
            'cfo_net_income_z': 'cfo_to_net_income',
        }
        
        for z_col, indicator_info in quality_indicators.items():
            if isinstance(indicator_info, tuple):
                orig_col, invert = indicator_info
            else:
                orig_col, invert = indicator_info, False
            
            if orig_col in period_df.columns:
                period_data = period_df[orig_col]
                if len(period_data.dropna()) > 10:
                    for idx, row in target_df.iterrows():
                        if not pd.isna(row[orig_col]):
                            target_val = row[orig_col]
                            z_score = (target_val - period_data.mean()) / period_data.std()
                            z_score = max(-3, min(3, z_score))
                            target_df.loc[idx, z_col] = -z_score if invert else z_score
                        else:
                            target_df.loc[idx, z_col] = 0.0
                else:
                    target_df[z_col] = 0.0
            else:
                target_df[z_col] = 0.0
        
        # Altman Z代理スコア
        if 'roe' in target_df.columns and 'roic' in target_df.columns:
            altman_proxy = (target_df['roe'].fillna(0) + target_df['roic'].fillna(0)) / 2
            target_df['altman_z_score'] = (altman_proxy > 15).astype(float)  # 15%以上で1点
        else:
            target_df['altman_z_score'] = 0.0
        
        # Piotroski F代理スコア
        piotroski_score = np.zeros(len(target_df))
        if 'roic' in target_df.columns:
            piotroski_score += (target_df['roic'].fillna(0) > 10).astype(float) * 2
        if 'roe' in target_df.columns:
            piotroski_score += (target_df['roe'].fillna(0) > 15).astype(float) * 2
        if 'cfo_to_net_income' in target_df.columns:
            piotroski_score += (target_df['cfo_to_net_income'].fillna(0) > 1).astype(float) * 2
        
        target_df['piotroski_proxy_z'] = piotroski_score / 6.0  # 0-1スケール
        
        # Qualityスコア合成
        quality_weights = {
            'roic_z': 0.25,
            'roe_z': 0.15,
            'debt_equity_z': 0.15,
            'altman_z_score': 0.15,
            'piotroski_proxy_z': 0.15,
            'cfo_net_income_z': 0.15,
        }
        
        target_df['quality_component_score'] = 0.0
        for indicator, weight in quality_weights.items():
            if indicator in target_df.columns:
                if indicator in ['altman_z_score', 'piotroski_proxy_z']:
                    # 既に0-1スケール
                    normalized = target_df[indicator]
                else:
                    # Z-scoreを0-1スケールに正規化
                    normalized = (target_df[indicator] + 3) / 6
                target_df['quality_component_score'] += normalized * weight
        
        # 最終Qualityスコア（25点満点）
        target_df['quality_score'] = target_df['quality_component_score'] * self.factor_weights['quality']
        target_df['quality_score'] = target_df['quality_score'].clip(0, self.factor_weights['quality'])
        
        return target_df
    
    def calculate_improved_momentum_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Momentumスコア計算（15点満点）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return target_df
        
        # 既存のMomentum計算をそのまま使用（重みを15点に調整）
        momentum_scores = []
        
        for idx, row in target_df.iterrows():
            _, momentum_details = self.calculate_momentum_score(row)
            
            # スコア合成（15点満点に調整）
            total_momentum = sum(momentum_details.values())
            # 既存の20点から15点にスケール調整
            scaled_momentum = total_momentum * (self.factor_weights['momentum'] / 20)
            
            target_df.loc[idx, 'momentum_score'] = max(0, min(self.factor_weights['momentum'], scaled_momentum))
        
        return target_df
    
    def calculate_daily_scores(self, df: pd.DataFrame, target_date: Optional[str] = None, limit_days: Optional[int] = None) -> pd.DataFrame:
        """
        改良版日次スコア計算のメイン処理
        
        Args:
            df: マスターデータ
            target_date: 計算対象日（Noneの場合は全日）
            limit_days: 処理する日数制限（テスト用）
            
        Returns:
            計算済みスコアのDataFrame
        """
        if target_date:
            dates = [target_date]
        else:
            dates = sorted(df['date'].unique())
            
            # 期間制限がある場合は最新のN日分のみ処理
            if limit_days and len(dates) > limit_days:
                dates = dates[-limit_days:]
                logger.info(f"期間制限により最新 {limit_days} 日分のみ処理: {dates[0]} ～ {dates[-1]}")
        
        results = []
        
        logger.info(f"改良版スコア計算開始: {len(dates)}日分")
        
        # DataFrameの日付列を文字列に統一
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        for date in tqdm(dates, desc="改良版スコア計算中"):
            date_str = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
            
            # その日のデータを取得
            daily_df = df[df['date'] == date_str].copy()
            if len(daily_df) == 0:
                logger.warning(f"日付 {date_str} のデータがありません")
                continue
            
            logger.info(f"日付 {date_str}: {len(daily_df)}銘柄のデータを処理中")
            
            # ステップ1: Quality/Growthフィルター適用
            filtered_df = self.apply_quality_growth_filters(daily_df)
            if len(filtered_df) == 0:
                logger.warning(f"日付 {date_str}: Quality/Growthフィルター後に銘柄が残りませんでした")
                continue
            
            # ステップ2: バリュートラップフィルター適用
            filtered_df = self.apply_value_trap_filters(filtered_df)
            if len(filtered_df) == 0:
                logger.warning(f"日付 {date_str}: バリュートラップフィルター後に銘柄が残りませんでした")
                continue
            
            logger.info(f"日付 {date_str}: フィルター後 {len(filtered_df)}銘柄を処理")
            
            # ステップ3: 各ファクタースコア計算（Z-scoreベース）
            try:
                # Valueスコア計算
                value_df = self.calculate_improved_value_score(df, date_str)
                value_scores = value_df[value_df['symbol'].isin(filtered_df['symbol'])][['symbol', 'value_score']].set_index('symbol')['value_score']
                
                # Growthスコア計算
                growth_df = self.calculate_improved_growth_score(df, date_str)
                growth_scores = growth_df[growth_df['symbol'].isin(filtered_df['symbol'])][['symbol', 'growth_score']].set_index('symbol')['growth_score']
                
                # Qualityスコア計算
                quality_df = self.calculate_improved_quality_score(df, date_str)
                quality_scores = quality_df[quality_df['symbol'].isin(filtered_df['symbol'])][['symbol', 'quality_score']].set_index('symbol')['quality_score']
                
                # Momentumスコア計算（既存ロジック使用、15点にスケール）
                momentum_scores = {}
                for idx, row in filtered_df.iterrows():
                    _, momentum_details = self.calculate_momentum_score(row)
                    total_momentum = sum(momentum_details.values())
                    scaled_momentum = total_momentum * (self.factor_weights['momentum'] / 20)
                    momentum_scores[row['symbol']] = max(0, min(self.factor_weights['momentum'], scaled_momentum))
                
                # Macro & Sectorスコア計算（既存ロジック使用）
                sector_top3 = self.calculate_sector_rankings(df, date_str)
                macro_scores = {}
                for idx, row in filtered_df.iterrows():
                    macro_score, _ = self.calculate_macro_sector_score(row, sector_top3)
                    macro_scores[row['symbol']] = max(0, min(self.factor_weights['macro_sector'], macro_score))
                
            except Exception as e:
                logger.error(f"日付 {date_str} のスコア計算でエラー: {e}")
                continue
            
            # ステップ4: 総合スコア計算と結果記録
            processed_count = 0
            error_count = 0
            
            for idx, row in filtered_df.iterrows():
                try:
                    symbol = row['symbol']
                    
                    # 各ファクタースコア取得
                    value_score = value_scores.get(symbol, 0.0)
                    growth_score = growth_scores.get(symbol, 0.0)
                    quality_score = quality_scores.get(symbol, 0.0)
                    momentum_score = momentum_scores.get(symbol, 0.0)
                    macro_sector_score = macro_scores.get(symbol, 0.0)
                    
                    # スコアをクリップ
                    value_score = self.safe_clip_score(value_score, 0, self.factor_weights['value'])
                    growth_score = self.safe_clip_score(growth_score, 0, self.factor_weights['growth'])
                    quality_score = self.safe_clip_score(quality_score, 0, self.factor_weights['quality'])
                    momentum_score = self.safe_clip_score(momentum_score, 0, self.factor_weights['momentum'])
                    macro_sector_score = self.safe_clip_score(macro_sector_score, 0, self.factor_weights['macro_sector'])
                    
                    # 総合スコア計算（100点満点）
                    total_score = value_score + growth_score + quality_score + momentum_score + macro_sector_score
                    total_score = self.safe_clip_score(total_score, 0, 100)
                    
                    # 結果記録（データベーススキーマに合わせて）
                    score_record = {
                        'symbol': symbol,
                        'date': date_str,
                        'value_score': value_score,
                        'growth_score': growth_score,
                        'quality_score': quality_score,
                        'momentum_score': momentum_score,
                        'macro_sector_score': macro_sector_score,
                        'total_score': total_score,
                        # メタデータは削除（データベーススキーマに存在しない）
                    }
                    
                    results.append(score_record)
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"スコア計算エラー {row['symbol']} {date_str}: {e}")
                    error_count += 1
                    continue
            
            logger.info(f"日付 {date_str}: 処理完了 - 成功: {processed_count}, エラー: {error_count}")
        
        logger.info(f"改良版スコア計算完了: 合計 {len(results)} 件のスコアを計算")
        return pd.DataFrame(results)
    
    def calculate_momentum_score(self, row: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        Momentum スコア計算（従来ロジック維持、20点満点）
        
        Args:
            row: データ行
            
        Returns:
            (総合スコア, 詳細スコア辞書)
        """
        scores = {}
        
        # 20/40 SMA ゴールデンクロス +5
        sma_20 = row.get('sma_20', 0) or 0
        sma_40 = row.get('sma_40', 0) or 0
        if sma_20 > sma_40 and sma_20 > 0 and sma_40 > 0:
            scores['golden_cross_score'] = 5
        else:
            scores['golden_cross_score'] = 0
        
        # RSI 50–70 +3、70–80 +1、>80 0
        rsi = row.get('rsi_14', 50) or 50
        if 50 <= rsi <= 70:
            scores['rsi_score'] = 3
        elif 70 < rsi <= 80:
            scores['rsi_score'] = 1
        else:
            scores['rsi_score'] = 0
        
        # MACD ヒストグラム >0 +2
        macd_hist = row.get('macd_hist', 0) or 0
        scores['macd_hist_score'] = 2 if macd_hist > 0 else 0
        
        # Vol-Adj Momentum = (Price/SMA20-1)/ATR14 → Z → +5
        price = row.get('close', 0) or 0
        atr = row.get('atr_14', 1) or 1
        if sma_20 > 0 and atr > 0:
            vol_adj_momentum = (price / sma_20 - 1) / (atr / price)
            # Z-score風に正規化（-2〜2の範囲を0〜5点にマッピング）
            z_score = max(-2, min(2, vol_adj_momentum))
            scores['vol_adj_momentum_score'] = (z_score + 2) * 5 / 4
        else:
            scores['vol_adj_momentum_score'] = 0
        
        # Relative Strength vs Industry +5
        industry_close = row.get('industry_avg_close', 0) or 0
        if industry_close > 0 and price > 0:
            relative_strength = price / industry_close
            # 1.0を基準として正規化
            scores['relative_strength_score'] = min(5, max(0, (relative_strength - 0.9) * 5 / 0.2))
        else:
            scores['relative_strength_score'] = 0
        
        total_score = sum(scores.values())
        return total_score, scores
    
    def calculate_macro_sector_score(self, row: pd.Series, sector_top3: set) -> Tuple[float, Dict[str, float]]:
        """
        Macro & Sector Overlay スコア計算（15点満点）
        
        Args:
            row: データ行
            sector_top3: その日の上位3セクターのセット
            
        Returns:
            (総合スコア, 詳細スコア辞書)
        """
        scores = {}
        
        # Tail-wind (VIX<18 +2, yield_spread>0.8 +2, CPI↓&FFR↑ +1) max 10
        tail_wind_score = 0
        
        # VIX < 18
        vix = row.get('vix_value', 25) or 25
        if vix < 18:
            tail_wind_score += 2
        
        # yield_spread > 0.8
        dgs10 = row.get('dgs10_value', 0) or 0
        dgs2 = row.get('dgs2_value', 0) or 0
        if dgs10 > 0 and dgs2 > 0:
            yield_spread = dgs10 - dgs2
            if yield_spread > 0.8:
                tail_wind_score += 2
        
        # CPI下落 & FFR上昇（簡易判定）
        cpi = row.get('cpi_value', 0) or 0
        ffr = row.get('ffr_value', 0) or 0
        if ffr > 2:  # FFRが一定水準以上
            tail_wind_score += 1
        
        scores['tail_wind_score'] = min(10, tail_wind_score)
        
        # Sector Rotation: 同日 sector_SMA20_RS 上位3セクター銘柄 +5
        sector_id = row.get('sector_id')
        if sector_id in sector_top3:
            scores['sector_rotation_score'] = 5
        else:
            scores['sector_rotation_score'] = 0
        
        total_score = sum(scores.values())
        return total_score, scores
    
    def calculate_sector_rankings(self, df: pd.DataFrame, date: str) -> set:
        """
        指定日のセクター別相対強度ランキングを計算
        
        Args:
            df: データフレーム
            date: 計算対象日
            
        Returns:
            上位3セクターのIDセット
        """
        date_df = df[df['date'] == date].copy()
        
        if len(date_df) == 0:
            return set()
        
        # セクター別の平均相対強度を計算
        sector_performance = date_df.groupby('sector_id').agg({
            'close': 'mean',
            'sma_20': 'mean'
        }).reset_index()
        
        # 相対強度計算（SMA20比）
        sector_performance['relative_strength'] = sector_performance['close'] / sector_performance['sma_20']
        
        # 上位3セクターを取得
        top3_sectors = sector_performance.nlargest(3, 'relative_strength')['sector_id'].tolist()
        
        return set(top3_sectors)
    
    def safe_clip_score(self, score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        スコアを安全な範囲内にクリップ
        
        Args:
            score: 元のスコア
            min_val: 最小値
            max_val: 最大値
            
        Returns:
            クリップされたスコア
        """
        if pd.isna(score):
            return 0.0
        return max(min_val, min(max_val, float(score)))

    def save_scores(self, scores_df: pd.DataFrame) -> None:
        """
        計算済みスコアをデータベースに保存
        
        Args:
            scores_df: スコアデータフレーム
        """
        if len(scores_df) == 0:
            logger.warning("保存するスコアデータがありません")
            return
        
        logger.info(f"改良版スコアデータを保存中: {len(scores_df)}行")
        
        try:
            # 重複データを削除（日付ごとに処理）
            unique_dates = scores_df['date'].unique()
            for date in unique_dates:
                delete_query = text("DELETE FROM backtest_results.daily_scores WHERE date = :date")
                with self.engine.connect() as conn:
                    result = conn.execute(delete_query, {'date': date})
                    conn.commit()
                    if result.rowcount > 0:
                        logger.info(f"日付 {date} の既存データ {result.rowcount} 行を削除")
            
            # 大量データの場合はバッチ処理
            batch_size = 500
            
            for i in range(0, len(scores_df), batch_size):
                end_idx = min(i + batch_size, len(scores_df))
                batch_df = scores_df.iloc[i:end_idx].copy()
                
                logger.info(f"バッチ {i//batch_size + 1}: {i+1}-{end_idx}行目を保存中")
                
                # Noneを含む行を削除
                batch_df = batch_df.dropna(subset=['total_score'])
                
                if len(batch_df) == 0:
                    logger.warning(f"バッチ {i//batch_size + 1}: 有効なスコアがありません")
                    continue
                
                # データベースに保存
                batch_df.to_sql(
                    'daily_scores',
                    self.engine,
                    schema='backtest_results',
                    if_exists='append',
                    index=False
                )
            
            logger.info("改良版スコアデータ保存完了")
            
        except Exception as e:
            logger.error(f"スコアデータ保存エラー: {e}")
            raise


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="オリジナル総合スコア計算")
    parser.add_argument("--start-date", default="2015-01-01", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--target-date", help="特定日のみ計算 (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/score_weights.yaml", help="設定ファイルパス")
    parser.add_argument("--limit-days", type=int, help="処理する日数制限（テスト用）")
    
    args = parser.parse_args()
    
    try:
        # スコア計算器を初期化
        calculator = ImprovedScoreCalculator(args.config)
        
        # マスターデータ読み込み
        df = calculator.load_master_data(args.start_date, args.end_date, args.target_date)
        
        # スコア計算
        scores_df = calculator.calculate_daily_scores(df, args.target_date, args.limit_days)
        
        # データベースに保存
        calculator.save_scores(scores_df)
        
        logger.info("スコア計算処理完了")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 