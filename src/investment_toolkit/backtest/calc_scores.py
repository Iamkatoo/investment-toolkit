#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改良版総合スコア計算（改良版フィルタリング統合）

オリジナル総合スコア計算の改良版：
- Z-score標準化によるスコア暴走抑制
- Quality/Growthフィルターによるバリュートラップ対策
- Valueトラップフィルターによる異常値除外
- 詳細スコア項目の追加
- フィルタリング状態を記録してデータベースに保存

変更点：
- 全銘柄のスコア計算（フィルタリング関係なく）
- フィルタリング状態をBOOLEANカラムで記録
- レポート出力時にフィルタリング適用
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy import create_engine, text
import yaml

# tqdmをオプショナルで使用
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# プロジェクトルートへのパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedScoreCalculator:
    """
    改良版総合スコア計算クラス（フィルタリング統合版）
    
    主な改善点：
    - 全銘柄スコア計算 + フィルタリング状態記録
    - Z-score標準化による暴走抑制
    - バリュートラップ/Quality対策フィルター
    - 詳細スコア項目の充実
    """
    
    def __init__(self, config_file: str = "config/score_weights.yaml"):
        """
        初期化
        
        Args:
            config_file: 設定ファイルパス
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.factor_weights = self.config['scoring_weights']
        self.quality_filters = self.config['quality_growth_filters']
        self.value_filters = self.config['value_trap_filters']
        self.zscore_settings = self.config['zscore_settings']
        self.engine = self._create_db_engine()
        
        logger.info("改良版スコア計算器初期化完了")
        logger.info(f"Factor weights: {self.factor_weights}")
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"設定ファイル読み込み完了: {self.config_file}")
            return config
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
    
    def check_quality_growth_filter(self, row: pd.Series) -> bool:
        """
        Quality/Growthフィルターにかかるかどうかを判定
        
        Args:
            row: 銘柄データの行
            
        Returns:
            True: フィルターにかかる（除外対象）, False: 通過
        """
        try:
            # ROIC最低値チェック
            roic_val = row.get('roic')
            if pd.notna(roic_val) and isinstance(roic_val, (int, float)):
                if roic_val < self.quality_filters['min_roic']:
                    return True
            
            # Debt/Equity最大値チェック
            debt_equity_val = row.get('debt_to_equity')
            if pd.notna(debt_equity_val) and isinstance(debt_equity_val, (int, float)):
                if debt_equity_val > self.quality_filters['max_debt_equity']:
                    return True
            
            # EPS CAGR 3y最低値チェック
            eps_cagr_val = row.get('eps_cagr_3y')
            if pd.notna(eps_cagr_val) and isinstance(eps_cagr_val, (int, float)):
                if eps_cagr_val < self.quality_filters['min_eps_cagr_3y']:
                    return True
            
            # Altman Z代理指標チェック
            roe_val = row.get('roe')
            roic_val = row.get('roic')
            if pd.notna(roe_val) and pd.notna(roic_val) and isinstance(roe_val, (int, float)) and isinstance(roic_val, (int, float)):
                altman_proxy = (roe_val + roic_val) / 2
                if altman_proxy < self.quality_filters['min_altman_z_proxy']:
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Quality/Growthフィルター判定エラー {row.get('symbol', 'Unknown')}: {e}")
            return False
    
    def check_value_trap_filter(self, row: pd.Series) -> bool:
        """
        バリュートラップフィルターにかかるかどうかを判定
        
        Args:
            row: 銘柄データの行
            
        Returns:
            True: フィルターにかかる（除外対象）, False: 通過
        """
        try:
            # PER範囲チェック
            per_val = row.get('per')
            if pd.notna(per_val) and isinstance(per_val, (int, float)):
                if per_val < self.value_filters['min_per'] or per_val > self.value_filters['max_per']:
                    return True
            
            # PBR最低値チェック
            pbr_val = row.get('pbr')
            if pd.notna(pbr_val) and isinstance(pbr_val, (int, float)):
                if pbr_val < self.value_filters['min_pbr']:
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Value trapフィルター判定エラー {row.get('symbol', 'Unknown')}: {e}")
            return False
    
    def safe_clip_score(self, score: float, min_val: float, max_val: float) -> float:
        """スコアを安全にクリップ"""
        if pd.isna(score) or score == float('inf') or score == float('-inf'):
            return 0.0
        return max(min_val, min(max_val, score))
    
    def _get_scalar(self, df_idxed, key, col, default=0.0, prefer='last', logger=None):
        """
        df_idxed: symbol を index にした DataFrame
        key:      取り出したいシンボル
        col:      列名
        prefer:   'last' or 'first'（重複時どちらを採用するか）
        """
        try:
            if df_idxed is None or df_idxed.empty:
                return default
            if key not in df_idxed.index or col not in df_idxed.columns:
                return default
            val = df_idxed.loc[key, col]
            # Series / DataFrame を単一スカラーへ
            if isinstance(val, pd.Series):
                return float(val.iloc[-1] if prefer == 'last' else val.iloc[0])
            if isinstance(val, pd.DataFrame):
                s = val[col]
                return float(s.iloc[-1] if prefer == 'last' else s.iloc[0])
            return float(val)
        except Exception as e:
            if logger:
                logger.debug(f"_get_scalar error key={key} col={col}: {e}")
            return default
    
    def load_master_data(self, start_date: str = "2015-01-01", end_date: Optional[str] = None, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        マスターデータを読み込み
        
        Args:
            start_date: 開始日
            end_date: 終了日
            target_date: 特定日のみ取得する場合
            
        Returns:
            マスターデータのDataFrame
        """
        logger.info("マスターデータ読み込み開始...")
        
        # クエリ構築
        if target_date:
            date_condition = f"AND vm.date = '{target_date}'"
            logger.info(f"特定日データ取得: {target_date}")
        else:
            date_condition = f"AND vm.date >= '{start_date}'"
            if end_date:
                date_condition += f" AND vm.date <= '{end_date}'"
            logger.info(f"期間データ取得: {start_date} ～ {end_date or '最新'}")
        
        query = f"""
        SELECT 
            vm.*,
            COALESCE(cp.sector, 'Unknown') as sector,
            COALESCE(cp.industry, 'Unknown') as industry
        FROM backtest_results.vw_daily_master vm
        LEFT JOIN fmp_data.company_profile cp ON vm.symbol = cp.symbol
        WHERE vm.close > 0
        {date_condition}
        ORDER BY vm.date, vm.symbol
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"マスターデータ読み込み完了: {len(df):,}行, {df['symbol'].nunique():,}銘柄")
            return df
        except Exception as e:
            logger.error(f"マスターデータ読み込みエラー: {e}")
            raise
    
    def calculate_improved_value_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Valueスコア計算（Z-scoreベース）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return pd.DataFrame()
        
        # PER Z-score（低い方が良い）
        if 'per' in target_df.columns:
            target_df['per_z'] = -self.calculate_z_score(target_df['per'])  # 低い方が良いので反転
        else:
            target_df['per_z'] = 0
        
        # PER ロジスティックスコア（最適範囲重視）
        if 'per' in target_df.columns:
            optimal_range = self.value_filters['optimal_per_range']
            target_df['per_logistic'] = target_df['per'].apply(
                lambda x: 1 if optimal_range[0] <= x <= optimal_range[1] else 0
            )
        else:
            target_df['per_logistic'] = 0
        
        # FCF Yield Z-score（高い方が良い）
        if 'fcf_yield' in target_df.columns:
            target_df['fcf_yield_z'] = self.calculate_z_score(target_df['fcf_yield'])
        else:
            target_df['fcf_yield_z'] = 0
        
        # PBR Z-score（低い方が良い）
        if 'pbr' in target_df.columns:
            target_df['pbr_z'] = -self.calculate_z_score(target_df['pbr'])  # 低い方が良いので反転
        else:
            target_df['pbr_z'] = 0
        
        return target_df[['symbol', 'per_z', 'per_logistic', 'fcf_yield_z', 'pbr_z']]
    
    def calculate_improved_growth_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Growthスコア計算（Z-scoreベース）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return pd.DataFrame()
        
        # EPS CAGR Z-scores（高い方が良い）
        for period in ['3y', '5y']:
            col = f'eps_cagr_{period}'
            if col in target_df.columns:
                target_df[f'{col}_z'] = self.calculate_z_score(target_df[col])
            else:
                target_df[f'{col}_z'] = 0
        
        # Revenue CAGR Z-scores（高い方が良い）
        for period in ['3y', '5y']:
            col = f'revenue_cagr_{period}'
            if col in target_df.columns:
                target_df[f'{col}_z'] = self.calculate_z_score(target_df[col])
            else:
                target_df[f'{col}_z'] = 0
        
        # Growth Consistency Z-score
        if 'growth_consistency' in target_df.columns:
            target_df['growth_consistency_z'] = self.calculate_z_score(target_df['growth_consistency'])
        else:
            target_df['growth_consistency_z'] = 0
        
        return target_df[['symbol', 'eps_cagr_3y_z', 'eps_cagr_5y_z', 'revenue_cagr_3y_z', 'revenue_cagr_5y_z', 'growth_consistency_z']]
    
    def calculate_improved_quality_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Qualityスコア計算（Z-scoreベース）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return pd.DataFrame()
        
        # ROIC Z-score（高い方が良い）
        if 'roic' in target_df.columns:
            target_df['roic_z'] = self.calculate_z_score(target_df['roic'])
        else:
            target_df['roic_z'] = 0
        
        # ROE Z-score（高い方が良い）
        if 'roe' in target_df.columns:
            target_df['roe_z'] = self.calculate_z_score(target_df['roe'])
        else:
            target_df['roe_z'] = 0
        
        # Debt/Equity Z-score（低い方が良い）
        if 'debt_to_equity' in target_df.columns:
            target_df['debt_equity_z'] = -self.calculate_z_score(target_df['debt_to_equity'])  # 低い方が良いので反転
        else:
            target_df['debt_equity_z'] = 0
        
        # Altman Z代理スコア（ROE + ROIC平均）
        if 'roe' in target_df.columns and 'roic' in target_df.columns:
            target_df['altman_z_score'] = (target_df['roe'].fillna(0) + target_df['roic'].fillna(0)) / 2
            target_df['altman_z_score'] = target_df['altman_z_score'].apply(lambda x: min(4, max(0, x / 20)))  # 0-4スケール
        else:
            target_df['altman_z_score'] = 0
        
        # Piotroski F代理スコア（ROE + ROAベース）
        if 'roe' in target_df.columns and 'roa' in target_df.columns:
            target_df['piotroski_proxy_z'] = self.calculate_z_score(
                (target_df['roe'].fillna(0) + target_df['roa'].fillna(0)) / 2
            )
        else:
            target_df['piotroski_proxy_z'] = 0
        
        # CFO/Net Income比率
        if 'operating_cash_flow' in target_df.columns and 'net_income' in target_df.columns:
            target_df['cfo_net_income'] = target_df['operating_cash_flow'] / target_df['net_income'].replace(0, np.nan)
            target_df['cfo_net_income_z'] = self.calculate_z_score(target_df['cfo_net_income'])
        else:
            target_df['cfo_net_income_z'] = 0
        
        return target_df[['symbol', 'roic_z', 'roe_z', 'debt_equity_z', 'altman_z_score', 'piotroski_proxy_z', 'cfo_net_income_z']]

    def calculate_improved_momentum_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        改良版Momentumスコア計算（テクニカル指標ベース）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return pd.DataFrame()
        
        # ゴールデンクロススコア（SMA20 > SMA40）
        target_df['golden_cross'] = 0
        if 'sma_20' in target_df.columns and 'sma_40' in target_df.columns:
            target_df['golden_cross'] = (target_df['sma_20'] > target_df['sma_40']).astype(int)
        
        # RSIスコア（30-70範囲で最適化）
        target_df['rsi_score'] = 0
        if 'rsi_14' in target_df.columns:
            target_df['rsi_score'] = target_df['rsi_14'].apply(
                lambda x: 3 if 50 <= x <= 70 else (1 if 70 < x <= 80 else 0) if pd.notna(x) else 0
            )
        
        # MACD Histogramスコア
        target_df['macd_hist_score'] = 0
        if 'macd_hist' in target_df.columns:
            target_df['macd_hist_score'] = (target_df['macd_hist'] > 0).astype(int)
        
        # ボラティリティ調整モメンタム（12ヶ月リターン / ボラティリティ）
        target_df['vol_adj_momentum'] = 0
        if 'return_12m' in target_df.columns and 'vol_30d' in target_df.columns:
            target_df['vol_adj_momentum'] = target_df['return_12m'] / target_df['vol_30d'].replace(0, np.nan)
            target_df['vol_adj_momentum_z'] = self.calculate_z_score(target_df['vol_adj_momentum'])
        else:
            target_df['vol_adj_momentum_z'] = 0
        
        # 相対強度（業界比較）
        target_df['relative_strength'] = 0
        if 'return_12m' in target_df.columns and 'industry' in target_df.columns:
            industry_avg = target_df.groupby('industry')['return_12m'].mean()
            target_df['relative_strength'] = target_df.apply(
                lambda row: row['return_12m'] - industry_avg.get(row['industry'], 0) if pd.notna(row['return_12m']) else 0, 
                axis=1
            )
        
        return target_df[['symbol', 'golden_cross', 'rsi_score', 'macd_hist_score', 'vol_adj_momentum_z', 'relative_strength']]

    def calculate_macro_sector_score(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        マクロ・セクタースコア計算（固定バフ撤廃版）
        """
        target_df = df[df['date'] == date].copy()
        if len(target_df) == 0:
            return pd.DataFrame()
        
        # 固定バフを撤廃（定数加点問題を解決）
        target_df['tail_wind_score'] = 0.0  # 当面は0に固定（定数バフ防止）
        target_df['sector_rotation_score'] = 0.0  # 当面は0に固定
        
        # TODO: 将来的に実データ連動版を実装
        # - VIX < 18 → +3-4点
        # - イールドスプレッド > 0.8 → +2点  
        # - CPI低下＆FFR上昇 → +1点
        # - 業種RS上位セクター比率 → 0-5点
        
        return target_df[['symbol', 'tail_wind_score', 'sector_rotation_score']]
    
    def calculate_daily_scores(self, df: pd.DataFrame, target_date: Optional[str] = None, limit_days: Optional[int] = None) -> pd.DataFrame:
        """
        改良版日次スコア計算のメイン処理（全銘柄計算 + フィルタリング状態記録）
        
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
        
        logger.info(f"改良版スコア計算開始（全銘柄対象）: {len(dates)}日分")
        
        # DataFrameの日付列を文字列に統一
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        for date in tqdm(dates, desc="改良版スコア計算中"):
            date_str = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
            
            # その日のデータを取得
            daily_df = df[df['date'] == date_str].copy()
            if len(daily_df) == 0:
                logger.warning(f"日付 {date_str} のデータがありません")
                continue
            
            logger.info(f"日付 {date_str}: {len(daily_df)}銘柄のデータを処理中（全銘柄対象）")
            
            # ステップ1: 各ファクタースコア計算（全銘柄対象）
            try:
                # Valueスコア計算
                value_df = self.calculate_improved_value_score(df, date_str)
                if not value_df.empty:
                    value_df = value_df.sort_values(['symbol']).drop_duplicates(subset=['symbol'], keep='last')
                value_data = value_df.set_index('symbol') if not value_df.empty else pd.DataFrame()
                
                # Growthスコア計算
                growth_df = self.calculate_improved_growth_score(df, date_str)
                if not growth_df.empty:
                    growth_df = growth_df.sort_values(['symbol']).drop_duplicates(subset=['symbol'], keep='last')
                growth_data = growth_df.set_index('symbol') if not growth_df.empty else pd.DataFrame()
                
                # Qualityスコア計算
                quality_df = self.calculate_improved_quality_score(df, date_str)
                if not quality_df.empty:
                    quality_df = quality_df.sort_values(['symbol']).drop_duplicates(subset=['symbol'], keep='last')
                quality_data = quality_df.set_index('symbol') if not quality_df.empty else pd.DataFrame()
                
                # Momentumスコア計算
                momentum_df = self.calculate_improved_momentum_score(df, date_str)
                if not momentum_df.empty:
                    momentum_df = momentum_df.sort_values(['symbol']).drop_duplicates(subset=['symbol'], keep='last')
                momentum_data = momentum_df.set_index('symbol') if not momentum_df.empty else pd.DataFrame()
                
                # マクロ・セクタースコア計算
                macro_df = self.calculate_macro_sector_score(df, date_str)
                if not macro_df.empty:
                    macro_df = macro_df.sort_values(['symbol']).drop_duplicates(subset=['symbol'], keep='last')
                macro_data = macro_df.set_index('symbol') if not macro_df.empty else pd.DataFrame()
                
            except Exception as e:
                logger.error(f"日付 {date_str} のスコア計算でエラー: {e}")
                continue
            
            # ステップ2: 総合スコア計算と結果記録（全銘柄対象）
            processed_count = 0
            error_count = 0
            
            for idx, row in daily_df.iterrows():
                try:
                    symbol = row['symbol']
                    
                    # フィルタリング状態を判定
                    is_quality_growth_filtered = self.check_quality_growth_filter(row)
                    is_value_trap_filtered = self.check_value_trap_filter(row)
                    
                    # 各ファクタースコア取得（0点ベース）
                    value_score = 0.0
                    growth_score = 0.0
                    quality_score = 0.0
                    momentum_score = 0.0
                    macro_sector_score = 0.0
                    
                    # 簡易スコア計算（weightから取得）
                    weights = self.factor_weights
                    
                    # 改良版スコア計算（多指標合算・固定バフ撤廃）
                    
                    # Valueスコア（多指標合算）
                    w_v = weights['value']['total_weight']  # 20
                    comp_v = weights['value']['components']
                    
                    per_z = self._get_scalar(value_data, symbol, 'per_z', default=0.0, prefer='last', logger=logger)
                    fcf_z = self._get_scalar(value_data, symbol, 'fcf_yield_z', default=0.0, prefer='last', logger=logger)
                    pbr_z = self._get_scalar(value_data, symbol, 'pbr_z', default=0.0, prefer='last', logger=logger)
                    per_logi = self._get_scalar(value_data, symbol, 'per_logistic', default=0.0, prefer='last', logger=logger)
                    
                    def z_to_score(z):
                        """Z-scoreを0-1スコアに変換"""
                        z = max(-3.0, min(3.0, z))
                        return (z + 3.0) / 6.0
                    
                    value_score = min(w_v,
                        z_to_score(per_z) * comp_v['per_z_score'] +
                        per_logi * comp_v['per_logistic_score'] +
                        z_to_score(fcf_z) * comp_v['fcf_yield_z_score'] +
                        z_to_score(pbr_z) * comp_v['pbr_z_score']
                    )
                    
                    # Growthスコア（多指標合算）
                    w_g = weights['growth']['total_weight']  # 25
                    comp_g = weights['growth']['components']
                    
                    eps_3y_z = self._get_scalar(growth_data, symbol, 'eps_cagr_3y_z', default=0.0, prefer='last', logger=logger)
                    eps_5y_z = self._get_scalar(growth_data, symbol, 'eps_cagr_5y_z', default=0.0, prefer='last', logger=logger)
                    rev_3y_z = self._get_scalar(growth_data, symbol, 'revenue_cagr_3y_z', default=0.0, prefer='last', logger=logger)
                    rev_5y_z = self._get_scalar(growth_data, symbol, 'revenue_cagr_5y_z', default=0.0, prefer='last', logger=logger)
                    consistency_z = self._get_scalar(growth_data, symbol, 'growth_consistency_z', default=0.0, prefer='last', logger=logger)
                    
                    growth_score = min(w_g,
                        z_to_score(eps_3y_z) * comp_g['eps_cagr_3y_z'] +
                        z_to_score(eps_5y_z) * comp_g['eps_cagr_5y_z'] +
                        z_to_score(rev_3y_z) * comp_g['revenue_cagr_3y_z'] +
                        z_to_score(rev_5y_z) * comp_g['revenue_cagr_5y_z'] +
                        z_to_score(consistency_z) * comp_g['growth_consistency_z']
                    )
                    
                    # Qualityスコア（多指標合算）
                    w_q = weights['quality']['total_weight']  # 25
                    comp_q = weights['quality']['components']
                    
                    roic_z = self._get_scalar(quality_data, symbol, 'roic_z', default=0.0, prefer='last', logger=logger)
                    roe_z = self._get_scalar(quality_data, symbol, 'roe_z', default=0.0, prefer='last', logger=logger)
                    debt_z = self._get_scalar(quality_data, symbol, 'debt_equity_z', default=0.0, prefer='last', logger=logger)
                    altman_z = self._get_scalar(quality_data, symbol, 'altman_z_score', default=0.0, prefer='last', logger=logger)
                    piotroski_z = self._get_scalar(quality_data, symbol, 'piotroski_proxy_z', default=0.0, prefer='last', logger=logger)
                    cash_z = self._get_scalar(quality_data, symbol, 'cfo_net_income_z', default=0.0, prefer='last', logger=logger)
                    
                    quality_score = min(w_q,
                        z_to_score(roic_z) * comp_q['roic_z'] +
                        z_to_score(roe_z) * comp_q['roe_z'] +
                        z_to_score(debt_z) * comp_q['debt_equity_z'] +
                        min(1.0, altman_z / 4.0) * comp_q['altman_z_proxy'] +  # 0-4スケールを0-1に正規化
                        z_to_score(piotroski_z) * comp_q['piotroski_proxy'] +
                        z_to_score(cash_z) * comp_q['cash_conversion_z']
                    )
                    
                    # Momentumスコア（5コンポーネント合算）
                    w_m = weights['momentum']['total_weight']  # 15
                    comp_m = weights['momentum']['components']
                    
                    golden_cross = self._get_scalar(momentum_data, symbol, 'golden_cross', default=0.0, prefer='last', logger=logger)
                    rsi_score = self._get_scalar(momentum_data, symbol, 'rsi_score', default=0.0, prefer='last', logger=logger)
                    macd_hist = self._get_scalar(momentum_data, symbol, 'macd_hist_score', default=0.0, prefer='last', logger=logger)
                    vol_adj_z = self._get_scalar(momentum_data, symbol, 'vol_adj_momentum_z', default=0.0, prefer='last', logger=logger)
                    rel_strength = self._get_scalar(momentum_data, symbol, 'relative_strength', default=0.0, prefer='last', logger=logger)
                    
                    momentum_score = min(w_m,
                        golden_cross * comp_m['golden_cross'] +
                        (rsi_score / 3.0) * comp_m['rsi_score'] +  # 0-3スケールを0-1に正規化
                        macd_hist * comp_m['macd_hist'] +
                        z_to_score(vol_adj_z) * comp_m['vol_adj_momentum'] +
                        z_to_score(rel_strength) * comp_m['relative_strength']
                    )
                    
                    # Macro/Sectorスコア（両コンポーネント合算・固定バフ撤廃）
                    w_macro = weights['macro_sector']['total_weight']  # 15
                    comp_macro = weights['macro_sector']['components']
                    
                    tail_wind = self._get_scalar(macro_data, symbol, 'tail_wind_score', default=0.0, prefer='last', logger=logger)
                    sector_rot = self._get_scalar(macro_data, symbol, 'sector_rotation_score', default=0.0, prefer='last', logger=logger)
                    
                    macro_sector_score = min(w_macro,
                        (tail_wind / 10.0) * comp_macro['tail_wind'] +  # 0-10スケール
                        (sector_rot / 5.0) * comp_macro['sector_rotation']  # 0-5スケール
                    )
                    
                    # 総合スコア計算（100点満点）
                    total_score = value_score + growth_score + quality_score + momentum_score + macro_sector_score
                    total_score = self.safe_clip_score(total_score, 0, 100)
                    
                    # 詳細スコア（簡易版）
                    per_score = value_score * 0.6
                    fcf_yield_score = value_score * 0.25
                    ev_ebitda_score = 0.0
                    eps_cagr_score = growth_score * 0.6
                    revenue_cagr_score = growth_score * 0.3
                    growth_consistency_score = growth_score * 0.1
                    roic_score = quality_score * 0.25
                    roe_score = quality_score * 0.15
                    debt_equity_score = quality_score * 0.15
                    altman_z_score = quality_score * 0.15
                    piotroski_f_score = quality_score * 0.15
                    cash_conversion_score = quality_score * 0.15
                    golden_cross_score = momentum_score * 0.4
                    rsi_score = momentum_score * 0.2
                    macd_hist_score = momentum_score * 0.2
                    vol_adj_momentum_score = momentum_score * 0.1
                    relative_strength_score = momentum_score * 0.1
                    tail_wind_score = macro_sector_score * 0.67
                    sector_rotation_score = macro_sector_score * 0.33
                    
                    # 結果記録（データベーススキーマに合わせて + フィルタリング状態）
                    score_record = {
                        'symbol': symbol,
                        'date': date_str,
                        'value_score': value_score,
                        'growth_score': growth_score,
                        'quality_score': quality_score,
                        'momentum_score': momentum_score,
                        'macro_sector_score': macro_sector_score,
                        'total_score': total_score,
                        # 詳細スコア
                        'per_score': per_score,
                        'fcf_yield_score': fcf_yield_score,
                        'ev_ebitda_score': ev_ebitda_score,
                        'eps_cagr_score': eps_cagr_score,
                        'revenue_cagr_score': revenue_cagr_score,
                        'growth_consistency_score': growth_consistency_score,
                        'roic_score': roic_score,
                        'roe_score': roe_score,
                        'debt_equity_score': debt_equity_score,
                        'altman_z_score': altman_z_score,
                        'piotroski_f_score': piotroski_f_score,
                        'cash_conversion_score': cash_conversion_score,
                        'golden_cross_score': golden_cross_score,
                        'rsi_score': rsi_score,
                        'macd_hist_score': macd_hist_score,
                        'vol_adj_momentum_score': vol_adj_momentum_score,
                        'relative_strength_score': relative_strength_score,
                        'tail_wind_score': tail_wind_score,
                        'sector_rotation_score': sector_rotation_score,
                        # フィルタリング状態（新機能）
                        'is_value_trap_filtered': is_value_trap_filtered,
                        'is_quality_growth_filtered': is_quality_growth_filtered,
                    }
                    
                    results.append(score_record)
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # 最初の5件のみログ出力
                        logger.warning(f"銘柄 {row['symbol']} のスコア計算エラー: {e}")
                    continue
            
            logger.info(f"日付 {date_str}: 処理完了 {processed_count}件, エラー {error_count}件")
        
        logger.info(f"改良版スコア計算完了: {len(results)}件の結果")
        
        # スコア分布監視（異常値検出）
        if results:
            self._monitor_score_distribution(results)
        
        return pd.DataFrame(results)
    
    def _monitor_score_distribution(self, results: List[Dict]) -> None:
        """
        スコア分布を監視し、異常値を検出
        
        Args:
            results: スコア計算結果のリスト
        """
        try:
            df_results = pd.DataFrame(results)
            
            # 日次分布統計
            daily_stats = df_results.groupby('date')['total_score'].agg([
                'count', 'mean', 'median', 
                lambda x: x.quantile(0.75),  # p75
                lambda x: x.quantile(0.95)   # p95
            ]).round(2)
            daily_stats.columns = ['count', 'mean', 'p50', 'p75', 'p95']
            
            # 最新日の統計
            latest_date = daily_stats.index[-1]
            latest_stats = daily_stats.loc[latest_date]
            
            logger.info(f"スコア分布監視 - 日付: {latest_date}")
            logger.info(f"  件数: {latest_stats['count']:,}")
            logger.info(f"  平均: {latest_stats['mean']:.1f}")
            logger.info(f"  中央値: {latest_stats['p50']:.1f}")
            logger.info(f"  75%: {latest_stats['p75']:.1f}")
            logger.info(f"  95%: {latest_stats['p95']:.1f}")
            
            # 異常値検出（中央値が50を大幅に超える場合）
            if latest_stats['p50'] > 60:
                logger.warning(f"⚠️ スコア中央値が異常に高い: {latest_stats['p50']:.1f} (閾値: 60)")
            
            if latest_stats['p95'] > 85:
                logger.warning(f"⚠️ スコア95%が異常に高い: {latest_stats['p95']:.1f} (閾値: 85)")
            
            # ファクター別平均スコア
            factor_means = df_results[df_results['date'] == latest_date][
                ['value_score', 'growth_score', 'quality_score', 'momentum_score', 'macro_sector_score']
            ].mean().round(2)
            
            logger.info("ファクター別平均スコア:")
            for factor, mean_score in factor_means.items():
                logger.info(f"  {factor}: {mean_score:.1f}")
            
            # 異常に高いファクターを検出
            for factor, mean_score in factor_means.items():
                if mean_score > 15:  # 各ファクターの理論最大値の75%以上
                    logger.warning(f"⚠️ {factor}が異常に高い: {mean_score:.1f}")
            
        except Exception as e:
            logger.error(f"スコア分布監視エラー: {e}")

    def save_scores(self, scores_df: pd.DataFrame) -> None:
        """
        スコアデータをdaily_scoresテーブルに保存（新しいフィルタリングカラム対応）
        
        Args:
            scores_df: 保存するスコアデータ
        """
        if scores_df.empty:
            logger.warning("保存するスコアデータがありません")
            return
        
        try:
            logger.info(f"スコアデータ保存開始: {len(scores_df)}行")
            
            # 古いデータを削除（同じ日付・銘柄の組み合わせ）
            dates = scores_df['date'].unique()
            
            logger.info(f"既存データ削除中: {len(dates)}日分")
            
            with self.engine.connect() as conn:
                for date in dates:
                    delete_query = text("""
                    DELETE FROM backtest_results.daily_scores 
                    WHERE date = :date
                    """)
                    result = conn.execute(delete_query, {"date": date})
                    logger.info(f"日付 {date}: {result.rowcount} 行を削除")
                conn.commit()
            
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
                
                try:
                    # データベースに保存（新しいカラム含む）
                    batch_df.to_sql(
                        'daily_scores',
                        self.engine,
                        schema='backtest_results',
                        if_exists='append',
                        index=False,
                        method='multi'  # バッチ挿入を有効化
                    )
                    logger.info(f"バッチ {i//batch_size + 1}: 保存完了")
                except Exception as e:
                    logger.error(f"バッチ {i//batch_size + 1} 保存エラー: {e}")
                    # 個別の行で挿入を試行
                    logger.info(f"バッチ {i//batch_size + 1}: 個別挿入を試行中...")
                    with self.engine.connect() as conn:
                        for _, row in batch_df.iterrows():
                            try:
                                # 個別の行を挿入（UPSERT）
                                insert_query = text("""
                                INSERT INTO backtest_results.daily_scores (
                                    symbol, date, value_score, growth_score, quality_score, 
                                    momentum_score, macro_sector_score, total_score,
                                    per_score, fcf_yield_score, ev_ebitda_score,
                                    eps_cagr_score, revenue_cagr_score, growth_consistency_score,
                                    roic_score, roe_score, debt_equity_score, altman_z_score,
                                    piotroski_f_score, cash_conversion_score,
                                    golden_cross_score, rsi_score, macd_hist_score,
                                    vol_adj_momentum_score, relative_strength_score,
                                    tail_wind_score, sector_rotation_score,
                                    is_value_trap_filtered, is_quality_growth_filtered
                                ) VALUES (
                                    :symbol, :date, :value_score, :growth_score, :quality_score,
                                    :momentum_score, :macro_sector_score, :total_score,
                                    :per_score, :fcf_yield_score, :ev_ebitda_score,
                                    :eps_cagr_score, :revenue_cagr_score, :growth_consistency_score,
                                    :roic_score, :roe_score, :debt_equity_score, :altman_z_score,
                                    :piotroski_f_score, :cash_conversion_score,
                                    :golden_cross_score, :rsi_score, :macd_hist_score,
                                    :vol_adj_momentum_score, :relative_strength_score,
                                    :tail_wind_score, :sector_rotation_score,
                                    :is_value_trap_filtered, :is_quality_growth_filtered
                                )
                                ON CONFLICT (symbol, date) DO UPDATE SET
                                    value_score = EXCLUDED.value_score,
                                    growth_score = EXCLUDED.growth_score,
                                    quality_score = EXCLUDED.quality_score,
                                    momentum_score = EXCLUDED.momentum_score,
                                    macro_sector_score = EXCLUDED.macro_sector_score,
                                    total_score = EXCLUDED.total_score,
                                    per_score = EXCLUDED.per_score,
                                    fcf_yield_score = EXCLUDED.fcf_yield_score,
                                    ev_ebitda_score = EXCLUDED.ev_ebitda_score,
                                    eps_cagr_score = EXCLUDED.eps_cagr_score,
                                    revenue_cagr_score = EXCLUDED.revenue_cagr_score,
                                    growth_consistency_score = EXCLUDED.growth_consistency_score,
                                    roic_score = EXCLUDED.roic_score,
                                    roe_score = EXCLUDED.roe_score,
                                    debt_equity_score = EXCLUDED.debt_equity_score,
                                    altman_z_score = EXCLUDED.altman_z_score,
                                    piotroski_f_score = EXCLUDED.piotroski_f_score,
                                    cash_conversion_score = EXCLUDED.cash_conversion_score,
                                    golden_cross_score = EXCLUDED.golden_cross_score,
                                    rsi_score = EXCLUDED.rsi_score,
                                    macd_hist_score = EXCLUDED.macd_hist_score,
                                    vol_adj_momentum_score = EXCLUDED.vol_adj_momentum_score,
                                    relative_strength_score = EXCLUDED.relative_strength_score,
                                    tail_wind_score = EXCLUDED.tail_wind_score,
                                    sector_rotation_score = EXCLUDED.sector_rotation_score,
                                    is_value_trap_filtered = EXCLUDED.is_value_trap_filtered,
                                    is_quality_growth_filtered = EXCLUDED.is_quality_growth_filtered
                                """)
                                
                                conn.execute(insert_query, {
                                    'symbol': row['symbol'],
                                    'date': row['date'],
                                    'value_score': row['value_score'],
                                    'growth_score': row['growth_score'],
                                    'quality_score': row['quality_score'],
                                    'momentum_score': row['momentum_score'],
                                    'macro_sector_score': row['macro_sector_score'],
                                    'total_score': row['total_score'],
                                    'per_score': row['per_score'],
                                    'fcf_yield_score': row['fcf_yield_score'],
                                    'ev_ebitda_score': row['ev_ebitda_score'],
                                    'eps_cagr_score': row['eps_cagr_score'],
                                    'revenue_cagr_score': row['revenue_cagr_score'],
                                    'growth_consistency_score': row['growth_consistency_score'],
                                    'roic_score': row['roic_score'],
                                    'roe_score': row['roe_score'],
                                    'debt_equity_score': row['debt_equity_score'],
                                    'altman_z_score': row['altman_z_score'],
                                    'piotroski_f_score': row['piotroski_f_score'],
                                    'cash_conversion_score': row['cash_conversion_score'],
                                    'golden_cross_score': row['golden_cross_score'],
                                    'rsi_score': row['rsi_score'],
                                    'macd_hist_score': row['macd_hist_score'],
                                    'vol_adj_momentum_score': row['vol_adj_momentum_score'],
                                    'relative_strength_score': row['relative_strength_score'],
                                    'tail_wind_score': row['tail_wind_score'],
                                    'sector_rotation_score': row['sector_rotation_score'],
                                    'is_value_trap_filtered': row['is_value_trap_filtered'],
                                    'is_quality_growth_filtered': row['is_quality_growth_filtered']
                                })
                            except Exception as row_error:
                                logger.warning(f"行 {row['symbol']} の挿入エラー: {row_error}")
                                continue
                        conn.commit()
            
            logger.info("改良版スコアデータ保存完了（フィルタリング状態含む）")
            
        except Exception as e:
            logger.error(f"スコアデータ保存エラー: {e}")
            raise


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="改良版総合スコア計算（フィルタリング統合版）")
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
        
        # スコア計算（全銘柄対象 + フィルタリング状態記録）
        scores_df = calculator.calculate_daily_scores(df, args.target_date, args.limit_days)
        
        # データベースに保存
        calculator.save_scores(scores_df)
        
        logger.info("改良版スコア計算処理完了（フィルタリング統合版）")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()