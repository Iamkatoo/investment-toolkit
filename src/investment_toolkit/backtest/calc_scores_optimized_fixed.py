#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修正版オリジナル総合スコア計算（個別スコア項目も含む）
2024年データの個別スコア項目null問題を解決
"""

import os
import sys
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine, text

# 警告を抑制
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/score_calculation_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FixedScoreCalculator:
    """修正版スコア計算器（個別スコア項目も含む）"""
    
    def __init__(self, config_path: str = "config/score_weights.yaml"):
        """初期化"""
        logger.info("修正版スコア計算器を初期化中...")
        self.config = self._load_config(config_path)
        self.engine = self._create_db_engine()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"設定ファイル読み込み完了: {config_path}")
            return config
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            raise
    
    def _create_db_engine(self) -> Any:
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
    
    def get_date_range(self, start_date: str = "2015-01-01", end_date: Optional[str] = None) -> List[str]:
        """処理対象日付範囲を取得"""
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # データベースから営業日を取得
        query = """
        SELECT DISTINCT date 
        FROM backtest_results.vw_daily_master 
        WHERE date >= %s AND date <= %s 
        ORDER BY date
        """
        
        try:
            df = pd.read_sql(query, self.engine, params=(start_date, end_date))
            
            # dateカラムの型を確認して適切に変換
            if df['date'].dtype == 'object':
                # 文字列またはdate型の場合
                dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').tolist()
            else:
                # 既にdatetime型の場合
                dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
            
            logger.info(f"対象日付範囲: {len(dates)}日分 ({dates[0]} ～ {dates[-1]})")
            return dates
        except Exception as e:
            logger.error(f"日付範囲取得エラー: {e}")
            return []
    
    def load_batch_data(self, dates: List[str]) -> pd.DataFrame:
        """バッチデータ読み込み"""
        if not dates:
            return pd.DataFrame()
        
        # SQLでIN句を使用（パフォーマンス向上）
        date_list = "', '".join(dates)
        
        query = f"""
        SELECT 
            vm.symbol, vm.date, vm.close, 
            vm.per, vm.fcf_yield, vm.pbr,
            vm.eps_cagr_3y, vm.eps_cagr_5y, vm.revenue_cagr_3y, vm.revenue_cagr_5y,
            vm.roic, vm.roe, vm.debt_to_equity, vm.cfo_to_net_income,
            vm.sma_20, vm.sma_40, vm.rsi_14, vm.macd_hist,
            vm.atr_14, vm.market_cap,
            vm.vix_value, vm.dgs10_value, vm.dgs2_value, vm.cpi_value, vm.ffr_value,
            vm.raw_sector as sector, vm.raw_industry as industry,
            vm.industry_avg_close,
            cvm.peg_3y, cvm.peg_5y, cvm.pegy_3y, cvm.pegy_5y,
            cvm.garp_flag_3y, cvm.garp_flag_5y,
            cvm.ev_ebitda, cvm.earnings_yield, cvm.altman_z, cvm.piotroski_f
        FROM backtest_results.vw_daily_master vm
        LEFT JOIN calculated_metrics.composite_valuation_metrics cvm 
            ON vm.symbol = cvm.symbol AND vm.date = cvm.as_of_date
        WHERE vm.date IN ('{date_list}')
        AND vm.symbol IS NOT NULL
        ORDER BY vm.date, vm.symbol
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"バッチデータ読み込み完了: {len(df)}行")
            return df
        except Exception as e:
            logger.error(f"バッチデータ読み込みエラー: {e}")
            return pd.DataFrame()
    
    def calculate_percentiles(self, df: pd.DataFrame, date: str) -> Dict[str, float]:
        """パーセンタイル計算（その日の全銘柄）"""
        date_df = df[df['date'].astype(str) == date].copy()
        
        if len(date_df) == 0:
            return {}
        
        percentiles = {}
        
        # 数値列のみを対象（実際のビューのカラムに合わせる）
        numeric_cols = ['per', 'fcf_yield', 'pbr', 'roic', 'roe', 'debt_to_equity', 'rsi_14']
        
        for col in numeric_cols:
            if col in date_df.columns:
                valid_values = date_df[col].dropna()
                if len(valid_values) > 0:
                    percentiles[f'{col}_percentile'] = valid_values.rank(pct=True)
                    # 各銘柄のパーセンタイルを辞書で保存
                    percentiles[col] = dict(zip(date_df['symbol'], 
                                              date_df[col].rank(pct=True, na_option='keep')))
        
        return percentiles
    
    def calculate_detailed_scores(self, row: pd.Series, percentiles: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """詳細スコア計算（個別スコア項目も含む）"""
        symbol = row['symbol']
        scores = {}
        
        # =============================================================================
        # Value Score (20点) - 個別項目も記録
        # =============================================================================
        
        # PER スコア (6点)
        per_rank = percentiles.get('per', {}).get(symbol, 0.5)
        if pd.notna(per_rank):
            per_score = (1 - per_rank) * 6  # 反転（低いほど良い）
        else:
            per_score = 0
        scores['per_score'] = per_score
        
        # FCF Yield スコア (7点)
        fcf_yield_rank = percentiles.get('fcf_yield', {}).get(symbol, 0.5)
        if pd.notna(fcf_yield_rank):
            fcf_yield_score = fcf_yield_rank * 7
        else:
            fcf_yield_score = 0
        scores['fcf_yield_score'] = fcf_yield_score
        
        # PBR スコア (7点) - EV/EBITDAの代替
        pbr_rank = percentiles.get('pbr', {}).get(symbol, 0.5)
        if pd.notna(pbr_rank):
            ev_ebitda_score = (1 - pbr_rank) * 7  # 反転（低いほど良い）
        else:
            ev_ebitda_score = 0
        scores['ev_ebitda_score'] = ev_ebitda_score  # 互換性のためev_ebitda_scoreとして保存
        
        value_score = per_score + fcf_yield_score + ev_ebitda_score
        scores['value_score'] = value_score
        
        # =============================================================================
        # Growth Score (20点) - PEGベース段階的加点システム導入
        # =============================================================================
        
        # まずpeg_3yデータの有無を確認（必須条件）
        peg_3y = row.get('peg_3y', None)
        if pd.isna(peg_3y):
            # PEG 3年データがない場合は成長性評価対象外（0点）
            scores['eps_cagr_score'] = 0
            scores['revenue_cagr_score'] = 0
            scores['growth_consistency_score'] = 0
            scores['peg_quality_bonus'] = 0
            growth_score = 0
        else:
            # EPS CAGR スコア (8点) - 従来10点から縮小
            eps_3y = row.get('eps_cagr_3y', 0) or 0
            eps_5y = row.get('eps_cagr_5y', 0) or 0
            eps_avg = (eps_3y + eps_5y) / 2 if eps_3y != 0 and eps_5y != 0 else max(eps_3y, eps_5y)
            eps_cagr_score = min(8, max(0, eps_avg * 0.4)) if eps_avg > 0 else 0
            scores['eps_cagr_score'] = eps_cagr_score
            
            # Revenue CAGR スコア (4点) - 従来5点から縮小
            rev_3y = row.get('revenue_cagr_3y', 0) or 0
            rev_5y = row.get('revenue_cagr_5y', 0) or 0
            rev_avg = (rev_3y + rev_5y) / 2 if rev_3y != 0 and rev_5y != 0 else max(rev_3y, rev_5y)
            revenue_cagr_score = min(4, max(0, rev_avg * 0.26)) if rev_avg > 0 else 0
            scores['revenue_cagr_score'] = revenue_cagr_score
            
            # Growth Consistency スコア (3点) - 従来5点から縮小
            if eps_3y != 0 and eps_5y != 0:
                consistency = abs(eps_5y - eps_3y)
                growth_consistency_score = max(0, 3 * (1 - consistency / 20))  # 差が小さいほど高得点
            else:
                growth_consistency_score = 0
            scores['growth_consistency_score'] = growth_consistency_score
            
            # PEG Quality Bonus (5点) - 新規追加
            peg_quality_bonus = 0
            peg_5y = row.get('peg_5y', None)
            garp_flag_3y = row.get('garp_flag_3y', False)
            garp_flag_5y = row.get('garp_flag_5y', False)
            
            # Tier2条件チェック (peg_3y <= 2.0 and eps_cagr_3y > 0)
            tier2_condition = (
                peg_3y <= 2.0 and 
                eps_3y > 0
            )
            
            # Tier3条件チェック (より厳格)
            tier3_condition = (
                peg_3y <= 1.5 and
                pd.notna(peg_5y) and peg_5y <= 1.5 and
                eps_3y > 0 and
                garp_flag_3y == True
            )
            
            if tier3_condition:
                peg_quality_bonus = 5  # 最高品質GARP銘柄
            elif tier2_condition:
                peg_quality_bonus = 3  # 良質成長銘柄
            else:
                peg_quality_bonus = 1  # 成長データは存在するが品質は普通
            
            scores['peg_quality_bonus'] = peg_quality_bonus
            
            # Growth総合スコア
            growth_score = eps_cagr_score + revenue_cagr_score + growth_consistency_score + peg_quality_bonus
        
        # 最終的にgrowth_scoreを記録
        scores['growth_score'] = growth_score
        
        # =============================================================================
        # Quality Score (25点) - 個別項目も記録
        # =============================================================================
        
        # ROIC スコア (8点)
        roic = row.get('roic', 0) or 0
        roic_score = min(8, max(0, roic * 0.4)) if roic > 0 else 0
        scores['roic_score'] = roic_score
        
        # ROE スコア (4点)
        roe = row.get('roe', 0) or 0
        roe_score = min(4, max(0, roe * 0.2)) if roe > 0 else 0
        scores['roe_score'] = roe_score
        
        # Debt/Equity スコア (4点)
        debt_equity = row.get('debt_to_equity', 0) or 0
        if debt_equity > 0:
            debt_equity_score = max(0, 4 * (1 - min(1, debt_equity / 0.5)))
        else:
            debt_equity_score = 4  # 負債がない場合は満点
        scores['debt_equity_score'] = debt_equity_score
        
        # Altman Z スコア (3点) - 簡素化（データがないため固定）
        altman_z_score = 1.5  # 暫定値
        scores['altman_z_score'] = altman_z_score
        
        # Piotroski F スコア (6点) - 簡素化（データがないため固定）
        piotroski_f_score = 3.0  # 暫定値
        scores['piotroski_f_score'] = piotroski_f_score
        
        # Cash Conversion スコア (4点)
        cfo_ratio = row.get('cfo_to_net_income', 0) or 0
        cash_conversion_score = min(4, max(0, cfo_ratio * 4)) if cfo_ratio > 0 else 0
        scores['cash_conversion_score'] = cash_conversion_score
        
        quality_score = (roic_score + roe_score + debt_equity_score + 
                        altman_z_score + piotroski_f_score + cash_conversion_score)
        scores['quality_score'] = quality_score
        
        # =============================================================================
        # Momentum Score (20点) - 個別項目も記録
        # =============================================================================
        
        # Golden Cross スコア (5点)
        sma_20 = row.get('sma_20', 0) or 0
        sma_40 = row.get('sma_40', 0) or 0
        golden_cross_score = 5 if sma_20 > sma_40 and sma_20 > 0 else 0
        scores['golden_cross_score'] = golden_cross_score
        
        # RSI スコア (3点)
        rsi = row.get('rsi_14', 50) or 50
        if 50 <= rsi <= 70:
            rsi_score = 3
        elif 70 < rsi <= 80:
            rsi_score = 1
        else:
            rsi_score = 0
        scores['rsi_score'] = rsi_score
        
        # MACD Histogram スコア (2点)
        macd_hist = row.get('macd_hist', 0) or 0
        macd_hist_score = 2 if macd_hist > 0 else 0
        scores['macd_hist_score'] = macd_hist_score
        
        # Vol-adjusted Momentum スコア (5点)
        close = row.get('close', 0) or 0
        atr_14 = row.get('atr_14', 1) or 1  # ゼロ除算回避
        if sma_20 > 0 and atr_14 > 0:
            momentum_ratio = (close / sma_20 - 1) / (atr_14 / close)
            vol_adj_momentum_score = min(5, max(0, momentum_ratio * 2.5))
        else:
            vol_adj_momentum_score = 0
        scores['vol_adj_momentum_score'] = vol_adj_momentum_score
        
        # Relative Strength スコア (5点) - 簡素化
        # 業界平均との比較で代替
        industry_avg = row.get('industry_avg_close', close) or close
        if industry_avg > 0:
            relative_strength = close / industry_avg
            relative_strength_score = min(5, max(0, (relative_strength - 1) * 10))
        else:
            relative_strength_score = 2.5  # 中間値
        scores['relative_strength_score'] = relative_strength_score
        
        momentum_score = (golden_cross_score + rsi_score + macd_hist_score + 
                         vol_adj_momentum_score + relative_strength_score)
        scores['momentum_score'] = momentum_score
        
        # =============================================================================
        # Macro & Sector Score (15点) - 個別項目も記録
        # =============================================================================
        
        # Tail Wind スコア (10点)
        vix = row.get('vix_value', 25) or 25
        dgs10 = row.get('dgs10_value', 0) or 0
        dgs2 = row.get('dgs2_value', 0) or 0
        cpi_value = row.get('cpi_value', 0) or 0
        ffr_value = row.get('ffr_value', 0) or 0
        
        vix_score = 2 if vix < 18 else 0
        yield_spread_score = 2 if (dgs10 - dgs2) > 0.8 else 0
        
        # CPI下落かつFed上昇の判定（簡素化）
        macro_bonus = 1 if cpi_value < 3 and ffr_value > 2 else 0
        
        tail_wind_score = vix_score + yield_spread_score + macro_bonus
        scores['tail_wind_score'] = tail_wind_score
        
        # Sector Rotation スコア (5点) - 簡素化
        sector = row.get('sector', '')
        # 実際のセクターローテーション計算は複雑なので、ここでは簡素化
        sector_rotation_score = 2.5  # 暫定値
        scores['sector_rotation_score'] = sector_rotation_score
        
        macro_sector_score = tail_wind_score + sector_rotation_score
        scores['macro_sector_score'] = macro_sector_score
        
        # =============================================================================
        # 総合スコア計算
        # =============================================================================
        total_score = value_score + growth_score + quality_score + momentum_score + macro_sector_score
        total_score = max(0, min(100, total_score))
        
        return total_score, scores
    
    def process_batch(self, dates: List[str]) -> pd.DataFrame:
        """バッチ処理（個別スコア項目も含む）"""
        logger.info(f"修正版バッチ処理開始: {len(dates)}日分")
        
        # バッチデータ読み込み
        df = self.load_batch_data(dates)
        if len(df) == 0:
            logger.warning("バッチデータが空です")
            return pd.DataFrame()
        
        results = []
        
        for date in dates:
            date_str = date if isinstance(date, str) else date.strftime('%Y-%m-%d')
            
            # その日のデータを取得
            daily_df = df[df['date'].astype(str) == date_str].copy()
            
            if len(daily_df) == 0:
                logger.warning(f"日付 {date_str} のデータが見つかりません")
                continue
            
            logger.info(f"日付 {date_str}: {len(daily_df)}銘柄を処理中")
            
            # パーセンタイル計算
            percentiles = self.calculate_percentiles(df, date_str)
            
            # 各銘柄のスコア計算
            processed_count = 0
            for idx, row in daily_df.iterrows():
                try:
                    total_score, score_details = self.calculate_detailed_scores(row, percentiles)
                    
                    # 結果記録（すべての個別スコア項目を含む）
                    score_record = {
                        'symbol': row['symbol'],
                        'date': date_str,
                        'total_score': total_score,
                        **score_details  # 個別スコア項目をすべて展開
                    }
                    
                    results.append(score_record)
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"スコア計算エラー {row['symbol']} {date_str}: {e}")
                    continue
            
            logger.info(f"日付 {date_str}: {processed_count}件のスコアを計算")
        
        logger.info(f"修正版バッチ処理完了: {len(results)}件のスコアを計算")
        
        # データベースに保存
        if len(results) > 0:
            self.save_to_database(results)
        
        return pd.DataFrame(results)
    
    def save_to_database(self, results: List[Dict]) -> None:
        """データベースに保存（すべての個別スコア項目を含む）"""
        df = pd.DataFrame(results)
        
        # 重複削除
        df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
        
        logger.info(f"データベース保存開始: {len(df)}行")
        
        try:
            # 小さなバッチに分けて保存
            save_batch_size = 100
            for i in range(0, len(df), save_batch_size):
                batch_df = df.iloc[i:i + save_batch_size].copy()
                
                # created_atカラムを追加
                batch_df['created_at'] = datetime.now()
                
                # 一時テーブルに保存
                temp_table = 'temp_daily_scores_fixed'
                batch_df.to_sql(
                    temp_table,
                    self.engine,
                    if_exists='replace',
                    index=False,
                    method='multi'
                )
                
                # UPSERT実行（すべてのカラムを対象、peg_quality_bonusは一旦除外）
                upsert_query = f"""
                INSERT INTO backtest_results.daily_scores (
                    symbol, date, total_score, value_score, growth_score, quality_score, 
                    momentum_score, macro_sector_score,
                    per_score, fcf_yield_score, ev_ebitda_score,
                    eps_cagr_score, revenue_cagr_score, growth_consistency_score,
                    roic_score, roe_score, debt_equity_score, altman_z_score, 
                    piotroski_f_score, cash_conversion_score,
                    golden_cross_score, rsi_score, macd_hist_score, 
                    vol_adj_momentum_score, relative_strength_score,
                    tail_wind_score, sector_rotation_score, created_at
                )
                SELECT 
                    symbol, date::date, total_score, value_score, growth_score, quality_score,
                    momentum_score, macro_sector_score,
                    per_score, fcf_yield_score, ev_ebitda_score,
                    eps_cagr_score, revenue_cagr_score, growth_consistency_score,
                    roic_score, roe_score, debt_equity_score, altman_z_score,
                    piotroski_f_score, cash_conversion_score,
                    golden_cross_score, rsi_score, macd_hist_score,
                    vol_adj_momentum_score, relative_strength_score,
                    tail_wind_score, sector_rotation_score, created_at
                FROM {temp_table}
                ON CONFLICT (symbol, date) 
                DO UPDATE SET
                    total_score = EXCLUDED.total_score,
                    value_score = EXCLUDED.value_score,
                    growth_score = EXCLUDED.growth_score,
                    quality_score = EXCLUDED.quality_score,
                    momentum_score = EXCLUDED.momentum_score,
                    macro_sector_score = EXCLUDED.macro_sector_score,
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
                    created_at = EXCLUDED.created_at
                """
                
                with self.engine.connect() as conn:
                    conn.execute(text(upsert_query))
                    conn.commit()
                
                # 一時テーブル削除
                with self.engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
                    conn.commit()
            
            logger.info(f"データベース保存完了: {len(df)}行")
            
        except Exception as e:
            logger.error(f"データベース保存エラー: {e}")
            raise
    
    def calculate_scores_range(self, start_date: str = "2024-03-17", end_date: Optional[str] = None, limit_days: Optional[int] = None) -> None:
        """指定期間のスコア計算"""
        logger.info("修正版スコア計算開始")
        
        # 処理対象日付を取得
        all_dates = self.get_date_range(start_date, end_date)
        
        if limit_days and len(all_dates) > limit_days:
            all_dates = all_dates[-limit_days:]
            logger.info(f"期間制限により最新 {limit_days} 日分のみ処理")
        
        logger.info(f"処理対象: {len(all_dates)}日分 ({all_dates[0]} ～ {all_dates[-1]})")
        
        if len(all_dates) == 0:
            logger.warning("処理対象日付が見つかりません")
            return
        
        # バッチサイズを小さく設定
        batch_size = 30
        
        # 日付をバッチに分割
        batches = [all_dates[i:i + batch_size] for i in range(0, len(all_dates), batch_size)]
        
        # バッチ処理
        for i, batch in enumerate(batches):
            logger.info(f"バッチ {i+1}/{len(batches)} を処理中...")
            
            try:
                self.process_batch(batch)
                logger.info(f"バッチ {i+1}/{len(batches)} 完了")
                
            except Exception as e:
                logger.error(f"バッチ {i+1} でエラー: {e}")
                continue
        
        logger.info("修正版スコア計算完了")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="修正版スコア計算（個別スコア項目も含む）")
    parser.add_argument("--start-date", default="2024-03-17", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/score_weights.yaml", help="設定ファイルパス")
    parser.add_argument("--limit-days", type=int, help="処理する日数制限（テスト用）")
    
    args = parser.parse_args()
    
    try:
        # ログディレクトリ作成
        os.makedirs("logs", exist_ok=True)
        
        # 修正版スコア計算器を初期化
        calculator = FixedScoreCalculator(args.config)
        
        # スコア計算実行
        calculator.calculate_scores_range(args.start_date, args.end_date, args.limit_days)
        
        logger.info("修正版スコア計算処理完了")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 