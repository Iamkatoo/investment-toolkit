#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ファクタースコア計算モジュール

以下のスコアを計算します：
- value_z: 割安性スコア（PERの対数の標準化スコア）
- growth_z: 成長性スコア（EPS成長率の標準化スコア）
- quality_z: 質的スコア（ROICの標準化スコア）
- momentum_z: モメンタムスコア（12ヶ月リターンの標準化スコア）
- risk_z: リスクスコア（30日ボラティリティの標準化スコア）
- sentiment_z: センチメントスコア（ニュースセンチメントの標準化スコア）
- composite_score: 総合スコア（0.4*value_z + 0.4*growth_z + 0.2*quality_z）

各スコアはセクター内でのデシルランク（1-10）も計算します。
"""

import pandas as pd
import numpy as np
import logging

# ログ設定
logger = logging.getLogger(__name__)


def z_score(series, reverse=False):
    """
    与えられた系列の標準化スコア（z-score）を計算
    
    Args:
        series (pd.Series): 標準化する系列
        reverse (bool): 値を反転するかどうか（大きい値が悪い指標の場合）
    
    Returns:
        pd.Series: 標準化された系列
    """
    # 欠損値の処理
    if series.isna().all():
        return pd.Series(np.nan, index=series.index)
    
    # 平均と標準偏差の計算（欠損値を除く）
    mean = series.mean()
    std = series.std()
    
    # 標準偏差が0または欠損値の場合の処理
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    
    # z-scoreの計算
    z = (series - mean) / std
    
    # 値を反転（必要な場合）
    if reverse:
        return -z
    else:
        return z


def calculate_factor_scores(valuation_daily, basic_metrics, symbols_master=None):
    """
    ファクタースコアの計算
    
    Args:
        valuation_daily (pd.DataFrame): 日次バリュエーション指標
            必須カラム: ['symbol', 'date', 'price', 'return_12m', 'vol_30d']
        basic_metrics (pd.DataFrame): 基本財務指標
            必須カラム: ['symbol', 'per', 'eps_growth', 'roe']
        symbols_master (pd.DataFrame): シンボルマスター
            必須カラム: ['symbol', 'sector']
    
    Returns:
        tuple: (factor_scores_df, factor_ranks_df)
            - factor_scores_df: 各ファクターのスコアを含むDataFrame
            - factor_ranks_df: 各ファクターのランク情報を含むDataFrame
    """
    logger.info(f"ファクタースコア計算開始: {len(valuation_daily)} 行のデータ")
    
    # 必要なカラムの存在確認
    val_cols = ['symbol', 'date', 'price', 'return_12m', 'vol_30d']
    basic_cols = ['symbol', 'per', 'eps_growth', 'roe']
    
    missing_val_cols = [col for col in val_cols if col not in valuation_daily.columns]
    missing_basic_cols = [col for col in basic_cols if col not in basic_metrics.columns]
    
    if missing_val_cols:
        logger.warning(f"valuation_dailyに必要なカラムがありません: {missing_val_cols}")
    if missing_basic_cols:
        logger.warning(f"basic_metricsに必要なカラムがありません: {missing_basic_cols}")
    
    # valuation_dailyとbasic_metricsの結合
    df = valuation_daily.merge(basic_metrics, on='symbol', how='left')
    
    # データ型をfloatに統一（decimal.Decimalとfloatの演算問題を解決）
    numeric_cols = [
        'price', 'return_6m', 'return_12m', 'high_52w_gap', 'vol_30d', 'beta_1y', 'amihud_illiquidity',
        'per', 'pbr', 'roe', 'roa', 'eps', 'eps_growth', 'revenue_growth', 'net_income_growth',
        'dividend_yield', 'fcf_yield', 'roic', 'news_sentiment'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 欠損値の確認
    for col in ['per', 'eps_growth', 'roe', 'return_12m', 'vol_30d']:
        if col in df.columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                logger.warning(f"カラム {col} に {na_count} 件の欠損値があります ({na_count/len(df):.1%})")
    
    # ファクタースコアの計算
    logger.info("各ファクターのz-scoreを計算中...")
    
    # 1. 割安性スコア (低PERが良い = 反転)
    if 'per' in df.columns:
        # PERが負の場合や異常に高い場合の処理
        df['per_adj'] = df['per'].copy()
        df.loc[df['per'] <= 0, 'per_adj'] = np.nan  # 負のPERは欠損値として扱う
        df.loc[df['per'] > 200, 'per_adj'] = 200  # 異常に高いPERは上限を設定
        
        # 対数変換してからz-score計算（正規分布に近づける）
        df['per_log'] = np.log(df['per_adj'])
        df['value_z'] = z_score(df['per_log'], reverse=True)  # 低PERが良いので反転
    else:
        df['value_z'] = np.nan
    
    # 2. 成長性スコア (高EPS成長率が良い)
    if 'eps_growth' in df.columns:
        # 異常値の処理
        df['eps_growth_adj'] = df['eps_growth'].copy()
        df.loc[df['eps_growth'] < -100, 'eps_growth_adj'] = -100  # 下限
        df.loc[df['eps_growth'] > 200, 'eps_growth_adj'] = 200    # 上限
        
        df['growth_z'] = z_score(df['eps_growth_adj'])  # 高い方が良いのでそのまま
    else:
        df['growth_z'] = np.nan
    
    # 3. 質的スコア (高ROICが良い)
    if 'roe' in df.columns:
        # 異常値の処理
        df['roic_adj'] = df['roe'].copy()
        df.loc[df['roe'] < -50, 'roic_adj'] = -50  # 下限
        df.loc[df['roe'] > 100, 'roic_adj'] = 100  # 上限
        
        df['quality_z'] = z_score(df['roic_adj'])  # 高い方が良いのでそのまま
    else:
        df['quality_z'] = np.nan
    
    # 4. モメンタムスコア (高リターンが良い)
    if 'return_12m' in df.columns:
        df['momentum_z'] = z_score(df['return_12m'])  # 高い方が良いのでそのまま
    else:
        df['momentum_z'] = np.nan
    
    # 5. リスクスコア (低ボラティリティが良い = 反転不要)
    if 'vol_30d' in df.columns:
        df['risk_z'] = z_score(df['vol_30d'])  # 低い方が良いのでそのまま
    else:
        df['risk_z'] = np.nan
    
    # 6. センチメントスコア (sentiment_scoreが既に計算されている場合)
    if 'news_sentiment' in df.columns:
        df['sentiment_z'] = z_score(df['news_sentiment'])  # 高い方が良いのでそのまま
    else:
        df['sentiment_z'] = np.nan
    
    # 7. 総合スコア (0.4*value + 0.4*growth + 0.2*quality)
    df['composite_score'] = (
        0.4 * df['value_z'] +
        0.4 * df['growth_z'] +
        0.2 * df['quality_z']
    )
    
    # factor_scores.daily用のデータフレーム作成
    factor_scores_cols = [
        'symbol', 'date', 'value_z', 'growth_z', 'quality_z',
        'momentum_z', 'risk_z', 'sentiment_z', 'composite_score'
    ]
    factor_scores_df = df[factor_scores_cols]
    
    # セクターマスターが提供されている場合はランクを計算
    if symbols_master is not None:
        logger.info("ファクターランク情報を計算中...")
        
        # symbols_masterとの結合
        df = df.merge(symbols_master[['symbol', 'sector']], on='symbol', how='left')
        
        # グローバルランク（全銘柄での順位）
        df['rank_global'] = df['composite_score'].rank(method='first').astype(int)
        
        # セクター内ランク
        df['rank_sector'] = df.groupby('sector')['composite_score'].rank(method='first').astype(int)
        
        # 各ファクターのデシル（セクター内での10分位）
        factor_columns = [
            ('value_z', 'decile_value'),
            ('growth_z', 'decile_growth'),
            ('quality_z', 'decile_quality'),
            ('momentum_z', 'decile_momentum'),
            ('risk_z', 'decile_risk'),
            ('sentiment_z', 'decile_sentiment'),
            ('composite_score', 'decile_composite')
        ]
        
        for factor_col, decile_col in factor_columns:
            # セクターごとに計算
            df[decile_col] = df.groupby('sector')[factor_col].transform(
                lambda x: pd.qcut(x.rank(method='first'), 10, labels=False, duplicates='drop') + 1
            ).astype('Int64')  # Int64は欠損値をサポートする整数型
        
        # factor_scores.rank用のデータフレーム作成
        factor_ranks_cols = [
            'symbol', 'date', 'rank_global', 'rank_sector',
            'decile_value', 'decile_growth', 'decile_quality',
            'decile_momentum', 'decile_risk', 'decile_sentiment',
            'decile_composite'
        ]
        factor_ranks_df = df[factor_ranks_cols]
    else:
        # セクターマスターがない場合は空のDataFrameを返す
        logger.warning("セクターマスターが提供されていないため、ランク情報は計算されません")
        factor_ranks_df = pd.DataFrame(columns=[
            'symbol', 'date', 'rank_global', 'rank_sector',
            'decile_value', 'decile_growth', 'decile_quality',
            'decile_momentum', 'decile_risk', 'decile_sentiment',
            'decile_composite'
        ])
    
    logger.info(f"ファクタースコア計算完了: {len(factor_scores_df)} 行のデータ")
    
    return factor_scores_df, factor_ranks_df 