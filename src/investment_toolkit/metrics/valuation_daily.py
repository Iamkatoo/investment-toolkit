#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日次バリュエーション指標計算モジュール
- return_6m: 6ヶ月リターン
- return_12m: 12ヶ月リターン
- high_52w_gap: 52週高値からの乖離率
- vol_30d: 30日ボラティリティ（年率換算）
- beta_1y: 1年ベータ（市場指数との相関）
- amihud_illiquidity: Amihudの非流動性指標
"""

import pandas as pd
import numpy as np
import logging

# ログ設定
logger = logging.getLogger(__name__)


def compute_valuation_indicators(df_prices, df_market=None):
    """
    日次価格データから各種バリュエーション指標を計算する

    Args:
        df_prices (pd.DataFrame): 日次価格データフレーム ['symbol', 'date', 'close', 'volume']
        df_market (pd.DataFrame): 市場指数データフレーム ['date', 'close']
        
    Returns:
        pd.DataFrame: 計算された指標を含むデータフレーム
    """
    logger.info(f"バリュエーション指標の計算を開始 (対象銘柄数: {df_prices['symbol'].nunique()})")
    
    # 安全性のため、コピーを作成
    df = df_prices.sort_values(['symbol', 'date']).copy()
    
    # 最低でもclose, volumeカラムが必要
    required_columns = ['symbol', 'date', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"必要なカラムが不足しています: {', '.join(missing_columns)}")
    
    # 基本リターン計算（6ヶ月、12ヶ月）
    # 営業日数指定 - 6ヶ月≈126日、12ヶ月≈252日
    df['return_6m'] = df.groupby('symbol')['close'].pct_change(126)
    df['return_12m'] = df.groupby('symbol')['close'].pct_change(252)

    # 52週高値からの乖離率
    rolling_max = (
        df.groupby('symbol')['close']
          .rolling(252, min_periods=252).max()
          .reset_index(level=0, drop=True)
    )
    df['high_52w_gap'] = (df['close'] - rolling_max) / rolling_max

    # 日次リターン計算
    df['ret'] = df.groupby('symbol')['close'].pct_change()

    # 30日ボラティリティ（年率換算）
    df['vol_30d'] = (
        df.groupby('symbol')['ret'].rolling(30, min_periods=30).std()
          .reset_index(level=0, drop=True) * np.sqrt(252)
    )

    # β値の計算（必要な場合のみ）
    df['beta_1y'] = None  # デフォルト値を設定
    if df_market is not None and 'close' in df_market.columns:
        # 市場指数の日次リターン計算
        df_market = df_market.sort_values('date')
        df_market['mkt_ret'] = df_market['close'].pct_change()
        
        # 株価データに市場リターンを結合
        df = df.merge(df_market[['date', 'mkt_ret']], on='date', how='left')
        
        # 銘柄ごとにβ（共分散÷市場分散）を計算
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            cov_values = []
            
            for i in range(len(symbol_data)):
                if i < 252:  # 初期のデータポイントには十分なデータがない
                    cov_values.append(None)
                else:
                    window = symbol_data.iloc[i-252:i+1]
                    if len(window) >= 252:
                        cov = window['ret'].cov(window['mkt_ret'])
                        var = window['mkt_ret'].var()
                        if var != 0:
                            cov_values.append(cov / var)
                        else:
                            cov_values.append(None)
                    else:
                        cov_values.append(None)
            
            # シンボルごとにベータ値を設定
            df.loc[df['symbol'] == symbol, 'beta_1y'] = cov_values
    else:
        logger.warning("市場指数データが提供されていないため、β値は計算されません")

    # Amihud非流動性指標
    # |リターン| / (出来高 × 価格)の30日平均
    df['amihud_illiquidity'] = (
        (df['ret'].abs() / (df['volume'] * df['close']))
          .groupby(df['symbol']).rolling(30, min_periods=30).mean()
          .reset_index(level=0, drop=True)
    )

    # 必要なカラムだけを抽出して返す
    result_columns = ['symbol', 'date', 'close', 'return_6m', 'return_12m', 
                     'high_52w_gap', 'vol_30d', 'beta_1y', 'amihud_illiquidity']
    
    return df[result_columns].rename(columns={'close': 'price'}) 