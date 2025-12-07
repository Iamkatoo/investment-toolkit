#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ボラティリティ指標計算モジュール
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
import logging

logger = logging.getLogger(__name__)


def calc_volatility(prices: pd.DataFrame, days=252) -> pd.Series:
    """
    株価のボラティリティを計算する（年率）
    
    Args:
        prices (pd.DataFrame): 株価データ（symbol, date, close列を含む）
        days (int): ボラティリティ計算に使用する日数（デフォルト: 252 営業日＝約1年）
        
    Returns:
        pd.Series: 各銘柄のボラティリティ
    """
    try:
        from sqlalchemy import create_engine
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

        # 全シンボルの過去データを一括で取得
        query = f"""
        SELECT symbol, date, close
        FROM fmp_data.daily_prices
        WHERE date >= (SELECT MAX(date) - {days} FROM fmp_data.daily_prices)
        ORDER BY symbol, date
        """
        df_hist = pd.read_sql(query, engine)
        df_hist.sort_values(["symbol", "date"], inplace=True)
        df_hist['prev_close'] = df_hist.groupby("symbol")['close'].shift(1)
        df_hist['return'] = df_hist['close'] / df_hist['prev_close'] - 1

        rolling_window = 20
        vol = df_hist.groupby("symbol")['return'].rolling(window=rolling_window, min_periods=1).std().reset_index(level=0, drop=True)
        df_hist['volatility'] = vol * np.sqrt(252)

        vol_series = df_hist.sort_values("date").groupby("symbol").last()['volatility']
        return vol_series
        
    except Exception as e:
        logger.error(f"ボラティリティ計算エラー: {e}")
        # エラー時は空のシリーズを返す
        return pd.Series(dtype='float64')


def calc_beta(prices: pd.DataFrame, market_index_df: pd.DataFrame, days=252) -> pd.Series:
    """
    株価のベータ値を計算する
    
    Args:
        prices (pd.DataFrame): 株価データ（symbol, date, close列を含む）
        market_index_df (pd.DataFrame): 市場インデックスデータ（symbol, date, close列を含む）
        days (int): ベータ計算に使用する日数（デフォルト: 252 営業日＝約1年）
        
    Returns:
        pd.Series: 各銘柄のベータ値
    """
    try:
        # Vectorized beta calculation
        from sqlalchemy import create_engine
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

        # 市場インデックスのシンボルを取得
        market_symbol = market_index_df['symbol'].iloc[0]

        # 取得対象銘柄: 市場指数以外
        symbols = [s for s in prices['symbol'].unique() if s != market_symbol]
        # クエリ対象は銘柄リストに市場指数も加える
        symbols_in_query = symbols + [market_symbol]
        symbols_str = ", ".join(f"'{s}'" for s in symbols_in_query)

        # 全対象銘柄の過去データを一括取得
        query = f"""
        SELECT symbol, date, close
        FROM fmp_data.daily_prices
        WHERE date >= (SELECT MAX(date) - {days} FROM fmp_data.daily_prices)
        AND symbol IN ({symbols_str})
        ORDER BY symbol, date
        """
        df_hist = pd.read_sql(query, engine)
        df_hist.sort_values(["symbol", "date"], inplace=True)
        df_hist['prev_close'] = df_hist.groupby("symbol")['close'].shift(1)
        df_hist['return'] = df_hist['close'] / df_hist['prev_close'] - 1

        # Pivot: 行=日付, 列=銘柄, 値=リターン
        pivot = df_hist.pivot(index="date", columns="symbol", values="return")
        market_ret = pivot[market_symbol]
        market_var = market_ret.var()

        # Vectorized covariance calculation
        demeaned = pivot - pivot.mean()
        market_demeaned = market_ret - market_ret.mean()
        n = pivot.notnull().sum() - 1
        covs = (demeaned.multiply(market_demeaned, axis=0)).sum() / n

        betas = covs / market_var

        # 市場指数のベータは除外して返す
        result = betas.drop(labels=[market_symbol], errors='ignore')
        return result
        
    except Exception as e:
        logger.error(f"ベータ計算エラー: {e}")
        # エラー時は空のシリーズを返す
        return pd.Series(dtype='float64') 