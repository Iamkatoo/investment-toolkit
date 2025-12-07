#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基本的な投資指標計算モジュール
"""

import pandas as pd
import numpy as np


def calc_eps(financials: pd.DataFrame) -> pd.Series:
    """
    一株当たり利益（EPS）を計算または取得

    Args:
        financials (pd.DataFrame): 財務データ（eps列を含む）

    Returns:
        pd.Series: 各銘柄のEPS
    """
    return financials['eps']


def calc_market_cap(market_cap_df: pd.DataFrame) -> pd.Series:
    """
    時価総額を取得

    Args:
        market_cap_df (pd.DataFrame): 時価総額データ（market_cap列を含む）

    Returns:
        pd.Series: 各銘柄の時価総額
    """
    return market_cap_df['market_cap']


def calc_revenue_growth(financials: pd.DataFrame) -> pd.Series:
    """
    売上高成長率（前年同期比）を計算

    Args:
        financials (pd.DataFrame): 財務データ（symbol, date, revenue, prev_year_revenue列を含む）

    Returns:
        pd.Series: 各銘柄の売上高成長率
    """
    # 前年の売上データがある場合のみ計算
    mask = (financials['prev_year_revenue'] > 0)
    growth = pd.Series(index=financials.index, dtype='float64')
    
    # 成長率計算: (今期売上 - 前期売上) / 前期売上
    growth[mask] = (financials.loc[mask, 'revenue'] - financials.loc[mask, 'prev_year_revenue']) / financials.loc[mask, 'prev_year_revenue']
    
    return growth


def calc_eps_growth(financials: pd.DataFrame) -> pd.Series:
    """
    EPS成長率（前年同期比）を計算

    Args:
        financials (pd.DataFrame): 財務データ（symbol, date, eps, prev_year_eps列を含む）

    Returns:
        pd.Series: 各銘柄のEPS成長率
    """
    # 前年のEPSデータがある場合、かつ前年EPSが正の場合のみ計算
    mask = (financials['prev_year_eps'] > 0)
    growth = pd.Series(index=financials.index, dtype='float64')
    
    # 成長率計算: (今期EPS - 前期EPS) / 前期EPS
    growth[mask] = (financials.loc[mask, 'eps'] - financials.loc[mask, 'prev_year_eps']) / financials.loc[mask, 'prev_year_eps']
    
    return growth


def calc_net_income_growth(financials: pd.DataFrame) -> pd.Series:
    """
    純利益成長率（前年同期比）を計算

    Args:
        financials (pd.DataFrame): 財務データ（symbol, date, net_income, prev_year_net_income列を含む）

    Returns:
        pd.Series: 各銘柄の純利益成長率
    """
    # 前年の純利益データがある場合、かつ前年純利益が正の場合のみ計算
    mask = (financials['prev_year_net_income'] > 0)
    growth = pd.Series(index=financials.index, dtype='float64')
    
    # 成長率計算: (今期純利益 - 前期純利益) / 前期純利益
    growth[mask] = (financials.loc[mask, 'net_income'] - financials.loc[mask, 'prev_year_net_income']) / financials.loc[mask, 'prev_year_net_income']
    
    return growth


def calc_dividend_yield(financials: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    配当利回りを計算

    Args:
        financials (pd.DataFrame): 財務データ（symbol, date, dividend列を含む）
        prices (pd.DataFrame): 価格データ（symbol, date, close列を含む）

    Returns:
        pd.Series: 各銘柄の配当利回り
    """
    # 財務データと価格データをマージ
    merged_df = pd.merge(
        financials[['symbol', 'dividend']],
        prices[['symbol', 'close']],
        on='symbol',
        how='inner'
    )
    
    # 株価が0より大きい場合のみ計算
    mask = (merged_df['close'] > 0)
    dividend_yield = pd.Series(index=merged_df.index, dtype='float64')
    
    # 配当利回り計算: 年間配当金 / 株価
    dividend_yield[mask] = merged_df.loc[mask, 'dividend'] / merged_df.loc[mask, 'close']
    
    return dividend_yield


def calc_fcf_yield(financials: pd.DataFrame, market_cap_df: pd.DataFrame) -> pd.Series:
    """
    フリーキャッシュフロー利回りを計算

    Args:
        financials (pd.DataFrame): 財務データ（symbol, date, fcf列を含む）
        market_cap_df (pd.DataFrame): 時価総額データ（symbol, date, market_cap列を含む）

    Returns:
        pd.Series: 各銘柄のFCF利回り
    """
    # 財務データと時価総額データをマージ
    merged_df = pd.merge(
        financials[['symbol', 'fcf']],
        market_cap_df[['symbol', 'market_cap']],
        on='symbol',
        how='inner'
    )
    
    # 時価総額が0より大きい場合のみ計算
    mask = (merged_df['market_cap'] > 0)
    fcf_yield = pd.Series(index=merged_df.index, dtype='float64')
    
    # FCF利回り計算: フリーキャッシュフロー / 時価総額
    fcf_yield[mask] = merged_df.loc[mask, 'fcf'] / merged_df.loc[mask, 'market_cap']
    
    return fcf_yield 