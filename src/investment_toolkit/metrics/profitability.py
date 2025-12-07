#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
収益性指標計算モジュール
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
import logging

logger = logging.getLogger(__name__)


def calc_roe(financials: pd.DataFrame) -> pd.Series:
    """
    自己資本利益率（ROE）を計算する
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, date, net_income列を含む）
        
    Returns:
        pd.Series: 各銘柄のROE
    """
    try:
        # 純利益データをもとに自己資本データを取得
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 自己資本データを取得するクエリ
        equity_query = """
        WITH income_dates AS (
            SELECT 
                symbol, 
                date
            FROM fmp_data.income_statements
            WHERE period_type = 'annual'
        )
        SELECT 
            id.symbol,
            (bs.total_stockholders_equity - COALESCE(bs.preferred_stock, 0)) as book_value
        FROM income_dates id
        JOIN fmp_data.balance_sheets bs 
            ON id.symbol = bs.symbol 
            AND id.date = bs.date
            AND bs.period_type = 'annual'
        """
        
        equity_df = pd.read_sql(equity_query, engine)
        
        # 財務データと自己資本データをマージ
        merged_df = pd.merge(
            financials[['symbol', 'net_income']],
            equity_df[['symbol', 'book_value']],
            on='symbol',
            how='inner'
        )
        
        # 自己資本が0より大きい銘柄のみ計算
        mask = (merged_df['book_value'] > 0)
        roe = pd.Series(index=merged_df.index, dtype='float64')
        
        # ROE計算: 当期純利益 / 自己資本
        roe[mask] = merged_df.loc[mask, 'net_income'] / merged_df.loc[mask, 'book_value']
        
        return roe
        
    except Exception as e:
        logger.error(f"ROE計算エラー: {e}")
        # エラー時は空のシリーズを返す
        return pd.Series(dtype='float64')


def calc_roa(financials: pd.DataFrame) -> pd.Series:
    """
    総資産利益率（ROA）を計算する
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, date, net_income列を含む）
        
    Returns:
        pd.Series: 各銘柄のROA
    """
    try:
        # 純利益データをもとに総資産データを取得
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 総資産データを取得するクエリ
        assets_query = """
        WITH income_dates AS (
            SELECT 
                symbol, 
                date
            FROM fmp_data.income_statements
            WHERE period_type = 'annual'
        )
        SELECT 
            id.symbol,
            bs.total_assets
        FROM income_dates id
        JOIN fmp_data.balance_sheets bs 
            ON id.symbol = bs.symbol 
            AND id.date = bs.date
            AND bs.period_type = 'annual'
        """
        
        assets_df = pd.read_sql(assets_query, engine)
        
        # 財務データと総資産データをマージ
        merged_df = pd.merge(
            financials[['symbol', 'net_income']],
            assets_df[['symbol', 'total_assets']],
            on='symbol',
            how='inner'
        )
        
        # 総資産が0より大きい銘柄のみ計算
        mask = (merged_df['total_assets'] > 0)
        roa = pd.Series(index=merged_df.index, dtype='float64')
        
        # ROA計算: 当期純利益 / 総資産
        roa[mask] = merged_df.loc[mask, 'net_income'] / merged_df.loc[mask, 'total_assets']
        
        return roa
        
    except Exception as e:
        logger.error(f"ROA計算エラー: {e}")
        # エラー時は空のシリーズを返す
        return pd.Series(dtype='float64') 