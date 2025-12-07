#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
バリュエーション指標計算モジュール
"""

import pandas as pd
import numpy as np


def calc_per(prices: pd.DataFrame, financials: pd.DataFrame) -> pd.Series:
    """
    株価収益率（PER）を計算する
    
    Args:
        prices (pd.DataFrame): 株価データ（symbol, date, close列を含む）
        financials (pd.DataFrame): 財務データ（symbol, date, eps列を含む）
        
    Returns:
        pd.Series: 各銘柄のPER
    """
    # 株価データと財務データをマージ
    merged_df = pd.merge(
        prices[['symbol', 'close']],
        financials[['symbol', 'eps']],
        on='symbol',
        how='inner'
    )
    
    # EPSが0より大きい銘柄のみ計算
    mask = (merged_df['eps'] > 0)
    per = pd.Series(index=merged_df.index, dtype='float64')
    
    # PER計算: 株価 / EPS
    per[mask] = merged_df.loc[mask, 'close'] / merged_df.loc[mask, 'eps']
    
    return per


def calc_pbr(prices: pd.DataFrame, financials: pd.DataFrame) -> pd.Series:
    """
    株価純資産倍率（PBR）を計算する
    
    Args:
        prices (pd.DataFrame): 株価データ（symbol, date, close列を含む）
        financials (pd.DataFrame): 財務データ（symbol, date列を含む）
                                  ※BPSはデータベースから別途取得
        
    Returns:
        pd.Series: 各銘柄のPBR
    """
    try:
        # BPSデータを取得するためのSQLクエリを実行
        # SQLAlchemyのエンジンを作成
        from sqlalchemy import create_engine
        from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
        
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 最新の貸借対照表からBPS（一株当たり純資産）を計算
        bps_query = """
        WITH latest_balance_sheet AS (
            SELECT 
                symbol, 
                MAX(date) as latest_date
            FROM fmp_data.balance_sheets
            WHERE period_type = 'annual'
            GROUP BY symbol
        ),
        latest_shares AS (
            SELECT 
                symbol, 
                MAX(date) as latest_date
            FROM fmp_data.shares
            GROUP BY symbol
        )
        SELECT 
            bs.symbol,
            (bs.total_stockholders_equity - COALESCE(bs.preferred_stock, 0)) / s.outstanding_shares as bps
        FROM fmp_data.balance_sheets bs
        JOIN latest_balance_sheet lbs ON bs.symbol = lbs.symbol AND bs.date = lbs.latest_date
        JOIN fmp_data.shares s ON bs.symbol = s.symbol
        JOIN latest_shares ls ON s.symbol = ls.symbol AND s.date = ls.latest_date
        WHERE bs.period_type = 'annual'
        """
        bps_df = pd.read_sql(bps_query, engine)
        
        # 株価データとBPSデータをマージ
        merged_df = pd.merge(
            prices[['symbol', 'close']],
            bps_df[['symbol', 'bps']],
            on='symbol',
            how='inner'
        )
        
        # BPSが0より大きい銘柄のみ計算
        mask = (merged_df['bps'] > 0)
        pbr = pd.Series(index=merged_df.index, dtype='float64')
        
        # PBR計算: 株価 / BPS
        pbr[mask] = merged_df.loc[mask, 'close'] / merged_df.loc[mask, 'bps']
        
        return pbr
        
    except Exception as e:
        import logging
        logging.error(f"PBR計算エラー: {e}")
        # エラー時は空のシリーズを返す
        return pd.Series(dtype='float64') 