#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
拡張収益性指標計算モジュール
basic_metricsテーブルに追加された新しいカラムの計算ロジックを実装
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def calc_operating_margin(financials: pd.DataFrame) -> pd.Series:
    """
    営業利益率（営業利益÷売上高）を計算
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, operating_income, revenue列を含む）
        
    Returns:
        pd.Series: 各銘柄の営業利益率
    """
    # 売上が0より大きい場合のみ計算
    mask = (financials['revenue'] > 0)
    margin = pd.Series(index=financials.index, dtype='float64')
    
    # 営業利益率計算：営業利益 / 売上高
    margin[mask] = financials.loc[mask, 'operating_income'] / financials.loc[mask, 'revenue']
    
    return margin


def calc_gross_margin(financials: pd.DataFrame) -> pd.Series:
    """
    粗利益率（売上総利益÷売上高）を計算
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, gross_profit, revenue列を含む）
        
    Returns:
        pd.Series: 各銘柄の粗利益率
    """
    # 売上が0より大きい場合のみ計算
    mask = (financials['revenue'] > 0)
    margin = pd.Series(index=financials.index, dtype='float64')
    
    # 粗利益率計算：売上総利益 / 売上高
    margin[mask] = financials.loc[mask, 'gross_profit'] / financials.loc[mask, 'revenue']
    
    return margin


def calc_roic(financials: pd.DataFrame, balance_sheets: pd.DataFrame) -> pd.Series:
    """
    投下資本利益率（ROIC）を計算
    ROIC = 税引後営業利益(NOPAT) / (株主資本 + 有利子負債)
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, net_income, tax_rate列を含む）
        balance_sheets (pd.DataFrame): 貸借対照表データ（symbol, total_stockholders_equity, total_debt列を含む）
        
    Returns:
        pd.Series: 各銘柄のROIC
    """
    # 財務データと貸借対照表をマージ
    merged_df = pd.merge(
        financials[['symbol', 'operating_income', 'income_tax_expense', 'income_before_tax']],
        balance_sheets[['symbol', 'total_stockholders_equity', 'total_debt']],
        on='symbol',
        how='inner'
    )
    
    # 実効税率の計算
    merged_df['tax_rate'] = np.where(
        merged_df['income_before_tax'] > 0,
        merged_df['income_tax_expense'] / merged_df['income_before_tax'],
        0.25  # デフォルトの税率（利益がない場合）
    )
    
    # 税引後営業利益の計算
    merged_df['nopat'] = merged_df['operating_income'] * (1 - merged_df['tax_rate'])
    
    # 投下資本の計算
    merged_df['invested_capital'] = merged_df['total_stockholders_equity'] + merged_df['total_debt']
    
    # 投下資本が0より大きい場合のみ計算
    mask = (merged_df['invested_capital'] > 0)
    roic = pd.Series(index=merged_df.index, dtype='float64')
    
    # ROIC計算: NOPAT / 投下資本
    roic[mask] = merged_df.loc[mask, 'nopat'] / merged_df.loc[mask, 'invested_capital']
    
    return roic


def calc_roce(financials: pd.DataFrame, balance_sheets: pd.DataFrame) -> pd.Series:
    """
    使用資本利益率（ROCE）を計算
    ROCE = 営業利益 / (総資産 - 流動負債)
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, operating_income列を含む）
        balance_sheets (pd.DataFrame): 貸借対照表データ（symbol, total_assets, total_current_liabilities列を含む）
        
    Returns:
        pd.Series: 各銘柄のROCE
    """
    # 財務データと貸借対照表をマージ
    merged_df = pd.merge(
        financials[['symbol', 'operating_income']],
        balance_sheets[['symbol', 'total_assets', 'total_current_liabilities']],
        on='symbol',
        how='inner'
    )
    
    # 使用資本の計算
    merged_df['capital_employed'] = merged_df['total_assets'] - merged_df['total_current_liabilities']
    
    # 使用資本が0より大きい場合のみ計算
    mask = (merged_df['capital_employed'] > 0)
    roce = pd.Series(index=merged_df.index, dtype='float64')
    
    # ROCE計算: 営業利益 / 使用資本
    roce[mask] = merged_df.loc[mask, 'operating_income'] / merged_df.loc[mask, 'capital_employed']
    
    return roce


def calc_debt_to_equity(balance_sheets: pd.DataFrame) -> pd.Series:
    """
    負債資本比率（総負債÷株主資本）を計算
    
    Args:
        balance_sheets (pd.DataFrame): 貸借対照表データ（symbol, total_debt, total_stockholders_equity列を含む）
        
    Returns:
        pd.Series: 各銘柄の負債資本比率
    """
    # 株主資本が0より大きい場合のみ計算
    mask = (balance_sheets['total_stockholders_equity'] > 0)
    ratio = pd.Series(index=balance_sheets.index, dtype='float64')
    
    # 負債資本比率計算: 総負債 / 株主資本
    ratio[mask] = balance_sheets.loc[mask, 'total_debt'] / balance_sheets.loc[mask, 'total_stockholders_equity']
    
    return ratio


def calc_net_debt_to_ebitda(balance_sheets: pd.DataFrame, financials: pd.DataFrame) -> pd.Series:
    """
    純負債EBITDA倍率を計算
    
    Args:
        balance_sheets (pd.DataFrame): 貸借対照表データ（symbol, total_debt, cash_and_cash_equivalents列を含む）
        financials (pd.DataFrame): 財務データ（symbol, operating_income, depreciation_and_amortization列を含む）
        
    Returns:
        pd.Series: 各銘柄の純負債EBITDA倍率
    """
    # 財務データと貸借対照表をマージ
    merged_df = pd.merge(
        balance_sheets[['symbol', 'total_debt', 'cash_and_cash_equivalents']],
        financials[['symbol', 'operating_income', 'depreciation_and_amortization']],
        on='symbol',
        how='inner'
    )
    
    # 純負債の計算
    merged_df['net_debt'] = merged_df['total_debt'] - merged_df['cash_and_cash_equivalents']
    
    # EBITDAの計算
    merged_df['ebitda'] = merged_df['operating_income'] + merged_df['depreciation_and_amortization']
    
    # EBITDAが0より大きい場合のみ計算
    mask = (merged_df['ebitda'] > 0)
    ratio = pd.Series(index=merged_df.index, dtype='float64')
    
    # 純負債EBITDA倍率計算: 純負債 / EBITDA
    ratio[mask] = merged_df.loc[mask, 'net_debt'] / merged_df.loc[mask, 'ebitda']
    
    return ratio


def calc_interest_coverage(financials: pd.DataFrame) -> pd.Series:
    """
    インタレストカバレッジレシオ（営業利益÷支払利息）を計算
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, operating_income, interest_expense列を含む）
        
    Returns:
        pd.Series: 各銘柄のインタレストカバレッジレシオ
    """
    # 支払利息が0でない場合のみ計算
    mask = (financials['interest_expense'] != 0)
    ratio = pd.Series(index=financials.index, dtype='float64')
    
    # インタレストカバレッジレシオ計算: 営業利益 / 支払利息（絶対値を取る）
    ratio[mask] = financials.loc[mask, 'operating_income'] / abs(financials.loc[mask, 'interest_expense'])
    
    return ratio


def calc_cagr(initial_value: float, final_value: float, years: int) -> float:
    """
    年平均成長率（CAGR）を計算
    
    Args:
        initial_value (float): 初期値
        final_value (float): 最終値
        years (int): 年数
        
    Returns:
        float: CAGR値
    """
    if initial_value <= 0 or years <= 0:
        return np.nan
    
    return (final_value / initial_value) ** (1 / years) - 1


def calc_eps_cagr(financials_historical: Dict[str, pd.DataFrame], years: int = 3) -> pd.Series:
    """
    EPS年平均成長率を計算
    
    Args:
        financials_historical (Dict[str, pd.DataFrame]): 各銘柄の過去財務データ（symbol, date, eps列を含む）
        years (int): 計算する年数（デフォルト3年）
        
    Returns:
        pd.Series: 各銘柄のEPS CAGR
    """
    results = {}
    
    for symbol, df in financials_historical.items():
        if len(df) < years + 1:
            # 十分なデータがない場合はスキップ
            results[symbol] = np.nan
            continue
        
        # 日付でソート
        df = df.sort_values('date')
        
        # 最新のEPSと指定年前のEPSを取得
        latest_eps = df.iloc[-1]['eps']
        past_eps = df.iloc[-(years+1)]['eps']
        
        # CAGRを計算
        if past_eps > 0:
            results[symbol] = calc_cagr(past_eps, latest_eps, years)
        else:
            results[symbol] = np.nan
    
    return pd.Series(results)


def calc_revenue_cagr(financials_historical: Dict[str, pd.DataFrame], years: int = 3) -> pd.Series:
    """
    売上高年平均成長率を計算
    
    Args:
        financials_historical (Dict[str, pd.DataFrame]): 各銘柄の過去財務データ（symbol, date, revenue列を含む）
        years (int): 計算する年数（デフォルト3年）
        
    Returns:
        pd.Series: 各銘柄の売上高 CAGR
    """
    results = {}
    
    for symbol, df in financials_historical.items():
        if len(df) < years + 1:
            # 十分なデータがない場合はスキップ
            results[symbol] = np.nan
            continue
        
        # 日付でソート
        df = df.sort_values('date')
        
        # 最新の売上高と指定年前の売上高を取得
        latest_revenue = df.iloc[-1]['revenue']
        past_revenue = df.iloc[-(years+1)]['revenue']
        
        # CAGRを計算
        if past_revenue > 0:
            results[symbol] = calc_cagr(past_revenue, latest_revenue, years)
        else:
            results[symbol] = np.nan
    
    return pd.Series(results)


def calc_fcf_margin(financials: pd.DataFrame) -> pd.Series:
    """
    フリーキャッシュフローマージン（FCF÷売上高）を計算
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, fcf, revenue列を含む）
        
    Returns:
        pd.Series: 各銘柄のFCFマージン
    """
    # 売上が0より大きい場合のみ計算
    mask = (financials['revenue'] > 0)
    margin = pd.Series(index=financials.index, dtype='float64')
    
    # FCFマージン計算: FCF / 売上高
    margin[mask] = financials.loc[mask, 'fcf'] / financials.loc[mask, 'revenue']
    
    return margin


def calc_cfo_to_net_income(financials: pd.DataFrame, cash_flows: pd.DataFrame) -> pd.Series:
    """
    営業キャッシュフロー対純利益比率を計算
    
    Args:
        financials (pd.DataFrame): 財務データ（symbol, net_income列を含む）
        cash_flows (pd.DataFrame): キャッシュフロー計算書データ（symbol, operating_cash_flow列を含む）
        
    Returns:
        pd.Series: 各銘柄の営業CF対純利益比率
    """
    # 財務データとキャッシュフローデータをマージ
    merged_df = pd.merge(
        financials[['symbol', 'net_income']],
        cash_flows[['symbol', 'operating_cash_flow']],
        on='symbol',
        how='inner'
    )
    
    # 純利益が0より大きい場合のみ計算
    mask = (merged_df['net_income'] > 0)
    ratio = pd.Series(index=merged_df.index, dtype='float64')
    
    # 営業CF対純利益比率計算: 営業CF / 純利益
    ratio[mask] = merged_df.loc[mask, 'operating_cash_flow'] / merged_df.loc[mask, 'net_income']
    
    return ratio


def get_historical_financials(db_connection, symbol: str, years: int = 5) -> pd.DataFrame:
    """
    指定された銘柄の過去財務データを取得
    
    Args:
        db_connection: データベース接続オブジェクト
        symbol (str): 銘柄コード
        years (int): 取得する過去年数
        
    Returns:
        pd.DataFrame: 過去財務データ
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    query = f"""
    SELECT 
        symbol, 
        date, 
        revenue, 
        eps, 
        net_income
    FROM 
        fmp_data.income_statements
    WHERE 
        symbol = '{symbol}'
        AND period_type = 'annual'
        AND date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'
    ORDER BY 
        date DESC
    """
    
    return pd.read_sql(query, db_connection) 