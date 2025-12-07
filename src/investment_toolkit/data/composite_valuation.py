#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
複合バリュエーション指標計算モジュール

このモジュールは以下の複合バリュエーション指標を計算します：
- PEG比率（3年・5年）
- PEGY比率（3年・5年）
- GARP条件フラグ
- 企業価値（EV）
- EV/EBITDA、EV/FCF
- アーニングスイールド
- Altman Zスコア（将来実装）
- Piotroski Fスコア（将来実装）
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def get_composite_data(engine: Engine, start_date: str = '2010-01-01') -> pd.DataFrame:
    """
    複合バリュエーション指標計算に必要なデータを取得
    
    Args:
        engine: SQLAlchemyエンジン
        start_date: 開始日（YYYY-MM-DD形式）
        
    Returns:
        pd.DataFrame: 計算に必要な全データ
    """
    query = text("""
    WITH latest_ttm AS (
        -- 各銘柄の最新TTMデータ日付を取得
        SELECT 
            symbol,
            MAX(as_of_date) as latest_ttm_date
        FROM calculated_metrics.ttm_balance_sheets
        GROUP BY symbol
    )
    SELECT 
        bm.symbol,
        bm.as_of_date,
        bm.per,
        bm.market_cap,
        bm.dividend_yield,
        bm.eps_cagr_3y,
        bm.eps_cagr_5y,
        -- 損益計算書データ
        ti.ebitda,
        ti.operating_income,
        ti.net_income,
        ti.revenue,
        ti.interest_expense,
        ti.income_tax_expense,
        -- 貸借対照表データ
        tb.net_debt,
        tb.total_debt,
        tb.cash_and_cash_equivalents,
        tb.total_assets,
        tb.total_current_assets,
        tb.total_current_liabilities,
        tb.total_equity,
        tb.total_stockholders_equity,
        tb.retained_earnings,
        tb.net_receivables,
        tb.inventory,
        tb.property_plant_equipment_net,
        -- キャッシュフローデータ
        cf.free_cash_flow
    FROM calculated_metrics.basic_metrics bm
    LEFT JOIN latest_ttm lt ON bm.symbol = lt.symbol
    LEFT JOIN calculated_metrics.ttm_income_statements ti
           ON bm.symbol = ti.symbol AND lt.latest_ttm_date = ti.as_of_date
    LEFT JOIN calculated_metrics.ttm_balance_sheets tb
           ON bm.symbol = tb.symbol AND lt.latest_ttm_date = tb.as_of_date
    LEFT JOIN calculated_metrics.ttm_cash_flows cf
           ON bm.symbol = cf.symbol AND lt.latest_ttm_date = cf.as_of_date
    WHERE bm.as_of_date >= :start_date
    ORDER BY bm.symbol, bm.as_of_date
    """)
    
    logger.info(f"データ取得開始: {start_date}以降")
    df = pd.read_sql_query(query, engine, params={"start_date": start_date})
    logger.info(f"データ取得完了: {len(df):,}行")
    
    return df


def calc_peg_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    PEG比率（3年・5年）を計算
    
    Args:
        df: 入力データフレーム
        
    Returns:
        pd.DataFrame: PEG比率が追加されたデータフレーム
    """
    df = df.copy()
    
    # PEG 3年の計算
    # EPS CAGR 3年が正の値でPERが正の値の場合のみ計算
    mask_3y = (
        df['per'].notna() & 
        df['eps_cagr_3y'].notna() & 
        (df['per'] > 0) & 
        (df['eps_cagr_3y'] > 0)
    )
    df['peg_3y'] = np.where(
        mask_3y,
        df['per'] / (df['eps_cagr_3y'] * 100),  # EPS CAGRを%から小数に変換
        np.nan
    )
    
    # PEG 5年の計算
    mask_5y = (
        df['per'].notna() & 
        df['eps_cagr_5y'].notna() & 
        (df['per'] > 0) & 
        (df['eps_cagr_5y'] > 0)
    )
    df['peg_5y'] = np.where(
        mask_5y,
        df['per'] / (df['eps_cagr_5y'] * 100),  # EPS CAGRを%から小数に変換
        np.nan
    )
    
    logger.info(f"PEG 3年計算完了: {df['peg_3y'].notna().sum():,}件")
    logger.info(f"PEG 5年計算完了: {df['peg_5y'].notna().sum():,}件")
    
    return df


def calc_pegy_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    PEGY比率（3年・5年）を計算
    
    Args:
        df: 入力データフレーム（PEG比率計算済み）
        
    Returns:
        pd.DataFrame: PEGY比率が追加されたデータフレーム
    """
    df = df.copy()
    
    # PEGY 3年の計算
    mask_3y = (
        df['peg_3y'].notna() & 
        df['dividend_yield'].notna() & 
        (df['dividend_yield'] >= 0)
    )
    df['pegy_3y'] = np.where(
        mask_3y,
        df['peg_3y'] / (1 + df['dividend_yield']),
        np.nan
    )
    
    # PEGY 5年の計算
    mask_5y = (
        df['peg_5y'].notna() & 
        df['dividend_yield'].notna() & 
        (df['dividend_yield'] >= 0)
    )
    df['pegy_5y'] = np.where(
        mask_5y,
        df['peg_5y'] / (1 + df['dividend_yield']),
        np.nan
    )
    
    logger.info(f"PEGY 3年計算完了: {df['pegy_3y'].notna().sum():,}件")
    logger.info(f"PEGY 5年計算完了: {df['pegy_5y'].notna().sum():,}件")
    
    return df


def calc_garp_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    GARP条件フラグを計算
    
    Args:
        df: 入力データフレーム（PEG比率計算済み）
        
    Returns:
        pd.DataFrame: GARPフラグが追加されたデータフレーム
    """
    df = df.copy()
    
    # GARP 3年フラグ（PEG < 1.0）
    df['garp_flag_3y'] = np.where(
        df['peg_3y'].notna() & (df['peg_3y'] < 1.0),
        True,
        False
    )
    
    # GARP 5年フラグ（PEG < 1.0）
    df['garp_flag_5y'] = np.where(
        df['peg_5y'].notna() & (df['peg_5y'] < 1.0),
        True,
        False
    )
    
    garp_3y_count = df['garp_flag_3y'].sum() if df['garp_flag_3y'].notna().any() else 0
    garp_5y_count = df['garp_flag_5y'].sum() if df['garp_flag_5y'].notna().any() else 0
    
    logger.info(f"GARP 3年フラグ計算完了: {garp_3y_count:,}件がGARP条件を満たす")
    logger.info(f"GARP 5年フラグ計算完了: {garp_5y_count:,}件がGARP条件を満たす")
    
    return df


def calc_enterprise_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    企業価値（EV）を計算
    
    Args:
        df: 入力データフレーム
        
    Returns:
        pd.DataFrame: 企業価値が追加されたデータフレーム
    """
    df = df.copy()
    
    # net_debtが利用可能な場合
    mask_net_debt = df['net_debt'].notna() & df['market_cap'].notna()
    
    # net_debtが利用できない場合の代替計算
    mask_alt = (
        df['net_debt'].isna() & 
        df['total_debt'].notna() & 
        df['cash_and_cash_equivalents'].notna() & 
        df['market_cap'].notna()
    )
    
    # EVの計算
    df['ev'] = np.nan
    
    # net_debtが利用可能な場合
    df.loc[mask_net_debt, 'ev'] = (
        df.loc[mask_net_debt, 'market_cap'] + df.loc[mask_net_debt, 'net_debt']
    )
    
    # 代替計算
    df.loc[mask_alt, 'ev'] = (
        df.loc[mask_alt, 'market_cap'] + 
        df.loc[mask_alt, 'total_debt'] - 
        df.loc[mask_alt, 'cash_and_cash_equivalents']
    )
    
    logger.info(f"企業価値計算完了: {df['ev'].notna().sum():,}件")
    
    return df


def calc_ev_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    EV/EBITDA、EV/FCF比率を計算
    
    Args:
        df: 入力データフレーム（EV計算済み）
        
    Returns:
        pd.DataFrame: EV比率が追加されたデータフレーム
    """
    df = df.copy()
    
    # EV/EBITDA比率
    mask_ebitda = (
        df['ev'].notna() & 
        df['ebitda'].notna() & 
        (df['ebitda'] > 0)
    )
    df['ev_ebitda'] = np.where(
        mask_ebitda,
        df['ev'] / df['ebitda'],
        np.nan
    )
    
    # EV/FCF比率
    mask_fcf = (
        df['ev'].notna() & 
        df['free_cash_flow'].notna() & 
        (df['free_cash_flow'] > 0)
    )
    df['ev_fcf'] = np.where(
        mask_fcf,
        df['ev'] / df['free_cash_flow'],
        np.nan
    )
    
    logger.info(f"EV/EBITDA計算完了: {df['ev_ebitda'].notna().sum():,}件")
    logger.info(f"EV/FCF計算完了: {df['ev_fcf'].notna().sum():,}件")
    
    return df


def calc_earnings_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    アーニングスイールドを計算
    
    Args:
        df: 入力データフレーム（EV計算済み）
        
    Returns:
        pd.DataFrame: アーニングスイールドが追加されたデータフレーム
    """
    df = df.copy()
    
    mask = (
        df['ev'].notna() & 
        df['operating_income'].notna() & 
        (df['ev'] > 0)
    )
    
    df['earnings_yield'] = np.where(
        mask,
        df['operating_income'] / df['ev'],
        np.nan
    )
    
    logger.info(f"アーニングスイールド計算完了: {df['earnings_yield'].notna().sum():,}件")
    
    return df


def calc_altman_z_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Altman Zスコアを計算
    
    Altman Z-Score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    A = 運転資本 / 総資産
    B = 利益剰余金 / 総資産  
    C = 税引前利益 / 総資産
    D = 株式時価総額 / 総負債
    E = 売上高 / 総資産
    
    Args:
        df: 入力データフレーム
        
    Returns:
        pd.DataFrame: Altman Zスコアが追加されたデータフレーム
    """
    df = df.copy()
    
    # 必要なデータが揃っている行のマスク
    mask = (
        df['total_assets'].notna() & 
        df['total_current_assets'].notna() & 
        df['total_current_liabilities'].notna() &
        df['retained_earnings'].notna() &
        df['operating_income'].notna() &
        df['interest_expense'].notna() &
        df['market_cap'].notna() &
        df['total_debt'].notna() &
        df['revenue'].notna() &
        (df['total_assets'] > 0) &
        (df['total_debt'] > 0)
    )
    
    # 各比率の計算
    # A = 運転資本 / 総資産
    working_capital = df['total_current_assets'] - df['total_current_liabilities']
    ratio_a = working_capital / df['total_assets']
    
    # B = 利益剰余金 / 総資産
    ratio_b = df['retained_earnings'] / df['total_assets']
    
    # C = 税引前利益 / 総資産 (営業利益 + 受取利息 - 支払利息で近似)
    ebit = df['operating_income'] - df['interest_expense'].fillna(0)
    ratio_c = ebit / df['total_assets']
    
    # D = 株式時価総額 / 総負債
    ratio_d = df['market_cap'] / df['total_debt']
    
    # E = 売上高 / 総資産
    ratio_e = df['revenue'] / df['total_assets']
    
    # Altman Zスコアの計算
    df['altman_z'] = np.where(
        mask,
        1.2 * ratio_a + 1.4 * ratio_b + 3.3 * ratio_c + 0.6 * ratio_d + 1.0 * ratio_e,
        np.nan
    )
    
    altman_count = df['altman_z'].notna().sum()
    logger.info(f"Altman Zスコア計算完了: {altman_count:,}件")
    
    return df


def calc_piotroski_f_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Piotroski Fスコアを計算
    
    9つの財務指標に基づく0-9のスコア：
    収益性 (4項目):
    1. 当期純利益 > 0
    2. 営業キャッシュフロー > 0 (FCFで代用)
    3. ROA改善 (簡易版: 営業利益率 > 0)
    4. 営業CF > 純利益
    
    レバレッジ・流動性 (3項目):
    5. 負債比率改善 (簡易版: 負債比率 < 0.4)
    6. 流動比率改善 (流動比率 > 1.2)
    7. 発行済株式数減少 (簡易版: 配当利回り > 0)
    
    運営効率 (2項目):
    8. 売上総利益率改善 (簡易版: 売上総利益率 > 0.3)
    9. 資産回転率改善 (簡易版: 資産回転率 > 0.5)
    
    Args:
        df: 入力データフレーム
        
    Returns:
        pd.DataFrame: Piotroski Fスコアが追加されたデータフレーム
    """
    df = df.copy()
    
    # 基本的なデータが揃っている行のマスク
    basic_mask = (
        df['net_income'].notna() &
        df['free_cash_flow'].notna() &
        df['operating_income'].notna() &
        df['revenue'].notna() &
        df['total_assets'].notna() &
        df['total_debt'].notna() &
        df['total_current_assets'].notna() &
        df['total_current_liabilities'].notna() &
        (df['revenue'] > 0) &
        (df['total_assets'] > 0)
    )
    
    # スコア計算（各項目0または1）
    score = np.zeros(len(df))
    
    # 1. 当期純利益 > 0
    score += np.where(basic_mask & (df['net_income'] > 0), 1, 0)
    
    # 2. フリーキャッシュフロー > 0
    score += np.where(basic_mask & (df['free_cash_flow'] > 0), 1, 0)
    
    # 3. 営業利益率 > 0 (ROAの代用)
    operating_margin = df['operating_income'] / df['revenue']
    score += np.where(basic_mask & (operating_margin > 0), 1, 0)
    
    # 4. 営業CF > 純利益 (FCF > 純利益で代用)
    score += np.where(basic_mask & (df['free_cash_flow'] > df['net_income']), 1, 0)
    
    # 5. 負債比率 < 0.4 (健全性)
    debt_ratio = df['total_debt'] / df['total_assets']
    score += np.where(basic_mask & (debt_ratio < 0.4), 1, 0)
    
    # 6. 流動比率 > 1.2
    current_ratio = df['total_current_assets'] / df['total_current_liabilities']
    score += np.where(
        basic_mask & 
        df['total_current_liabilities'].notna() & 
        (df['total_current_liabilities'] > 0) & 
        (current_ratio > 1.2), 
        1, 0
    )
    
    # 7. 配当利回り > 0 (株主還元の代用)
    score += np.where(
        basic_mask & 
        df['dividend_yield'].notna() & 
        (df['dividend_yield'] > 0), 
        1, 0
    )
    
    # 8. 売上総利益率 > 30% (効率性)
    # 営業利益率で代用
    score += np.where(basic_mask & (operating_margin > 0.3), 1, 0)
    
    # 9. 資産回転率 > 0.5 (効率性)
    asset_turnover = df['revenue'] / df['total_assets']
    score += np.where(basic_mask & (asset_turnover > 0.5), 1, 0)
    
    # 最終スコアの設定
    df['piotroski_f'] = np.where(basic_mask, score, np.nan)
    
    piotroski_count = df['piotroski_f'].notna().sum()
    avg_score = df['piotroski_f'].mean() if piotroski_count > 0 else 0
    logger.info(f"Piotroski Fスコア計算完了: {piotroski_count:,}件 (平均スコア: {avg_score:.2f})")
    
    return df


def calc_composite_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    複合バリュエーション指標を一括計算
    
    Args:
        df: 入力データフレーム
        
    Returns:
        pd.DataFrame: 全ての複合指標が計算されたデータフレーム
    """
    logger.info("複合バリュエーション指標計算開始")
    
    # 空のデータフレームの場合は空の結果を返す
    if df.empty:
        result_columns = [
            'symbol', 'as_of_date', 'peg_3y', 'peg_5y', 'pegy_3y', 'pegy_5y',
            'garp_flag_3y', 'garp_flag_5y', 'ev', 'ev_ebitda', 'ev_fcf',
            'earnings_yield', 'altman_z', 'piotroski_f'
        ]
        return pd.DataFrame(columns=result_columns)
    
    # 各指標を順次計算
    df = calc_peg_ratios(df)
    df = calc_pegy_ratios(df)
    df = calc_garp_flags(df)
    df = calc_enterprise_value(df)
    df = calc_ev_ratios(df)
    df = calc_earnings_yield(df)
    df = calc_altman_z_score(df)
    df = calc_piotroski_f_score(df)
    
    # 結果データフレームの作成（必要な列のみ）
    result_columns = [
        'symbol', 'as_of_date', 'peg_3y', 'peg_5y', 'pegy_3y', 'pegy_5y',
        'garp_flag_3y', 'garp_flag_5y', 'ev', 'ev_ebitda', 'ev_fcf',
        'earnings_yield', 'altman_z', 'piotroski_f'
    ]
    
    result_df = df[result_columns].copy()
    
    logger.info(f"複合バリュエーション指標計算完了: {len(result_df):,}行")
    
    return result_df


def save_to_database(df: pd.DataFrame, engine: Engine, method: str = 'upsert') -> None:
    """
    計算結果をデータベースに保存
    
    Args:
        df: 保存するデータフレーム
        engine: SQLAlchemyエンジン
        method: 保存方法（'upsert' または 'replace'）
    """
    if df.empty:
        logger.warning("保存するデータがありません")
        return
    
    table_name = 'composite_valuation_metrics'
    schema = 'calculated_metrics'
    
    logger.info(f"データベース保存開始: {len(df):,}行")
    
    if method == 'upsert':
        # PostgreSQL用のUpsert処理
        from sqlalchemy.dialects.postgresql import insert
        from sqlalchemy import MetaData, Table
        
        # テーブルメタデータを取得
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine, schema=schema)
        
        # データをチャンクに分割して処理
        chunk_size = 500
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        with engine.begin() as conn:
            for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
                chunk_end = min(chunk_start + chunk_size, len(df))
                chunk_df = df.iloc[chunk_start:chunk_end]
                
                # 辞書形式に変換
                records = chunk_df.to_dict('records')
                
                # Upsert文の作成
                stmt = insert(table).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol', 'as_of_date'],
                    set_={col.name: stmt.excluded[col.name] for col in table.columns 
                          if col.name not in ['symbol', 'as_of_date']}
                )
                
                conn.execute(stmt)
                logger.info(f"チャンク {i+1}/{total_chunks} 保存完了")
    
    else:
        # 通常のto_sql使用
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists='append',
            index=False,
            chunksize=500
        )
    
    logger.info("データベース保存完了")


def process_composite_valuation_metrics(
    engine: Engine,
    start_date: str = '2010-01-01',
    save_to_db: bool = True
) -> pd.DataFrame:
    """
    複合バリュエーション指標の計算・保存を実行
    
    Args:
        engine: SQLAlchemyエンジン
        start_date: 開始日
        save_to_db: データベースに保存するかどうか
        
    Returns:
        pd.DataFrame: 計算結果
    """
    try:
        # データ取得
        df = get_composite_data(engine, start_date)
        
        if df.empty:
            logger.warning("計算対象データがありません")
            return pd.DataFrame()
        
        # 指標計算
        result_df = calc_composite_metrics(df)
        
        # データベース保存
        if save_to_db:
            save_to_database(result_df, engine)
        
        return result_df
        
    except Exception as e:
        logger.error(f"複合バリュエーション指標処理中にエラー: {e}", exc_info=True)
        raise 