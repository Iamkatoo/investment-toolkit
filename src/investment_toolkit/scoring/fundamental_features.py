#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fundamental Trend Features Calculator - Growth Pillar特徴量計算モジュール

This module implements Task 2: Growth trend features calculation for the new 5-pillar scoring system.
Calculates TTM YoY growth rates, growth acceleration, and consistency indicators.

Key Features:
- TTM YoY growth rates for revenue, EPS, and EBIT
- Growth acceleration (4-quarter slope analysis)
- Growth consistency (8-quarter positive growth counting)
- Robust handling of edge cases (negative EPS, IPOs, fiscal year changes)
- Data quality validation and outlier handling

Usage:
    calc = FundamentalTrendCalculator()
    features = calc.calculate_all_features(target_date="2025-09-11", symbols=["AAPL", "MSFT"])
    
References:
    - Specification: docs/calc_fundamental_trend_features_spec.md
    - Schema: config/score_schema.yaml (Growth pillar)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# プロジェクト内のモジュールをインポート
try:
    from investment_analysis.database.db_manager import DatabaseManager, get_db_connection
    from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")

logger = logging.getLogger(__name__)


class FundamentalTrendCalculationError(Exception):
    """Custom exception for fundamental trend calculation errors"""
    pass


class FundamentalTrendCalculator:
    """
    Growth fundamental features calculator for the new 5-pillar scoring system.
    
    This class calculates growth-related features including:
    - TTM YoY growth rates (revenue, EPS, EBIT)
    - Growth acceleration indicators (4Q slope)
    - Growth consistency metrics (8Q positive growth count)
    
    Attributes:
        engine (Engine): SQLAlchemy database engine
        data_quality_threshold (float): Minimum data quality threshold
        outlier_caps (Dict): Outlier trimming configuration
    """
    
    def __init__(self, engine: Optional[Engine] = None):
        """
        Initialize the fundamental trend calculator.
        
        Args:
            engine (Optional[Engine]): Database engine. If None, creates default connection.
        """
        if engine is None:
            try:
                connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                self.engine = create_engine(connection_string)
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise FundamentalTrendCalculationError(f"Database connection failed: {e}")
        else:
            self.engine = engine
        
        # Configuration for data quality and outlier handling
        self.data_quality_threshold = 0.5  # Minimum data completeness ratio
        self.outlier_caps = {
            'growth_rate_upper': 5.0,     # +500% cap
            'growth_rate_lower': -0.9,    # -90% cap  
            'slope_cap': 100.0            # ±100 percentage points per quarter
        }
        
        logger.info("FundamentalTrendCalculator initialized successfully")
    
    def calculate_all_features(
        self, 
        target_date: str, 
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate all growth trend features for given symbols and date.
        
        Args:
            target_date (str): Target date in 'YYYY-MM-DD' format
            symbols (Optional[List[str]]): List of symbols. If None, calculates for all available symbols.
            
        Returns:
            pd.DataFrame: DataFrame with calculated features
            
        Columns:
            - symbol: Stock symbol
            - as_of_date: Calculation date
            - ttm_rev_yoy: TTM revenue YoY growth rate
            - ttm_eps_yoy: TTM EPS YoY growth rate  
            - ttm_ebit_yoy: TTM EBIT YoY growth rate
            - eps_yoy_slope_4q: 4Q EPS YoY slope
            - rev_yoy_slope_4q: 4Q revenue YoY slope
            - growth_consistency_8q: 8Q growth consistency score
            - cagr_3y_eps: 3Y EPS CAGR (reference)
            - cagr_5y_eps: 5Y EPS CAGR (reference)
            - cagr_3y_rev: 3Y revenue CAGR (reference)
            - cagr_5y_rev: 5Y revenue CAGR (reference)
            - data_quality_flags: Data quality indicators
            - last_updated: Calculation timestamp
        """
        try:
            # Get base financial data
            financial_data = self._get_financial_data(target_date, symbols)
            
            if financial_data.empty:
                logger.warning(f"No financial data found for date {target_date}")
                return pd.DataFrame()
            
            # Calculate TTM YoY features
            ttm_features = self.calculate_ttm_growth_rates(financial_data)
            
            # Calculate acceleration features
            acceleration_features = self.calculate_growth_acceleration(financial_data)
            
            # Calculate consistency features
            consistency_features = self.calculate_growth_consistency(financial_data)
            
            # Get CAGR reference data
            cagr_features = self._get_cagr_reference_data(target_date, symbols)
            
            # Merge all features
            result = self._merge_features(
                ttm_features, 
                acceleration_features, 
                consistency_features,
                cagr_features,
                target_date
            )
            
            # Add data quality assessment
            result = self._add_data_quality_flags(result)
            
            logger.info(f"Calculated fundamental trend features for {len(result)} symbols")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating fundamental trend features: {e}")
            raise FundamentalTrendCalculationError(f"Feature calculation failed: {e}")
    
    def calculate_ttm_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate TTM (Trailing Twelve Months) YoY growth rates.

        Args:
            df (pd.DataFrame): Financial data with quarterly income statement data

        Returns:
            pd.DataFrame: DataFrame with TTM YoY growth rates

        Features calculated:
            - ttm_rev_yoy: TTM revenue YoY growth rate
            - ttm_eps_yoy: TTM EPS YoY growth rate
            - ttm_ebit_yoy: TTM EBIT YoY growth rate (calculated from EBITDA)
        """
        logger.debug("Calculating TTM YoY growth rates")

        try:
            # Get TTM EBITDA YoY data from calculated_metrics.ttm_income_statement
            ttm_ebitda_data = self._get_ttm_ebitda_yoy_data(df['symbol'].unique().tolist())

            result_list = []

            # Group by symbol for individual calculations
            for symbol, symbol_data in df.groupby('symbol'):
                symbol_data = symbol_data.sort_values('period_ending').copy()

                # Calculate TTM values for current and 1Y ago
                ttm_current = self._calculate_ttm_values(symbol_data.head(4))  # Latest 4Q
                ttm_1y_ago = self._calculate_ttm_values(symbol_data.iloc[4:8] if len(symbol_data) >= 8 else pd.DataFrame())

                # Get TTM EBITDA YoY from pre-calculated data
                symbol_ebitda_data = ttm_ebitda_data[ttm_ebitda_data['symbol'] == symbol]
                ttm_ebit_yoy = symbol_ebitda_data['ebitda_yoy'].iloc[0] if len(symbol_ebitda_data) > 0 else None

                if ttm_current is None or ttm_1y_ago is None:
                    # Insufficient data
                    result_list.append({
                        'symbol': symbol,
                        'ttm_rev_yoy': None,
                        'ttm_eps_yoy': None,
                        'ttm_ebit_yoy': ttm_ebit_yoy  # Use pre-calculated EBITDA YoY as EBIT YoY
                    })
                    continue

                # Calculate YoY growth rates with safe division
                ttm_features = {
                    'symbol': symbol,
                    'ttm_rev_yoy': self._safe_yoy_calculation(
                        ttm_current.get('revenue', 0),
                        ttm_1y_ago.get('revenue', 0)
                    ),
                    'ttm_eps_yoy': self._safe_yoy_calculation(
                        ttm_current.get('eps', 0),
                        ttm_1y_ago.get('eps', 0),
                        handle_negative=True
                    ),
                    'ttm_ebit_yoy': ttm_ebit_yoy  # Use pre-calculated EBITDA YoY as EBIT YoY
                }
                
                # Apply outlier caps
                ttm_features = self._apply_growth_rate_caps(ttm_features)
                result_list.append(ttm_features)
            
            result_df = pd.DataFrame(result_list)
            logger.debug(f"Calculated TTM YoY growth rates for {len(result_df)} symbols")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating TTM growth rates: {e}")
            raise FundamentalTrendCalculationError(f"TTM calculation failed: {e}")
    
    def calculate_growth_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth acceleration indicators using linear regression slopes.
        
        Args:
            df (pd.DataFrame): Financial data with quarterly data
            
        Returns:
            pd.DataFrame: DataFrame with growth acceleration metrics
            
        Features calculated:
            - eps_yoy_slope_4q: Linear regression slope of EPS YoY over 4 quarters
            - rev_yoy_slope_4q: Linear regression slope of revenue YoY over 4 quarters
        """
        logger.debug("Calculating growth acceleration indicators")
        
        try:
            result_list = []
            
            for symbol, symbol_data in df.groupby('symbol'):
                symbol_data = symbol_data.sort_values('period_ending').copy()
                
                if len(symbol_data) < 8:  # Need at least 8 quarters for YoY + slope
                    result_list.append({
                        'symbol': symbol,
                        'eps_yoy_slope_4q': None,
                        'rev_yoy_slope_4q': None
                    })
                    continue
                
                # Calculate quarterly YoY growth rates for latest 4 quarters
                quarterly_yoy = []
                for i in range(4):
                    current_q = symbol_data.iloc[i]
                    prior_year_q = symbol_data.iloc[i + 4] if i + 4 < len(symbol_data) else None
                    
                    if prior_year_q is not None:
                        eps_yoy = self._safe_yoy_calculation(
                            current_q.get('eps', 0), 
                            prior_year_q.get('eps', 0),
                            handle_negative=True
                        )
                        rev_yoy = self._safe_yoy_calculation(
                            current_q.get('revenue', 0), 
                            prior_year_q.get('revenue', 0)
                        )
                        quarterly_yoy.append({
                            'eps_yoy': eps_yoy,
                            'rev_yoy': rev_yoy,
                            'quarter_index': i
                        })
                
                if len(quarterly_yoy) < 4:
                    result_list.append({
                        'symbol': symbol,
                        'eps_yoy_slope_4q': None,
                        'rev_yoy_slope_4q': None
                    })
                    continue
                
                # Calculate linear regression slopes
                quarterly_df = pd.DataFrame(quarterly_yoy)
                
                eps_slope = self._calculate_linear_slope(
                    quarterly_df['quarter_index'].values,
                    quarterly_df['eps_yoy'].dropna().values
                ) if quarterly_df['eps_yoy'].notna().sum() >= 3 else None
                
                rev_slope = self._calculate_linear_slope(
                    quarterly_df['quarter_index'].values,
                    quarterly_df['rev_yoy'].dropna().values
                ) if quarterly_df['rev_yoy'].notna().sum() >= 3 else None
                
                # Apply slope caps
                if eps_slope is not None:
                    eps_slope = np.clip(eps_slope, -self.outlier_caps['slope_cap'], self.outlier_caps['slope_cap'])
                if rev_slope is not None:
                    rev_slope = np.clip(rev_slope, -self.outlier_caps['slope_cap'], self.outlier_caps['slope_cap'])
                
                result_list.append({
                    'symbol': symbol,
                    'eps_yoy_slope_4q': eps_slope,
                    'rev_yoy_slope_4q': rev_slope
                })
            
            result_df = pd.DataFrame(result_list)
            logger.debug(f"Calculated growth acceleration for {len(result_df)} symbols")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating growth acceleration: {e}")
            raise FundamentalTrendCalculationError(f"Acceleration calculation failed: {e}")
    
    def calculate_growth_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth consistency indicator over 8 quarters.
        
        Args:
            df (pd.DataFrame): Financial data with quarterly data
            
        Returns:
            pd.DataFrame: DataFrame with growth consistency scores
            
        Features calculated:
            - growth_consistency_8q: Count of quarters with positive YoY growth (0-8 scale)
        """
        logger.debug("Calculating growth consistency indicators")
        
        try:
            result_list = []
            
            for symbol, symbol_data in df.groupby('symbol'):
                symbol_data = symbol_data.sort_values('period_ending').copy()
                
                if len(symbol_data) < 16:  # Need 16 quarters for 8Q YoY comparisons
                    result_list.append({
                        'symbol': symbol,
                        'growth_consistency_8q': None
                    })
                    continue
                
                # Calculate YoY growth for 8 quarters and count positive growth
                positive_growth_count = 0
                total_valid_quarters = 0
                
                for i in range(8):
                    current_q = symbol_data.iloc[i]
                    prior_year_q = symbol_data.iloc[i + 8] if i + 8 < len(symbol_data) else None
                    
                    if prior_year_q is not None:
                        eps_yoy = self._safe_yoy_calculation(
                            current_q.get('eps', 0), 
                            prior_year_q.get('eps', 0),
                            handle_negative=True
                        )
                        rev_yoy = self._safe_yoy_calculation(
                            current_q.get('revenue', 0), 
                            prior_year_q.get('revenue', 0)
                        )
                        
                        # Count positive growth (both revenue and EPS)
                        if eps_yoy is not None and eps_yoy > 0:
                            positive_growth_count += 0.5
                        if rev_yoy is not None and rev_yoy > 0:
                            positive_growth_count += 0.5
                            
                        total_valid_quarters += 1
                
                # Calculate consistency score (0-8 scale)
                if total_valid_quarters > 0:
                    consistency_score = (positive_growth_count / total_valid_quarters) * 8
                else:
                    consistency_score = None
                
                result_list.append({
                    'symbol': symbol,
                    'growth_consistency_8q': consistency_score
                })
            
            result_df = pd.DataFrame(result_list)
            logger.debug(f"Calculated growth consistency for {len(result_df)} symbols")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating growth consistency: {e}")
            raise FundamentalTrendCalculationError(f"Consistency calculation failed: {e}")
    
    def _get_financial_data(self, target_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get financial data from database for calculations.
        
        Args:
            target_date (str): Target date for data retrieval
            symbols (Optional[List[str]]): Specific symbols to fetch
            
        Returns:
            pd.DataFrame: Financial data sorted by symbol and date
        """
        try:
            # パラメータ化クエリを使用
            if symbols and len(symbols) > 1000:
                logger.warning(f"Large symbol list ({len(symbols)} symbols), processing in batches")
                batch_size = 500
                all_dfs: List[pd.DataFrame] = []

                for i in range(0, len(symbols), batch_size):
                    batch_symbols = symbols[i:i + batch_size]
                    batch_df = self._get_financial_data(target_date, batch_symbols)
                    if not batch_df.empty:
                        all_dfs.append(batch_df)

                return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

            # 通常処理（1000銘柄以下）
            symbol_filter_clause = "      AND symbol = ANY(:symbols)" if symbols else ""

            query_lines = [
                "WITH quarterly_data AS (",
                "    SELECT",
                "        symbol,",
                "        date as period_ending,",
                "        filing_date,",
                "        revenue,",
                "        eps,",
                "        net_income,",
                "        ebitda,",
                "        period_type,",
                "        calendar_year,",
                "        period",
                "    FROM fmp_data.income_statements",
                "    WHERE date <= CAST(:target_date AS DATE)",
                "      AND period_type = 'quarterly'",
            ]

            if symbol_filter_clause:
                query_lines.append(symbol_filter_clause)

            query_lines.extend([
                "    ORDER BY symbol, date DESC",
                "    LIMIT 20000",
                "),",
                "enriched_data AS (",
                "    SELECT",
                "        qd.*,",
                "        bm.eps as calculated_eps,",
                "        bm.market_cap",
                "    FROM quarterly_data qd",
                "    LEFT JOIN calculated_metrics.basic_metrics bm ON qd.symbol = bm.symbol",
                "        AND bm.as_of_date = (",
                "            SELECT MAX(as_of_date)",
                "            FROM calculated_metrics.basic_metrics",
                "            WHERE symbol = qd.symbol AND as_of_date <= qd.period_ending",
                "        )",
                ")",
                "SELECT *,",
                "       COALESCE(eps, calculated_eps) as final_eps",
                "FROM enriched_data",
                "ORDER BY symbol, period_ending DESC",
            ])

            query = text("\n".join(query_lines))

            params: Dict[str, Any] = {'target_date': target_date}
            if symbols:
                params['symbols'] = symbols

            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            
            if df.empty:
                logger.warning(f"No financial data found for target date {target_date}")
                return pd.DataFrame()
            
            # Add derived fields
            df['ebit'] = df['ebitda']  # Simplified - could enhance with depreciation
            
            logger.debug(f"Retrieved financial data for {df['symbol'].nunique()} symbols, {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching financial data: {e}")
            raise FundamentalTrendCalculationError(f"Data retrieval failed: {e}")
    
    def _get_ttm_ebitda_yoy_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get TTM EBITDA YoY data from calculated_metrics.ttm_income_statements.

        Args:
            symbols (List[str]): List of symbols to get data for

        Returns:
            pd.DataFrame: TTM EBITDA YoY data
        """
        try:
            # Create parameterized query for TTM EBITDA YoY calculation
            symbol_placeholders = ','.join([f"'{symbol}'" for symbol in symbols])

            query = f"""
            WITH ttm_current AS (
                SELECT
                    symbol,
                    ebitda as current_ebitda,
                    report_date
                FROM calculated_metrics.ttm_income_statements
                WHERE symbol IN ({symbol_placeholders})
                AND report_date = (
                    SELECT MAX(report_date)
                    FROM calculated_metrics.ttm_income_statements t2
                    WHERE t2.symbol = ttm_income_statements.symbol
                )
            ),
            ttm_year_ago AS (
                SELECT
                    symbol,
                    ebitda as year_ago_ebitda
                FROM calculated_metrics.ttm_income_statements
                WHERE symbol IN ({symbol_placeholders})
                AND report_date = (
                    SELECT MAX(report_date)
                    FROM calculated_metrics.ttm_income_statements t2
                    WHERE t2.symbol = ttm_income_statements.symbol
                    AND t2.report_date <= (
                        SELECT report_date - INTERVAL '1 year'
                        FROM ttm_current tc
                        WHERE tc.symbol = ttm_income_statements.symbol
                    )
                )
            )
            SELECT
                c.symbol,
                CASE
                    WHEN y.year_ago_ebitda IS NULL OR y.year_ago_ebitda = 0 THEN NULL
                    ELSE (c.current_ebitda - y.year_ago_ebitda) / ABS(y.year_ago_ebitda)
                END as ebitda_yoy
            FROM ttm_current c
            LEFT JOIN ttm_year_ago y ON c.symbol = y.symbol
            """

            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)

            logger.debug(f"Retrieved TTM EBITDA YoY data for {len(df)} symbols")
            return df

        except Exception as e:
            logger.warning(f"Error fetching TTM EBITDA YoY data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['symbol', 'ebitda_yoy'])

    def _get_cagr_reference_data(self, target_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get existing CAGR calculations for reference from basic_metrics table.
        
        Args:
            target_date (str): Target date
            symbols (Optional[List[str]]): Specific symbols
            
        Returns:
            pd.DataFrame: CAGR reference data
        """
        try:
            # パラメータ化クエリを使用
            # Get the most recent CAGR data for each symbol up to target_date
            query = """
            WITH ranked_cagr AS (
                SELECT
                    symbol,
                    as_of_date,
                    eps_cagr_3y as cagr_3y_eps,
                    eps_cagr_5y as cagr_5y_eps,
                    revenue_cagr_3y as cagr_3y_rev,
                    revenue_cagr_5y as cagr_5y_rev,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY as_of_date DESC) as rn
                FROM calculated_metrics.basic_metrics
                WHERE as_of_date <= '{target_date}'::date
                {symbol_filter}
            )
            SELECT
                symbol,
                cagr_3y_eps,
                cagr_5y_eps,
                cagr_3y_rev,
                cagr_5y_rev
            FROM ranked_cagr
            WHERE rn = 1
            """

            # Use same symbol filter as above
            if symbols:
                symbol_placeholders = ','.join([f"'{symbol}'" for symbol in symbols])
                symbol_filter = f"AND symbol IN ({symbol_placeholders})"
            else:
                symbol_filter = ""

            # Apply symbol filter to query
            query = query.format(target_date=target_date, symbol_filter=symbol_filter)

            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            logger.debug(f"Retrieved CAGR reference data for {len(df)} symbols")
            return df
            
        except Exception as e:
            logger.warning(f"Error fetching CAGR reference data: {e}")
            # Return empty DataFrame rather than failing completely
            columns = ['symbol', 'cagr_3y_eps', 'cagr_5y_eps', 'cagr_3y_rev', 'cagr_5y_rev']
            return pd.DataFrame(columns=columns)
    
    def _calculate_ttm_values(self, quarterly_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Calculate TTM values from quarterly data.
        
        Args:
            quarterly_data (pd.DataFrame): 4 quarters of data
            
        Returns:
            Optional[Dict[str, float]]: TTM values or None if insufficient data
        """
        if len(quarterly_data) < 4:
            return None
        
        try:
            ttm_revenue = quarterly_data['revenue'].sum()
            ttm_net_income = quarterly_data['net_income'].sum()
            
            # Calculate weighted average EPS (simplified)
            ttm_eps = quarterly_data['eps'].sum() if quarterly_data['eps'].notna().any() else 0
            ttm_ebit = quarterly_data['ebit'].sum() if quarterly_data['ebit'].notna().any() else 0
            
            return {
                'revenue': ttm_revenue,
                'eps': ttm_eps,
                'ebit': ttm_ebit,
                'net_income': ttm_net_income
            }
            
        except Exception as e:
            logger.error(f"Error calculating TTM values: {e}")
            return None
    
    def _safe_yoy_calculation(
        self, 
        current: float, 
        prior: float, 
        handle_negative: bool = False
    ) -> Optional[float]:
        """
        Safely calculate YoY growth rate with edge case handling.
        
        Args:
            current (float): Current period value
            prior (float): Prior year same period value
            handle_negative (bool): Whether to handle negative values specially
            
        Returns:
            Optional[float]: YoY growth rate or None if invalid
        """
        try:
            # Handle None/NaN values
            if pd.isna(current) or pd.isna(prior):
                return None
            
            # Handle zero division
            if abs(prior) < 0.01:  # Effectively zero
                return None
            
            # Handle negative values (especially for EPS)
            if handle_negative:
                if prior < 0 and current < 0:
                    # Both negative - comparison meaningless
                    return None
                elif prior < 0 < current:
                    # Loss to profit - calculate as improvement
                    return (current - prior) / abs(prior)
                elif prior > 0 > current:
                    # Profit to loss - calculate as deterioration
                    return (current - prior) / prior
            
            # Standard YoY calculation
            yoy_growth = (current - prior) / abs(prior)
            
            return yoy_growth
            
        except Exception as e:
            logger.debug(f"Error in YoY calculation: {e}")
            return None
    
    def _calculate_linear_slope(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """
        Calculate linear regression slope.
        
        Args:
            x (np.ndarray): X values (quarter indices)
            y (np.ndarray): Y values (growth rates)
            
        Returns:
            Optional[float]: Linear slope or None if calculation fails
        """
        try:
            if len(x) < 2 or len(y) < 2:
                return None
            
            # Remove any NaN values
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 2:
                return None
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            # Calculate linear regression slope using numpy
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            
            return float(slope)
            
        except Exception as e:
            logger.debug(f"Error calculating linear slope: {e}")
            return None
    
    def _apply_growth_rate_caps(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply outlier caps to growth rate features.
        
        Args:
            features_dict (Dict[str, Any]): Features dictionary
            
        Returns:
            Dict[str, Any]: Features with caps applied
        """
        capped_features = features_dict.copy()
        
        growth_rate_fields = ['ttm_rev_yoy', 'ttm_eps_yoy', 'ttm_ebit_yoy']
        
        for field in growth_rate_fields:
            if field in capped_features and capped_features[field] is not None:
                value = capped_features[field]
                capped_value = np.clip(
                    value, 
                    self.outlier_caps['growth_rate_lower'], 
                    self.outlier_caps['growth_rate_upper']
                )
                capped_features[field] = capped_value
        
        return capped_features
    
    def _merge_features(
        self, 
        ttm_df: pd.DataFrame, 
        acceleration_df: pd.DataFrame, 
        consistency_df: pd.DataFrame,
        cagr_df: pd.DataFrame,
        target_date: str
    ) -> pd.DataFrame:
        """
        Merge all calculated features into final DataFrame.
        
        Args:
            ttm_df (pd.DataFrame): TTM features
            acceleration_df (pd.DataFrame): Acceleration features
            consistency_df (pd.DataFrame): Consistency features
            cagr_df (pd.DataFrame): CAGR reference features
            target_date (str): Calculation target date
            
        Returns:
            pd.DataFrame: Merged features DataFrame
        """
        try:
            # Start with TTM features
            result = ttm_df.copy()
            
            # Merge acceleration features
            if not acceleration_df.empty:
                result = result.merge(acceleration_df, on='symbol', how='left')
            
            # Merge consistency features
            if not consistency_df.empty:
                result = result.merge(consistency_df, on='symbol', how='left')
            
            # Merge CAGR features
            if not cagr_df.empty:
                result = result.merge(cagr_df, on='symbol', how='left')
            
            # Add metadata
            result['as_of_date'] = pd.to_datetime(target_date).date()
            result['last_updated'] = datetime.now()
            
            logger.debug(f"Merged features for {len(result)} symbols")
            return result
            
        except Exception as e:
            logger.error(f"Error merging features: {e}")
            raise FundamentalTrendCalculationError(f"Feature merging failed: {e}")
    
    def _add_data_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add data quality flags to the result DataFrame.
        
        Args:
            df (pd.DataFrame): Features DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with data quality flags added
        """
        try:
            result = df.copy()
            
            # Calculate data completeness for each symbol
            feature_columns = [
                'ttm_rev_yoy', 'ttm_eps_yoy', 'ttm_ebit_yoy',
                'eps_yoy_slope_4q', 'rev_yoy_slope_4q', 'growth_consistency_8q'
            ]
            
            data_quality_flags = []
            
            for _, row in result.iterrows():
                flags = []
                
                # Check data completeness
                available_features = sum(1 for col in feature_columns if pd.notna(row.get(col)))
                completeness_ratio = available_features / len(feature_columns)
                
                if completeness_ratio < self.data_quality_threshold:
                    flags.append('LOW_COMPLETENESS')
                
                # Check for extreme values
                if pd.notna(row.get('ttm_eps_yoy')) and abs(row['ttm_eps_yoy']) > 3.0:
                    flags.append('EXTREME_EPS_GROWTH')
                
                if pd.notna(row.get('eps_yoy_slope_4q')) and abs(row['eps_yoy_slope_4q']) > 50:
                    flags.append('EXTREME_ACCELERATION')
                
                data_quality_flags.append('|'.join(flags) if flags else 'OK')
            
            result['data_quality_flags'] = data_quality_flags
            
            logger.debug(f"Added data quality flags for {len(result)} symbols")
            return result
            
        except Exception as e:
            logger.error(f"Error adding data quality flags: {e}")
            return df  # Return original DataFrame if flagging fails
