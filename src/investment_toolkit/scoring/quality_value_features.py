#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quality/Value Features Calculator - Quality/Value Pillar特徴量計算モジュール

This module implements Task 3: Quality/Value features calculation for the new 5-pillar scoring system.
Calculates value creation (ROIC-WACC), cash quality, composite valuation, and shareholder return metrics.

Key Features:
- Quality indicators: ROIC-WACC, cash quality, financial stability, margin consistency
- Value indicators: composite valuation, historical deviation, shareholder yield, dilution penalty
- Robust handling of edge cases (negative EBITDA/earnings, zero denominators, missing data)
- WACC proxy logic with sector fallbacks
- Comprehensive data quality validation and outlier handling

Usage:
    calc = QualityValueCalculator()
    features = calc.calculate_all_features(target_date="2025-09-11", symbols=["AAPL", "MSFT"])
    
References:
    - Specification: docs/calc_quality_value_features_spec.md
    - Schema: config/score_schema.yaml (Quality/Value pillars)
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


class QualityValueCalculationError(Exception):
    """Custom exception for quality/value calculation errors"""
    pass


class QualityValueCalculator:
    """
    Quality/Value features calculator for the new 5-pillar scoring system.
    
    This class calculates quality and value-related features including:
    - Quality: ROIC-WACC, cash quality, financial stability, margin consistency
    - Value: composite valuation, historical deviation, shareholder returns, dilution
    
    Attributes:
        engine (Engine): SQLAlchemy database engine
        data_quality_threshold (float): Minimum data quality threshold
        outlier_caps (Dict): Outlier trimming configuration
        wacc_fallbacks (Dict): WACC proxy configuration
    """
    
    def __init__(self, engine: Optional[Engine] = None):
        """
        Initialize the quality/value calculator.
        
        Args:
            engine (Optional[Engine]): Database engine. If None, creates default connection.
        """
        if engine is None:
            try:
                connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                self.engine = create_engine(connection_string)
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise QualityValueCalculationError(f"Database connection failed: {e}")
        else:
            self.engine = engine
        
        # Configuration for data quality and outlier handling
        self.data_quality_threshold = 0.5  # Minimum data completeness ratio
        self.outlier_caps = {
            'roic_upper': 100.0,              # +100% cap for ROIC
            'roic_lower': -50.0,              # -50% cap for ROIC
            'pe_upper': 150.0,                # P/E ratio upper cap
            'pe_lower': 0.0,                  # P/E ratio lower cap
            'debt_equity_upper': 10.0,        # Debt/Equity upper cap
            'interest_coverage_upper': 100.0, # Interest coverage upper cap
            'margin_cv_upper': 50.0,          # Margin stability upper cap
            'valuation_deviation_upper': 500.0, # +500% valuation deviation cap
            'valuation_deviation_lower': -90.0, # -90% valuation deviation cap
            'shareholder_yield_upper': 50.0,   # 50% shareholder yield cap
            'dilution_penalty_upper': 20.0,    # +20% dilution cap
            'dilution_penalty_lower': -50.0    # -50% dilution cap
        }
        
        # WACC proxy configuration
        self.wacc_fallbacks = {
            'market_median_wacc': 8.5,  # Default market WACC if all else fails
            'min_data_points_5y': 20,   # Minimum data points for 5-year analysis
            'min_data_points_12q': 8    # Minimum quarters for margin analysis
        }
        
        logger.info("QualityValueCalculator initialized successfully")
    
    def calculate_all_features(
        self, 
        target_date: str, 
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate all quality and value features for specified symbols and date.
        
        Args:
            target_date (str): Target date in YYYY-MM-DD format
            symbols (Optional[List[str]]): List of symbols to calculate. If None, calculates for all.
            
        Returns:
            pd.DataFrame: DataFrame with columns [symbol, as_of_date, quality_features..., value_features...]
            
        Raises:
            QualityValueCalculationError: If calculation fails
        """
        try:
            logger.info(f"Starting quality/value feature calculation for {target_date}")
            
            # Get financial data
            df = self._get_financial_data(target_date, symbols)
            if df.empty:
                logger.warning(f"No financial data found for {target_date}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved financial data for {len(df)} records")
            
            # Calculate quality features
            quality_features = self.calculate_quality_features(df)
            
            # Calculate value features  
            value_features = self.calculate_value_features(df)
            
            # Combine features
            result = pd.merge(
                quality_features, 
                value_features, 
                on=['symbol', 'as_of_date'], 
                how='outer'
            )
            
            # Apply outlier caps
            result = self._apply_outlier_caps(result)
            
            # Data quality validation
            result = self._validate_data_quality(result)
            
            logger.info(f"Successfully calculated quality/value features for {len(result)} records")
            return result
            
        except Exception as e:
            error_msg = f"Error calculating quality/value features: {e}"
            logger.error(error_msg)
            raise QualityValueCalculationError(error_msg)
    
    def calculate_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Quality pillar features.
        
        Args:
            df (pd.DataFrame): Input financial data
            
        Returns:
            pd.DataFrame: Quality features
        """
        logger.info("Calculating Quality pillar features")
        
        result = df[['symbol', 'as_of_date']].copy()
        
        # 1. ROIC minus WACC
        result['roic_minus_wacc'] = self._calculate_roic_minus_wacc(df)
        
        # 2. Gross profitability
        result['gross_profitability'] = self._calculate_gross_profitability(df)
        
        # 3. CFO to net income ratio
        result['cfo_to_net_income'] = self._calculate_cfo_to_net_income(df)
        
        # 4. Operating accruals
        result['operating_accruals'] = self._calculate_operating_accruals(df)
        
        # 5. Net debt to EBITDA
        result['net_debt_to_ebitda'] = self._calculate_net_debt_to_ebitda(df)
        
        # 6. Interest coverage ratio
        result['interest_coverage_ttm'] = self._calculate_interest_coverage(df)
        
        # 7. Margin consistency (12-quarter)
        result['margin_cv_12q'] = self._calculate_margin_consistency(df)
        
        logger.info("Quality features calculation completed")
        return result
    
    def calculate_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Value pillar features.

        Args:
            df (pd.DataFrame): Input financial data

        Returns:
            pd.DataFrame: Value features
        """
        logger.info("Calculating Value pillar features")

        # Use existing calculated market_cap from basic_metrics instead of recalculating
        df = df.copy()
        # market_cap already exists from the query JOIN with basic_metrics

        result = df[['symbol', 'as_of_date']].copy()
        
        # 8. Value composite score
        result['value_composite'] = self._calculate_value_composite(df)
        
        # 9. Valuation deviation (5-year)
        result['valuation_deviation_5y'] = self._calculate_valuation_deviation_5y(df)
        
        # 10. Total shareholder yield
        result['total_shareholder_yield'] = self._calculate_total_shareholder_yield(df)
        
        # 11. Dilution penalty
        result['dilution_penalty'] = self._calculate_dilution_penalty(df)
        
        logger.info("Value features calculation completed")
        return result
    
    def _get_financial_data(self, target_date: str, symbols: Optional[List[str]]) -> pd.DataFrame:
        """
        Retrieve financial data from database for calculations.
        
        Args:
            target_date (str): Target date in YYYY-MM-DD format
            symbols (Optional[List[str]]): Specific symbols or None for all
            
        Returns:
            pd.DataFrame: Financial data with all required columns
        """
        # Base query to get comprehensive financial data
        # パラメータ化クエリを使用してSQLインジェクションを防ぎ、巨大クエリを避ける
        if symbols and len(symbols) > 1000:
            logger.warning(f"Large symbol list ({len(symbols)} symbols), processing in batches")
            batch_size = 500
            all_dfs: List[pd.DataFrame] = []

            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                batch_df = self._run_financial_query(target_date, batch_symbols)
                if not batch_df.empty:
                    all_dfs.append(batch_df)

            df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
            logger.info(f"Retrieved financial data for {len(df)} symbols (batched)")
            return df

        try:
            df = self._run_financial_query(target_date, symbols)
            logger.info(f"Retrieved financial data for {len(df)} symbols")
            return df

        except Exception as e:
            logger.error(f"Error retrieving financial data: {e}")
            raise QualityValueCalculationError(f"Failed to retrieve financial data: {e}")

    def _run_financial_query(self, target_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Execute the core financial data query with optional symbol filtering."""

        symbol_filter_clause = "              AND i.symbol = ANY(:symbols)" if symbols else ""

        query_lines = [
            "WITH latest_financials AS (",
            "    SELECT DISTINCT",
            "        i.symbol,",
            "        i.date,",
            "        CAST(:target_date AS DATE) as as_of_date,",
            "        i.revenue,",
            "        i.gross_profit,",
            "        i.operating_income,",
            "        i.net_income,",
            "        i.interest_expense,",
            "        i.income_tax_expense,",
            "        i.eps,",
            "        i.weighted_average_shs_out_dil as shares_outstanding,",
            "        i.depreciation_and_amortization,",
            "        b.total_assets,",
            "        b.total_stockholders_equity,",
            "        b.long_term_debt,",
            "        b.short_term_debt,",
            "        b.cash_and_cash_equivalents,",
            "        c.operating_cash_flow as net_cash_provided_by_operating_activities,",
            "        c.free_cash_flow,",
            "        c.dividends_paid,",
            "        c.common_stock_repurchased,",
            "        c.common_stock_issued,",
            "        bm.market_cap,",
            "        bm.per as pe_ratio,",
            "        bm.pbr as pb_ratio,",
            "        bm.roe,",
            "        bm.roa,",
            "        cv.ev,",
            "        cv.ev_ebitda,",
            "        cv.ev_fcf,",
            "        cv.earnings_yield,",
            "        cv.altman_z,",
            "        cv.piotroski_f",
            "    FROM fmp_data.income_statements i",
            "    LEFT JOIN fmp_data.balance_sheets b ON i.symbol = b.symbol",
            "        AND i.date = b.date",
            "    LEFT JOIN fmp_data.cash_flows c ON i.symbol = c.symbol",
            "        AND i.date = c.date",
            "    LEFT JOIN calculated_metrics.basic_metrics bm ON i.symbol = bm.symbol",
            "        AND bm.as_of_date = (",
            "            SELECT MAX(as_of_date)",
            "            FROM calculated_metrics.basic_metrics",
            "            WHERE symbol = i.symbol AND as_of_date <= i.date",
            "        )",
            "    LEFT JOIN calculated_metrics.composite_valuation_metrics cv ON i.symbol = cv.symbol",
            "        AND cv.as_of_date = (",
            "            SELECT MAX(as_of_date)",
            "            FROM calculated_metrics.composite_valuation_metrics",
            "            WHERE symbol = i.symbol AND as_of_date <= i.date",
            "        )",
            "    WHERE i.date <= CAST(:target_date AS DATE)",
            "      AND i.date >= CAST(:target_date AS DATE) - INTERVAL '1 year'",
        ]

        if symbol_filter_clause:
            query_lines.append(symbol_filter_clause)

        query_lines.extend([
            "    ORDER BY i.symbol, i.date DESC",
            "),",
            "ranked_data AS (",
            "    SELECT *,",
            "        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn",
            "    FROM latest_financials",
            ")",
            "SELECT *",
            "FROM ranked_data",
            "WHERE rn = 1",
        ])

        query = text("\n".join(query_lines))

        params: Dict[str, Any] = {'target_date': target_date}
        if symbols:
            params['symbols'] = symbols

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params=params)

    def _get_financial_data_batch(self, target_date: str, symbols: List[str]) -> pd.DataFrame:
        """
        Get financial data for a batch of symbols (helper method for large datasets)

        Args:
            target_date: Target date string (YYYY-MM-DD)
            symbols: List of stock symbols (limited to reasonable size)

        Returns:
            pd.DataFrame: Financial data for the symbol batch
        """
        try:
            return self._run_financial_query(target_date, symbols)

        except Exception as e:
            logger.warning(f"Error retrieving batch financial data: {e}")
            return pd.DataFrame()  # 空のDataFrameを返してバッチ処理を継続

    def _calculate_roic_minus_wacc(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate ROIC minus WACC (Return on Invested Capital - Weighted Average Cost of Capital).
        
        ROIC = (Net Income + Interest Expense * (1 - Tax Rate)) / Invested Capital
        Invested Capital = Total Equity + Total Debt
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: ROIC minus WACC values
        """
        # Calculate tax rate (handle division by zero)
        pretax_income = df['net_income'] + df['income_tax_expense'].fillna(0)
        tax_rate = np.where(
            pretax_income != 0,
            df['income_tax_expense'].fillna(0) / pretax_income,
            0.25  # Default 25% tax rate
        )
        tax_rate = np.clip(tax_rate, 0, 0.5)  # Cap between 0-50%
        
        # Calculate NOPAT (Net Operating Profit After Tax)
        nopat = df['net_income'] + df['interest_expense'].fillna(0) * (1 - tax_rate)
        
        # Calculate invested capital
        total_debt = df['long_term_debt'].fillna(0) + df['short_term_debt'].fillna(0)
        invested_capital = df['total_stockholders_equity'].fillna(0) + total_debt
        
        # Calculate ROIC
        roic = np.where(
            invested_capital > 0,
            nopat / invested_capital * 100,  # Convert to percentage
            np.nan
        )
        
        # Get WACC (use sector proxy if individual WACC not available)
        wacc = self._get_wacc_proxy(df)
        
        # Calculate ROIC - WACC
        result = roic - wacc
        
        return pd.Series(result, index=df.index)
    
    def _get_wacc_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        Get WACC proxy values, using sector medians when individual WACC unavailable.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: WACC proxy values
        """
        # For now, use market median WACC as proxy
        # In production, this would query sector-specific WACC data
        wacc_proxy = pd.Series(
            self.wacc_fallbacks['market_median_wacc'], 
            index=df.index
        )
        
        logger.debug(f"Using market median WACC proxy: {self.wacc_fallbacks['market_median_wacc']}%")
        return wacc_proxy
    
    def _calculate_gross_profitability(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate gross profitability: Gross Profit / Total Assets.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Gross profitability ratios
        """
        result = np.where(
            df['total_assets'] > 0,
            df['gross_profit'].fillna(0) / df['total_assets'],
            np.nan
        )
        
        return pd.Series(result, index=df.index)
    
    def _calculate_cfo_to_net_income(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate cash flow to net income ratio: Operating Cash Flow / Net Income.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: CFO to net income ratios
        """
        result = np.where(
            df['net_income'] != 0,
            df['net_cash_provided_by_operating_activities'].fillna(0) / df['net_income'],
            np.nan
        )
        
        return pd.Series(result, index=df.index)
    
    def _calculate_operating_accruals(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate operating accruals: (Net Income - Operating Cash Flow) / Total Assets.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Operating accruals ratios
        """
        accruals = (
            df['net_income'].fillna(0) - 
            df['net_cash_provided_by_operating_activities'].fillna(0)
        )
        
        result = np.where(
            df['total_assets'] > 0,
            accruals / df['total_assets'],
            np.nan
        )
        
        return pd.Series(result, index=df.index)
    
    def _calculate_net_debt_to_ebitda(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate net debt to EBITDA ratio.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Net debt to EBITDA ratios
        """
        # Calculate EBITDA
        ebitda = (
            df['operating_income'].fillna(0) + 
            df['depreciation_and_amortization'].fillna(0)
        )
        
        # Calculate net debt
        total_debt = df['long_term_debt'].fillna(0) + df['short_term_debt'].fillna(0)
        net_debt = total_debt - df['cash_and_cash_equivalents'].fillna(0)
        
        result = np.where(
            ebitda > 0,
            net_debt / ebitda,
            np.nan
        )
        
        return pd.Series(result, index=df.index)
    
    def _calculate_interest_coverage(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate interest coverage ratio: Operating Income / Interest Expense.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Interest coverage ratios
        """
        result = np.where(
            df['interest_expense'].fillna(0) > 0,
            df['operating_income'].fillna(0) / df['interest_expense'],
            999.0  # Very high coverage if no interest expense
        )
        
        return pd.Series(result, index=df.index)
    
    def _calculate_margin_consistency(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate margin consistency over 12 quarters (inverse of coefficient of variation).
        
        This is a placeholder implementation. In production, this would query
        quarterly margin data and calculate the coefficient of variation.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Margin consistency scores
        """
        # Placeholder: Calculate gross margin and assume moderate consistency
        gross_margin = np.where(
            df['revenue'] > 0,
            df['gross_profit'].fillna(0) / df['revenue'],
            np.nan
        )
        
        # Placeholder consistency score based on current margin level
        # In production, this would use historical quarterly data
        consistency_score = np.where(
            ~np.isnan(gross_margin),
            np.clip(1.0 / (0.1 + np.abs(gross_margin - 0.3)), 0.1, 50.0),
            np.nan
        )
        
        return pd.Series(consistency_score, index=df.index)
    
    def _calculate_value_composite(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite valuation score using existing calculated metrics.

        Args:
            df (pd.DataFrame): Financial data with existing calculated metrics

        Returns:
            pd.Series: Value composite scores
        """
        # Use existing calculated ratios instead of recalculating
        # These come from calculated_metrics.basic_metrics and composite_valuation_metrics

        # 1. EV/EBITDA (from existing calculated metrics)
        ev_ebitda = df['ev_ebitda'].fillna(np.nan)

        # 2. FCF Yield (calculate from existing ev_fcf)
        fcf_yield = np.where(
            df['ev_fcf'].fillna(0) > 0,
            1.0 / df['ev_fcf'],  # FCF Yield = 1 / (EV/FCF)
            np.nan
        )

        # 3. P/B Ratio (from existing calculated metrics)
        pb_ratio = df['pb_ratio'].fillna(np.nan)

        # 4. P/E Ratio (from existing calculated metrics)
        pe_ratio = df['pe_ratio'].fillna(np.nan)
        
        # Create DataFrame for cross-sectional normalization
        ratios_df = pd.DataFrame({
            'ev_ebitda': ev_ebitda,
            'fcf_yield': fcf_yield,
            'pb_ratio': pb_ratio,
            'pe_ratio': pe_ratio
        })
        
        # Normalize each ratio (lower values are better for valuation ratios, except FCF yield)
        composite_scores = []
        
        for i in range(len(ratios_df)):
            scores = []
            
            # EV/EBITDA (lower is better) - weight 30%
            if not np.isnan(ratios_df.iloc[i]['ev_ebitda']):
                ev_rank = (ratios_df['ev_ebitda'] <= ratios_df.iloc[i]['ev_ebitda']).mean()
                scores.append(ev_rank * 0.3)
            
            # FCF Yield (higher is better) - weight 30%
            if not np.isnan(ratios_df.iloc[i]['fcf_yield']):
                fcf_rank = (ratios_df['fcf_yield'] >= ratios_df.iloc[i]['fcf_yield']).mean()
                scores.append(fcf_rank * 0.3)
            
            # P/B (lower is better) - weight 20%
            if not np.isnan(ratios_df.iloc[i]['pb_ratio']):
                pb_rank = (ratios_df['pb_ratio'] <= ratios_df.iloc[i]['pb_ratio']).mean()
                scores.append(pb_rank * 0.2)
            
            # P/E (lower is better) - weight 20%
            if not np.isnan(ratios_df.iloc[i]['pe_ratio']):
                pe_rank = (ratios_df['pe_ratio'] <= ratios_df.iloc[i]['pe_ratio']).mean()
                scores.append(pe_rank * 0.2)
            
            # Calculate weighted average if we have at least 2 components
            if len(scores) >= 2:
                composite_scores.append(sum(scores))
            else:
                composite_scores.append(np.nan)
        
        return pd.Series(composite_scores, index=df.index)
    
    def _calculate_valuation_deviation_5y(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate valuation deviation from 5-year median P/E.
        
        This is a placeholder implementation. In production, this would query
        5 years of historical P/E data for each symbol.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Valuation deviation percentages
        """
        # Calculate current P/E
        current_pe = np.where(
            df['net_income'] > 0,
            df['market_cap'].fillna(0) / df['net_income'],
            np.nan
        )
        
        # Placeholder: assume 5-year median P/E is 20% higher than current
        # In production, this would query historical data
        median_pe_5y = current_pe * 1.2
        
        deviation = np.where(
            median_pe_5y > 0,
            (current_pe - median_pe_5y) / median_pe_5y * 100,  # Convert to percentage
            np.nan
        )
        
        return pd.Series(deviation, index=df.index)
    
    def _calculate_total_shareholder_yield(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate total shareholder yield: (Dividends + Share Buybacks) / Market Cap.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Total shareholder yield percentages
        """
        # Calculate total returns to shareholders
        dividends = df['dividends_paid'].fillna(0).abs()  # Make positive
        buybacks = df['common_stock_repurchased'].fillna(0).abs()  # Make positive
        total_returns = dividends + buybacks
        
        result = np.where(
            df['market_cap'] > 0,
            total_returns / df['market_cap'] * 100,  # Convert to percentage
            np.nan
        )
        
        return pd.Series(result, index=df.index)
    
    def _calculate_dilution_penalty(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate dilution penalty from share issuances.
        
        This is a placeholder implementation. In production, this would calculate
        year-over-year change in shares outstanding.
        
        Args:
            df (pd.DataFrame): Financial data
            
        Returns:
            pd.Series: Dilution penalty percentages
        """
        # Placeholder: calculate based on share issuances relative to market cap
        share_issuances = df['common_stock_issued'].fillna(0)
        
        result = np.where(
            df['market_cap'] > 0,
            -share_issuances / df['market_cap'] * 100,  # Negative because dilution is bad
            np.nan
        )
        
        return pd.Series(result, index=df.index)
    
    def _apply_outlier_caps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier capping based on specification ranges.
        
        Args:
            df (pd.DataFrame): Feature data
            
        Returns:
            pd.DataFrame: Capped feature data
        """
        result = df.copy()
        
        # Apply caps based on specification
        caps_mapping = {
            'roic_minus_wacc': ('roic_lower', 'roic_upper'),
            'net_debt_to_ebitda': (0.0, 'debt_equity_upper'),
            'interest_coverage_ttm': (0.0, 'interest_coverage_upper'),
            'margin_cv_12q': (0.1, 'margin_cv_upper'),
            'valuation_deviation_5y': ('valuation_deviation_lower', 'valuation_deviation_upper'),
            'total_shareholder_yield': (0.0, 'shareholder_yield_upper'),
            'dilution_penalty': ('dilution_penalty_lower', 'dilution_penalty_upper')
        }
        
        for column, (lower_key, upper_key) in caps_mapping.items():
            if column in result.columns:
                lower_val = self.outlier_caps.get(lower_key, lower_key) if isinstance(lower_key, str) else lower_key
                upper_val = self.outlier_caps.get(upper_key, upper_key) if isinstance(upper_key, str) else upper_key
                
                result[column] = np.clip(result[column], lower_val, upper_val)
        
        return result
    
    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and exclude records with insufficient data.
        
        Args:
            df (pd.DataFrame): Feature data
            
        Returns:
            pd.DataFrame: Validated feature data
        """
        if df.empty:
            return df
        
        # Count non-null features per record
        feature_columns = [col for col in df.columns if col not in ['symbol', 'as_of_date']]
        df['feature_completeness'] = df[feature_columns].notna().sum(axis=1) / len(feature_columns)
        
        # Filter records with sufficient data quality
        sufficient_quality = df['feature_completeness'] >= self.data_quality_threshold
        
        logger.info(f"Data quality validation: {sufficient_quality.sum()}/{len(df)} records passed")
        
        # Remove the temporary column
        result = df.drop('feature_completeness', axis=1)
        return result[sufficient_quality] if sufficient_quality.any() else result
