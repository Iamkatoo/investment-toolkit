#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Price Momentum Features Calculator - Momentum/Risk Pillar特徴量計算モジュール

This module implements Task 4: Price momentum and risk features calculation for the new 5-pillar scoring system.
Calculates volatility-adjusted momentum, trend indicators, relative strength, and risk metrics.

Key Features:
- Volatility-adjusted momentum indicators (12m, 6m, 3m, 1m)
- Trend state indicators (200DMA, 50DMA, Golden Cross)
- 52-week high proximity and maintenance scores
- Sector and size bucket relative strength
- Risk metrics (idiosyncratic volatility, turnover, drawdown, earnings surprise vol)
- Robust handling of edge cases (IPOs, trading halts, missing data)
- Data quality validation and outlier handling

Usage:
    calc = PriceMomentumCalculator()
    features = calc.calculate_all_features(target_date="2025-09-11", symbols=["AAPL", "MSFT"])
    
References:
    - Specification: docs/calc_price_trend_features_spec.md
    - Schema: config/score_schema.yaml (Momentum and Risk pillars)
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
    from investment_toolkit.database.db_manager import DatabaseManager, get_db_connection
    from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")

logger = logging.getLogger(__name__)


class PriceMomentumCalculationError(Exception):
    """Custom exception for price momentum calculation errors"""
    pass


class PriceMomentumCalculator:
    """
    Price momentum and risk features calculator for the new 5-pillar scoring system.
    
    This class calculates momentum and risk-related features including:
    - Volatility-adjusted momentum indicators (multiple time horizons)
    - Trend state indicators (moving averages, golden cross)
    - 52-week high proximity and maintenance
    - Relative strength vs sector and size bucket
    - Risk metrics (volatility, drawdown, turnover, earnings surprise vol)
    
    Attributes:
        engine (Engine): SQLAlchemy database engine
        data_quality_threshold (float): Minimum data quality threshold
        outlier_caps (Dict): Outlier trimming configuration
        trading_days_per_period (Dict): Trading days mapping for different periods
    """
    
    def __init__(self, engine: Optional[Engine] = None):
        """
        Initialize the price momentum calculator.
        
        Args:
            engine (Optional[Engine]): Database engine. If None, creates default connection.
        """
        if engine is None:
            try:
                connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                self.engine = create_engine(connection_string)
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise PriceMomentumCalculationError(f"Database connection failed: {e}")
        else:
            self.engine = engine
        
        # Configuration for data quality and outlier handling
        self.data_quality_threshold = 0.5  # Minimum data completeness ratio
        self.outlier_caps = {
            'daily_return_upper': 0.30,     # +30% daily return cap
            'daily_return_lower': -0.30,    # -30% daily return cap
            'annual_return_upper': 5.0,     # +500% annual return cap
            'annual_return_lower': -0.90,   # -90% annual return cap
            'volatility_upper': 2.0,        # 200% annualized volatility cap
            'turnover_upper': 0.5,          # 50% daily turnover cap
            'drawdown_cap': -0.99           # -99% maximum drawdown cap
        }
        
        # Trading days mapping for different periods
        self.trading_days_per_period = {
            '1m': 21,    # ~1 month
            '3m': 63,    # ~3 months
            '6m': 126,   # ~6 months  
            '12m': 252,  # ~12 months
            '24m': 504   # ~24 months
        }
        
        logger.info("PriceMomentumCalculator initialized successfully")
    
    def calculate_all_features(
        self,
        target_date: str,
        symbols: Optional[List[str]] = None,
        save_to_db: bool = True,
        table_name: str = "calculated_metrics.price_trend_features"
    ) -> pd.DataFrame:
        """
        Calculate all price momentum and risk features for specified symbols and date.
        Uses existing calculated data when available to avoid redundant computation.

        Args:
            target_date (str): Target calculation date in 'YYYY-MM-DD' format
            symbols (Optional[List[str]]): List of symbols to calculate. If None, calculates for all available symbols.
            save_to_db (bool): Whether to save results to database
            table_name (str): Database table name for saving results

        Returns:
            pd.DataFrame: DataFrame with calculated features

        Raises:
            PriceMomentumCalculationError: If calculation fails
        """
        try:
            logger.info(f"Starting price momentum feature calculation for date: {target_date}")

            # First, try to get existing calculated features
            existing_features = self._get_existing_price_trend_features(target_date, symbols)

            if not existing_features.empty:
                logger.info(f"Found existing price trend features for {len(existing_features)} symbols")
                # Use existing features and only calculate missing ones if needed
                if symbols:
                    missing_symbols = [s for s in symbols if s not in existing_features['symbol'].values]
                    if missing_symbols:
                        logger.info(f"Calculating features for {len(missing_symbols)} missing symbols")
                        missing_features = self._calculate_new_features(target_date, missing_symbols)
                        if not missing_features.empty:
                            existing_features = pd.concat([existing_features, missing_features], ignore_index=True)

                return existing_features

            # If no existing features, calculate from scratch
            logger.info("No existing features found, calculating from scratch")
            return self._calculate_new_features(target_date, symbols, save_to_db, table_name)

        except Exception as e:
            logger.error(f"Error in price momentum feature calculation: {e}")
            raise PriceMomentumCalculationError(f"Calculation failed: {e}")

    def _get_existing_price_trend_features(self, target_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get existing price trend features from calculated_metrics.price_trend_features table.

        Args:
            target_date (str): Target date for features
            symbols (Optional[List[str]]): List of symbols to retrieve

        Returns:
            pd.DataFrame: Existing calculated features
        """
        try:
            # Build symbol filter
            symbol_filter = ""
            if symbols:
                symbol_list = "', '".join(symbols)
                symbol_filter = f"AND symbol IN ('{symbol_list}')"

            query = f"""
            SELECT
                symbol,
                as_of_date,
                mom_1m,
                mom_3m,
                mom_6m,
                mom_12m,
                vol_adj_momentum_composite,
                momentum_persistence,
                is_above_200dma,
                is_above_50dma,
                is_gc_50_200,
                trend_strength,
                dist_to_52w_high,
                post_break_hold_score,
                sector_rel_strength,
                sizebucket_rel_strength,
                idio_vol,
                turnover,
                max_drawdown_12_24m,
                earnings_surprise_vol,
                data_quality_flags,
                last_updated
            FROM calculated_metrics.price_trend_features
            WHERE as_of_date = '{target_date}'
            {symbol_filter}
            """

            return pd.read_sql(query, self.engine)

        except Exception as e:
            logger.warning(f"Error retrieving existing price trend features: {e}")
            return pd.DataFrame()

    def _calculate_new_features(
        self,
        target_date: str,
        symbols: Optional[List[str]] = None,
        save_to_db: bool = True,
        table_name: str = "calculated_metrics.price_trend_features"
    ) -> pd.DataFrame:
        """
        Calculate new price momentum features using the original calculation logic.
        """
        try:
            # Validate target date
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')

            # Get base price data
            price_data = self._get_price_data(target_date, symbols)
            if price_data.empty:
                logger.warning(f"No price data available for date {target_date}")
                return pd.DataFrame()

            # Validate input data
            if not self.validate_input_data(price_data):
                raise PriceMomentumCalculationError("Input data validation failed")

            # Initialize results DataFrame
            results = pd.DataFrame()
            results['symbol'] = price_data['symbol'].unique()
            results['as_of_date'] = target_date

            # Calculate momentum indicators
            logger.info("Calculating momentum indicators...")
            momentum_features = self._calculate_momentum_indicators(price_data, target_date)
            results = results.merge(momentum_features, on='symbol', how='left')

            # Calculate trend state indicators
            logger.info("Calculating trend state indicators...")
            trend_features = self._calculate_trend_indicators(price_data, target_date)
            results = results.merge(trend_features, on='symbol', how='left')

            # Calculate 52-week high indicators
            logger.info("Calculating 52-week high indicators...")
            high52w_features = self._calculate_52week_high_indicators(price_data, target_date)
            results = results.merge(high52w_features, on='symbol', how='left')

            # Calculate relative strength indicators
            logger.info("Calculating relative strength indicators...")
            relative_strength_features = self._calculate_relative_strength_indicators(price_data, target_date)
            results = results.merge(relative_strength_features, on='symbol', how='left')

            # Calculate risk indicators
            logger.info("Calculating risk indicators...")
            risk_features = self._calculate_risk_indicators(price_data, target_date)
            results = results.merge(risk_features, on='symbol', how='left')

            # Add metadata
            results['data_quality_flags'] = self._generate_quality_flags(results)
            results['last_updated'] = datetime.now()

            # Apply outlier caps and quality filters
            results = self._apply_outlier_caps(results)

            logger.info(f"Calculated features for {len(results)} symbols")

            # Save to database if requested
            if save_to_db:
                self._save_to_database(results, table_name)
                logger.info(f"Saved results to database table: {table_name}")

            return results

        except Exception as e:
            logger.error(f"Error in new feature calculation: {e}")
            raise PriceMomentumCalculationError(f"New calculation failed: {e}")
    
    def _get_price_data(self, target_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve price and volume data from database.
        
        Args:
            target_date (str): Target date for calculation
            symbols (Optional[List[str]]): List of symbols to retrieve
            
        Returns:
            pd.DataFrame: Price and volume data with calculated daily returns
        """
        try:
            # Calculate lookback date (need at least 24 months of data)
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            lookback_date = target_dt - timedelta(days=800)  # ~2.2 years for safety
            lookback_str = lookback_date.strftime('%Y-%m-%d')
            
            # Build symbol filter
            symbol_filter = ""
            if symbols:
                symbol_list = "', '".join(symbols)
                symbol_filter = f"AND symbol IN ('{symbol_list}')"
            
            query = f"""
            WITH base_prices AS (
                SELECT 
                    symbol,
                    date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
                FROM daily_prices
                WHERE date >= '{lookback_str}'
                  AND date <= '{target_date}'
                  AND volume > 0
                  {symbol_filter}
            ),
            price_with_returns AS (
                SELECT *,
                    CASE 
                        WHEN prev_close > 0 AND prev_close IS NOT NULL THEN 
                            (close - prev_close) / prev_close
                        ELSE NULL 
                    END as daily_return
                FROM base_prices
            )
            SELECT * FROM price_with_returns
            WHERE daily_return IS NOT NULL
            ORDER BY symbol, date
            """
            
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            logger.error(f"Error retrieving price data: {e}")
            raise PriceMomentumCalculationError(f"Price data retrieval failed: {e}")
    
    def _calculate_momentum_indicators(self, price_data: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Calculate volatility-adjusted momentum indicators.
        
        Calculates momentum for multiple time horizons (12m, 6m, 3m, 1m) adjusted for volatility,
        composite momentum score, and momentum persistence.
        
        Args:
            price_data (pd.DataFrame): Price data with daily returns
            target_date (str): Target calculation date
            
        Returns:
            pd.DataFrame: DataFrame with momentum indicators by symbol
        """
        results = []
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()

        for symbol, symbol_data in price_data.groupby('symbol', sort=False):
            symbol_data = symbol_data.sort_values('date').copy()

            # Get data up to target date (convert target_date to date type for comparison)
            symbol_data = symbol_data[symbol_data['date'] <= target_dt]
            
            if len(symbol_data) == 0:
                continue
            
            row = {'symbol': symbol}
            
            # Calculate momentum for each period
            momentum_values = {}
            for period, days in self.trading_days_per_period.items():
                if period == '24m':  # Skip 24m for momentum calculation
                    continue
                    
                mom_value, vol_value = self._calculate_period_momentum(symbol_data, days, target_date)
                row[f'mom_{period}'] = mom_value
                momentum_values[period] = mom_value
            
            # Calculate composite momentum (weighted average of periods)
            # Weights: 12M(40%), 6M(30%), 3M(20%), 1M(10%)
            weights = {'12m': 0.4, '6m': 0.3, '3m': 0.2, '1m': 0.1}
            valid_momentum = {k: v for k, v in momentum_values.items() if pd.notna(v)}
            
            if len(valid_momentum) >= 2:  # Need at least 2 periods for composite
                total_weight = sum(weights[k] for k in valid_momentum.keys())
                if total_weight > 0:
                    row['vol_adj_momentum_composite'] = sum(
                        valid_momentum[k] * weights[k] for k in valid_momentum.keys()
                    ) / total_weight
                else:
                    row['vol_adj_momentum_composite'] = None
            else:
                row['vol_adj_momentum_composite'] = None
            
            # Calculate momentum persistence (6 months of positive monthly returns)
            row['momentum_persistence'] = self._calculate_momentum_persistence(symbol_data, target_date)
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _calculate_period_momentum(self, symbol_data: pd.DataFrame, lookback_days: int, target_date: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate volatility-adjusted momentum for a specific period.
        
        Args:
            symbol_data (pd.DataFrame): Price data for single symbol
            lookback_days (int): Number of trading days to look back
            target_date (str): Target calculation date
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (momentum_value, volatility_value)
        """
        try:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            
            # Get the most recent data point (target date or most recent available)
            current_data = symbol_data[symbol_data['date'] <= pd.to_datetime(target_date).date()]
            if current_data.empty:
                return None, None
            
            current_price = current_data.iloc[-1]['close']
            current_date = current_data.iloc[-1]['date']
            
            # Find price from lookback_days ago
            min_date = current_date - timedelta(days=lookback_days * 2)  # Safety margin for weekends/holidays
            lookback_data = symbol_data[symbol_data['date'] >= min_date]
            
            if len(lookback_data) < lookback_days * 0.8:  # Need at least 80% of expected data
                return None, None
            
            # Get price from approximately lookback_days trading days ago
            if len(lookback_data) >= lookback_days:
                past_price = lookback_data.iloc[-(lookback_days)]['close']
            else:
                past_price = lookback_data.iloc[0]['close']
            
            if past_price <= 0:
                return None, None
            
            # Calculate raw return
            raw_return = (current_price - past_price) / past_price
            
            # Calculate volatility (standard deviation of daily returns)
            recent_returns = symbol_data[symbol_data['date'] >= current_date - timedelta(days=lookback_days * 2)]
            daily_returns = recent_returns['daily_return'].dropna()
            
            if len(daily_returns) < 10:  # Need minimum data for volatility
                return None, None
            
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
            
            if volatility <= 0.001:  # Very low volatility case
                momentum = raw_return * 1000  # High momentum score for low-vol positive returns
            else:
                momentum = raw_return / volatility  # Risk-adjusted return
            
            # Apply outlier caps
            momentum = np.clip(
                momentum,
                self.outlier_caps['annual_return_lower'],
                self.outlier_caps['annual_return_upper']
            )
            
            return momentum, volatility
            
        except Exception as e:
            logger.warning(f"Error calculating momentum for period {lookback_days}: {e}")
            return None, None
    
    def _calculate_momentum_persistence(self, symbol_data: pd.DataFrame, target_date: str) -> Optional[float]:
        """
        Calculate momentum persistence (ratio of positive monthly returns in recent 6 months).

        Args:
            symbol_data (pd.DataFrame): Price data for single symbol
            target_date (str): Target calculation date

        Returns:
            Optional[float]: Persistence ratio (0-1) or None if insufficient data
        """
        try:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')

            # Get 6 months of data - Convert to date for consistent comparison
            lookback_date = target_dt - timedelta(days=200)  # ~6.5 months for safety
            lookback_date_obj = lookback_date.date()
            target_date_obj = target_dt.date()

            # Filter data using date objects for consistency
            recent_data = symbol_data[symbol_data['date'] >= lookback_date_obj]
            recent_data = recent_data[recent_data['date'] <= target_date_obj]
            
            if len(recent_data) < 100:  # Need sufficient data
                return None
            
            # Calculate monthly returns (approximately every 21 trading days)
            monthly_returns = []
            for i in range(6):  # 6 months
                end_idx = len(recent_data) - 1 - (i * 21)
                start_idx = end_idx - 21
                
                if start_idx < 0 or end_idx < 21:
                    break
                
                start_price = recent_data.iloc[start_idx]['close']
                end_price = recent_data.iloc[end_idx]['close']
                
                if start_price > 0:
                    monthly_return = (end_price - start_price) / start_price
                    monthly_returns.append(monthly_return)
            
            if len(monthly_returns) < 3:  # Need at least 3 months
                return None
            
            # Calculate persistence (ratio of positive months)
            positive_months = sum(1 for ret in monthly_returns if ret > 0)
            persistence = positive_months / len(monthly_returns)
            
            return persistence
            
        except Exception as e:
            logger.warning(f"Error calculating momentum persistence: {e}")
            return None
    
    def _calculate_trend_indicators(self, price_data: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Calculate trend state indicators.
        
        Calculates moving average based trend indicators including:
        - Above 200-day moving average flag
        - Above 50-day moving average flag 
        - Golden cross status (50DMA > 200DMA)
        - Composite trend strength score
        
        Args:
            price_data (pd.DataFrame): Price data with daily closes
            target_date (str): Target calculation date
            
        Returns:
            pd.DataFrame: DataFrame with trend indicators by symbol
        """
        results = []
        
        for symbol, symbol_data in price_data.groupby('symbol', sort=False):
            symbol_data = symbol_data.sort_values('date').copy()
            
            # Get data up to target date
            symbol_data = symbol_data[symbol_data['date'] <= pd.to_datetime(target_date).date()]
            
            if len(symbol_data) < 200:  # Need at least 200 days for 200DMA
                results.append({
                    'symbol': symbol,
                    'is_above_200dma': None,
                    'is_above_50dma': None,
                    'is_gc_50_200': None,
                    'trend_strength': None
                })
                continue
            
            # Calculate moving averages
            symbol_data['sma_50'] = symbol_data['close'].rolling(window=50, min_periods=45).mean()
            symbol_data['sma_200'] = symbol_data['close'].rolling(window=200, min_periods=180).mean()
            
            # Get most recent values
            latest_data = symbol_data.iloc[-1]
            current_price = latest_data['close']
            sma_50 = latest_data['sma_50'] if pd.notna(latest_data['sma_50']) else None
            sma_200 = latest_data['sma_200'] if pd.notna(latest_data['sma_200']) else None
            
            # Calculate indicators (convert to boolean for database compatibility)
            is_above_50dma = bool(sma_50 and current_price > sma_50)
            is_above_200dma = bool(sma_200 and current_price > sma_200)
            is_gc_50_200 = bool(sma_50 and sma_200 and sma_50 > sma_200)
            
            # Calculate trend strength composite (weighted combination)
            # Weights: 200DMA(50%), 50DMA(30%), Golden Cross(20%)
            trend_strength = (is_above_200dma * 0.5 + is_above_50dma * 0.3 + is_gc_50_200 * 0.2)
            
            results.append({
                'symbol': symbol,
                'is_above_200dma': is_above_200dma,
                'is_above_50dma': is_above_50dma,
                'is_gc_50_200': is_gc_50_200,
                'trend_strength': trend_strength
            })
        
        return pd.DataFrame(results)
    
    def _calculate_52week_high_indicators(self, price_data: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Calculate 52-week high proximity and maintenance indicators.
        
        Calculates:
        - Distance to 52-week high (current price / 52W high)
        - Post-breakout holding score (ability to maintain above previous highs)
        
        Args:
            price_data (pd.DataFrame): Price data with daily highs and closes
            target_date (str): Target calculation date
            
        Returns:
            pd.DataFrame: DataFrame with 52-week high indicators by symbol
        """
        results = []
        
        for symbol, symbol_data in price_data.groupby('symbol', sort=False):
            symbol_data = symbol_data.sort_values('date').copy()
            
            # Get data up to target date
            symbol_data = symbol_data[symbol_data['date'] <= pd.to_datetime(target_date).date()]
            
            if len(symbol_data) < 252:  # Need at least 1 year of data
                results.append({
                    'symbol': symbol,
                    'dist_to_52w_high': None,
                    'post_break_hold_score': None
                })
                continue
            
            # Calculate 52-week high
            symbol_data['high_52w'] = symbol_data['high'].rolling(window=252, min_periods=200).max()
            
            # Get most recent values
            latest_data = symbol_data.iloc[-1]
            current_price = latest_data['close']
            high_52w = latest_data['high_52w']
            
            # Calculate distance to 52-week high
            if pd.notna(high_52w) and high_52w > 0:
                dist_to_52w_high = current_price / high_52w
            else:
                dist_to_52w_high = None
            
            # Calculate post-breakout holding score
            post_break_hold_score = self._calculate_post_break_hold_score(symbol_data, target_date)
            
            results.append({
                'symbol': symbol,
                'dist_to_52w_high': dist_to_52w_high,
                'post_break_hold_score': post_break_hold_score
            })
        
        return pd.DataFrame(results)
    
    def _calculate_post_break_hold_score(self, symbol_data: pd.DataFrame, target_date: str) -> Optional[float]:
        """
        Calculate post-breakout holding score.
        
        Measures the ability to maintain price levels after breaking previous highs.
        
        Args:
            symbol_data (pd.DataFrame): Price data for single symbol
            target_date (str): Target calculation date
            
        Returns:
            Optional[float]: Holding score (0-1) or None if no breakout detected
        """
        try:
            if len(symbol_data) < 252:
                return None
            
            # Calculate rolling 52-week highs (excluding current day)
            symbol_data = symbol_data.copy()
            symbol_data['prev_high_52w'] = symbol_data['high'].shift(1).rolling(window=252, min_periods=200).max()
            
            # Identify breakout points (close > previous 52-week high)
            symbol_data['is_breakout'] = (
                (symbol_data['close'] > symbol_data['prev_high_52w']) & 
                (symbol_data['prev_high_52w'].notna())
            )
            
            # Find most recent breakout
            breakout_dates = symbol_data[symbol_data['is_breakout']]['date']
            if breakout_dates.empty:
                return 0.0  # No breakouts detected
            
            most_recent_breakout = breakout_dates.iloc[-1]
            breakout_idx = symbol_data[symbol_data['date'] == most_recent_breakout].index[0]
            
            # Get data since breakout (up to 20 trading days)
            post_breakout_data = symbol_data.loc[breakout_idx:].head(21)  # Include breakout day + 20 days
            
            if len(post_breakout_data) < 2:
                return None
            
            # Calculate holding strength (percentage of days maintaining >= 95% of breakout high)
            breakout_high = symbol_data.loc[breakout_idx, 'prev_high_52w']
            threshold_price = breakout_high * 0.95  # 95% of breakout level
            
            days_above_threshold = (post_breakout_data['close'] >= threshold_price).sum()
            total_days = len(post_breakout_data)
            
            hold_score = days_above_threshold / total_days if total_days > 0 else 0.0
            
            return hold_score
            
        except Exception as e:
            logger.warning(f"Error calculating post-breakout hold score: {e}")
            return None
    
    def _calculate_relative_strength_indicators(self, price_data: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Calculate sector and size bucket relative strength indicators.
        
        Calculates 3-month relative performance vs:
        - Sector average
        - Size bucket (market cap category) average
        
        Args:
            price_data (pd.DataFrame): Price data with daily returns
            target_date (str): Target calculation date
            
        Returns:
            pd.DataFrame: DataFrame with relative strength indicators by symbol
        """
        results = []
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()

        # Get sector and market cap information
        sector_data = self._get_sector_data(target_date)
        size_bucket_data = self._get_size_bucket_data(target_date)

        for symbol, symbol_data in price_data.groupby('symbol', sort=False):
            symbol_data = symbol_data.sort_values('date').copy()

            # Get data up to target date
            symbol_data = symbol_data[symbol_data['date'] <= target_dt]
            
            if len(symbol_data) < 63:  # Need at least 3 months of data
                results.append({
                    'symbol': symbol,
                    'sector_rel_strength': None,
                    'sizebucket_rel_strength': None
                })
                continue
            
            # Calculate 3-month relative strength vs sector
            sector_rel_strength = self._calculate_sector_relative_strength(
                symbol_data, symbol, sector_data, target_date
            )
            
            # Calculate 3-month relative strength vs size bucket  
            sizebucket_rel_strength = self._calculate_size_bucket_relative_strength(
                symbol_data, symbol, size_bucket_data, target_date
            )
            
            results.append({
                'symbol': symbol,
                'sector_rel_strength': sector_rel_strength,
                'sizebucket_rel_strength': sizebucket_rel_strength
            })
        
        return pd.DataFrame(results)
    
    def _get_sector_data(self, target_date: str) -> pd.DataFrame:
        """
        Get sector classification and returns data.
        
        Args:
            target_date (str): Target calculation date
            
        Returns:
            pd.DataFrame: Sector data with returns
        """
        try:
            query = f"""
            WITH sector_returns AS (
                SELECT
                    c.symbol,
                    c.raw_sector as sector,
                    dp.date,
                    (dp.close / LAG(dp.close) OVER (PARTITION BY dp.symbol ORDER BY dp.date) - 1) as daily_return
                FROM reference.company_gics c
                JOIN fmp_data.daily_prices dp ON c.symbol = dp.symbol
                WHERE dp.date >= ('{target_date}'::date - INTERVAL '4 months')
                  AND dp.date <= '{target_date}'
                  AND dp.volume > 0
                  AND c.raw_sector IS NOT NULL
            )
            SELECT * FROM sector_returns
            WHERE daily_return IS NOT NULL
            ORDER BY sector, date
            """
            
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            logger.warning(f"Error retrieving sector data: {e}")
            return pd.DataFrame()
    
    def _get_size_bucket_data(self, target_date: str) -> pd.DataFrame:
        """
        Get size bucket classification and returns data.
        
        Args:
            target_date (str): Target calculation date
            
        Returns:
            pd.DataFrame: Size bucket data with returns
        """
        try:
            query = f"""
            WITH market_cap_buckets AS (
                SELECT
                    symbol,
                    CASE
                        WHEN market_cap >= 10000000000 THEN 'large_cap'
                        WHEN market_cap >= 2000000000 THEN 'mid_cap'
                        WHEN market_cap >= 300000000 THEN 'small_cap'
                        ELSE 'micro_cap'
                    END as size_bucket
                FROM calculated_metrics.basic_metrics
                WHERE as_of_date = (
                    SELECT MAX(as_of_date)
                    FROM calculated_metrics.basic_metrics
                    WHERE as_of_date <= '{target_date}'
                )
            ),
            size_bucket_returns AS (
                SELECT
                    mcb.symbol,
                    mcb.size_bucket,
                    dp.date,
                    (dp.close / LAG(dp.close) OVER (PARTITION BY dp.symbol ORDER BY dp.date) - 1) as daily_return
                FROM market_cap_buckets mcb
                JOIN fmp_data.daily_prices dp ON mcb.symbol = dp.symbol
                WHERE dp.date >= ('{target_date}'::date - INTERVAL '4 months')
                  AND dp.date <= '{target_date}'
                  AND dp.volume > 0
            )
            SELECT * FROM size_bucket_returns
            ORDER BY size_bucket, date
            """
            
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            logger.warning(f"Error retrieving size bucket data: {e}")
            return pd.DataFrame()
    
    def _calculate_sector_relative_strength(
        self, 
        symbol_data: pd.DataFrame, 
        symbol: str, 
        sector_data: pd.DataFrame,
        target_date: str
    ) -> Optional[float]:
        """
        Calculate 3-month sector relative strength.
        
        Args:
            symbol_data (pd.DataFrame): Individual symbol data
            symbol (str): Symbol identifier
            sector_data (pd.DataFrame): Sector returns data  
            target_date (str): Target calculation date
            
        Returns:
            Optional[float]: 3-month relative strength vs sector or None
        """
        try:
            if sector_data.empty:
                return None
            
            # Get symbol's sector
            symbol_sector_data = sector_data[sector_data['symbol'] == symbol]
            if symbol_sector_data.empty:
                return None
            
            sector = symbol_sector_data['sector'].iloc[0]
            
            # Get 3 months of data
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = target_dt - timedelta(days=100)  # ~3 months with margin
            start_date_obj = start_date.date()

            symbol_recent = symbol_data[symbol_data['date'] >= start_date_obj]
            if len(symbol_recent) < 50:  # Need sufficient data
                return None
            
            # Calculate symbol's average daily return over 3 months
            symbol_avg_return = symbol_recent['daily_return'].mean()
            
            # Calculate sector average daily return over 3 months
            start_date_obj = start_date.date()
            sector_recent = sector_data[
                (sector_data['sector'] == sector) &
                (sector_data['date'] >= start_date_obj)
            ]
            
            if len(sector_recent) < 50:
                return None
            
            sector_avg_return = sector_recent.groupby('date')['daily_return'].mean().mean()
            
            # Calculate relative strength (difference in average returns)
            relative_strength = symbol_avg_return - sector_avg_return
            
            return relative_strength
            
        except Exception as e:
            logger.warning(f"Error calculating sector relative strength for {symbol}: {e}")
            return None
    
    def _calculate_size_bucket_relative_strength(
        self, 
        symbol_data: pd.DataFrame, 
        symbol: str, 
        size_bucket_data: pd.DataFrame,
        target_date: str
    ) -> Optional[float]:
        """
        Calculate 3-month size bucket relative strength.
        
        Args:
            symbol_data (pd.DataFrame): Individual symbol data
            symbol (str): Symbol identifier  
            size_bucket_data (pd.DataFrame): Size bucket returns data
            target_date (str): Target calculation date
            
        Returns:
            Optional[float]: 3-month relative strength vs size bucket or None
        """
        try:
            if size_bucket_data.empty:
                return None
            
            # Get symbol's size bucket
            symbol_size_data = size_bucket_data[size_bucket_data['symbol'] == symbol]
            if symbol_size_data.empty:
                return None
            
            size_bucket = symbol_size_data['size_bucket'].iloc[0]
            
            # Get 3 months of data
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = target_dt - timedelta(days=100)  # ~3 months with margin
            start_date_obj = start_date.date()

            symbol_recent = symbol_data[symbol_data['date'] >= start_date_obj]
            if len(symbol_recent) < 50:  # Need sufficient data
                return None
            
            # Calculate symbol's average daily return over 3 months  
            symbol_avg_return = symbol_recent['daily_return'].mean()
            
            # Calculate size bucket average daily return over 3 months
            start_date_obj = start_date.date()
            bucket_recent = size_bucket_data[
                (size_bucket_data['size_bucket'] == size_bucket) &
                (size_bucket_data['date'] >= start_date_obj)
            ]
            
            if len(bucket_recent) < 50:
                return None
            
            bucket_avg_return = bucket_recent.groupby('date')['daily_return'].mean().mean()
            
            # Calculate relative strength (difference in average returns)
            relative_strength = symbol_avg_return - bucket_avg_return
            
            return relative_strength
            
        except Exception as e:
            logger.warning(f"Error calculating size bucket relative strength for {symbol}: {e}")
            return None
    
    def _calculate_risk_indicators(self, price_data: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Calculate risk and stability indicators.
        
        Calculates:
        - Idiosyncratic volatility (market/sector regression residuals)
        - Turnover (liquidity indicator)
        - Maximum drawdown stability (12-24 month)  
        - Earnings surprise volatility (predictability)
        
        Args:
            price_data (pd.DataFrame): Price data with daily returns and volume
            target_date (str): Target calculation date
            
        Returns:
            pd.DataFrame: DataFrame with risk indicators by symbol
        """
        results = []
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()

        # Get market and sector data for idiosyncratic vol calculation
        market_sector_data = self._get_market_sector_data(target_date)
        
        # Get earnings surprise data  
        earnings_surprise_data = self._get_earnings_surprise_data(target_date)
        
        # Get shares outstanding data for turnover calculation
        shares_data = self._get_shares_outstanding_data(target_date)
        
        for symbol, symbol_data in price_data.groupby('symbol', sort=False):
            symbol_data = symbol_data.sort_values('date').copy()

            # Get data up to target date
            symbol_data = symbol_data[symbol_data['date'] <= target_dt]
            
            if len(symbol_data) < 252:  # Need at least 1 year of data
                results.append({
                    'symbol': symbol,
                    'idio_vol': None,
                    'turnover': None,
                    'max_drawdown_12_24m': None,
                    'earnings_surprise_vol': None
                })
                continue
            
            # Calculate idiosyncratic volatility
            idio_vol = self._calculate_idiosyncratic_volatility(
                symbol_data, symbol, market_sector_data, target_date
            )
            
            # Calculate turnover
            turnover = self._calculate_turnover(
                symbol_data, symbol, shares_data, target_date
            )
            
            # Calculate maximum drawdown stability
            max_drawdown_stability = self._calculate_max_drawdown_stability(
                symbol_data, target_date
            )
            
            # Calculate earnings surprise volatility
            earnings_surprise_vol = self._calculate_earnings_surprise_volatility(
                symbol, earnings_surprise_data, target_date
            )
            
            results.append({
                'symbol': symbol,
                'idio_vol': idio_vol,
                'turnover': turnover,
                'max_drawdown_12_24m': max_drawdown_stability,
                'earnings_surprise_vol': earnings_surprise_vol
            })
        
        return pd.DataFrame(results)
    
    def _get_market_sector_data(self, target_date: str) -> pd.DataFrame:
        """Get market and sector returns for idiosyncratic vol calculation."""
        try:
            query = f"""
            WITH daily_returns AS (
                SELECT
                    symbol,
                    date,
                    (close / LAG(close) OVER (PARTITION BY symbol ORDER BY date) - 1) as daily_return
                FROM fmp_data.daily_prices
                WHERE date >= ('{target_date}'::date - INTERVAL '14 months')
                  AND date <= '{target_date}'
                  AND volume > 0
            ),
            market_returns AS (
                SELECT
                    date,
                    AVG(daily_return) as market_return
                FROM daily_returns
                WHERE daily_return IS NOT NULL
                GROUP BY date
            ),
            sector_daily_returns AS (
                SELECT
                    c.raw_sector as sector,
                    dr.date,
                    dr.daily_return
                FROM reference.company_gics c
                JOIN daily_returns dr ON c.symbol = dr.symbol
                WHERE c.raw_sector IS NOT NULL
                  AND dr.daily_return IS NOT NULL
            ),
            sector_returns AS (
                SELECT
                    sector,
                    date,
                    AVG(daily_return) as sector_return
                FROM sector_daily_returns
                GROUP BY sector, date
            )
            SELECT
                mr.date,
                mr.market_return,
                sr.sector,
                sr.sector_return
            FROM market_returns mr
            LEFT JOIN sector_returns sr ON mr.date = sr.date
            ORDER BY date, sector
            """
            
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            logger.warning(f"Error retrieving market/sector data: {e}")
            return pd.DataFrame()
    
    def _get_earnings_surprise_data(self, target_date: str) -> pd.DataFrame:
        """Get earnings surprise data for volatility calculation."""
        try:
            # earnings_surprises table does not exist, return empty DataFrame
            # This feature will be disabled until proper data source is available
            logger.warning("Earnings surprise data not available - table does not exist")
            return pd.DataFrame(columns=['symbol', 'announcement_date', 'actual_eps', 'consensus_eps', 'surprise_pct'])

        except Exception as e:
            logger.warning(f"Error retrieving earnings surprise data: {e}")
            return pd.DataFrame()
    
    def _get_shares_outstanding_data(self, target_date: str) -> pd.DataFrame:
        """Get shares outstanding data for turnover calculation."""
        try:
            query = f"""
            SELECT
                symbol,
                outstanding_shares as shares_outstanding
            FROM fmp_data.shares
            WHERE date = (
                SELECT MAX(date)
                FROM fmp_data.shares
                WHERE date <= '{target_date}'
            )
            AND outstanding_shares > 0
            """

            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            logger.warning(f"Error retrieving shares outstanding data: {e}")
            return pd.DataFrame()
    
    def _calculate_idiosyncratic_volatility(
        self, 
        symbol_data: pd.DataFrame, 
        symbol: str,
        market_sector_data: pd.DataFrame, 
        target_date: str
    ) -> Optional[float]:
        """
        Calculate idiosyncratic volatility (residuals from market/sector regression).
        
        Args:
            symbol_data (pd.DataFrame): Individual symbol data
            symbol (str): Symbol identifier
            market_sector_data (pd.DataFrame): Market and sector returns data
            target_date (str): Target calculation date
            
        Returns:
            Optional[float]: Annualized idiosyncratic volatility or None
        """
        try:
            if market_sector_data.empty:
                return None
            
            # Get 1 year of data
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = target_dt - timedelta(days=400)  # ~13 months with margin
            start_date_obj = start_date.date()

            symbol_recent = symbol_data[symbol_data['date'] >= start_date_obj]
            if len(symbol_recent) < 200:  # Need sufficient data
                return None
            
            # Get symbol's sector from market_sector_data or estimate
            # For simplicity, use a market-only model if sector data not available
            market_data = market_sector_data[['date', 'market_return']].drop_duplicates()
            
            # Merge symbol returns with market returns
            merged_data = symbol_recent[['date', 'daily_return']].merge(
                market_data, on='date', how='inner'
            )
            
            if len(merged_data) < 150:
                return None
            
            # Calculate residuals from simplified market model
            # Using basic approach: residual = individual_return - beta * market_return
            symbol_returns = merged_data['daily_return'].fillna(0)
            market_returns = merged_data['market_return'].fillna(0)
            
            # Simple beta calculation (correlation approach)
            if market_returns.std() > 0.0001:
                beta = np.corrcoef(symbol_returns, market_returns)[0,1] * (
                    symbol_returns.std() / market_returns.std()
                )
            else:
                beta = 1.0
            
            # Calculate residuals
            residuals = symbol_returns - beta * market_returns
            
            # Calculate annualized idiosyncratic volatility
            idio_vol = residuals.std() * np.sqrt(252)
            
            return idio_vol
            
        except Exception as e:
            logger.warning(f"Error calculating idiosyncratic volatility for {symbol}: {e}")
            return None
    
    def _calculate_turnover(
        self, 
        symbol_data: pd.DataFrame, 
        symbol: str,
        shares_data: pd.DataFrame, 
        target_date: str
    ) -> Optional[float]:
        """
        Calculate average daily turnover (volume / shares outstanding).
        
        Args:
            symbol_data (pd.DataFrame): Individual symbol data
            symbol (str): Symbol identifier
            shares_data (pd.DataFrame): Shares outstanding data
            target_date (str): Target calculation date
            
        Returns:
            Optional[float]: 30-day average turnover ratio or None
        """
        try:
            if shares_data.empty:
                return None
            
            # Get shares outstanding for symbol
            symbol_shares = shares_data[shares_data['symbol'] == symbol]
            if symbol_shares.empty:
                return None
            
            shares_outstanding = symbol_shares['shares_outstanding'].iloc[0]
            if shares_outstanding <= 0:
                return None
            
            # Get 30 days of data
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = target_dt - timedelta(days=45)  # 30 days with margin
            start_date_obj = start_date.date()

            symbol_recent = symbol_data[symbol_data['date'] >= start_date_obj]
            symbol_recent = symbol_recent[symbol_recent['volume'] > 0]
            
            if len(symbol_recent) < 20:  # Need sufficient data
                return None
            
            # Calculate daily turnover ratios
            symbol_recent['daily_turnover'] = symbol_recent['volume'] / shares_outstanding
            
            # Calculate 30-day average turnover
            avg_turnover = symbol_recent['daily_turnover'].mean()
            
            return avg_turnover
            
        except Exception as e:
            logger.warning(f"Error calculating turnover for {symbol}: {e}")
            return None
    
    def _calculate_max_drawdown_stability(
        self, 
        symbol_data: pd.DataFrame, 
        target_date: str
    ) -> Optional[float]:
        """
        Calculate maximum drawdown stability score (inverse of max drawdown).
        
        Args:
            symbol_data (pd.DataFrame): Individual symbol data
            target_date (str): Target calculation date
            
        Returns:
            Optional[float]: Stability score (higher = more stable) or None
        """
        try:
            # Get 24 months of data
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = target_dt - timedelta(days=750)  # ~24 months with margin
            start_date_obj = start_date.date()

            symbol_recent = symbol_data[symbol_data['date'] >= start_date_obj]
            
            if len(symbol_recent) < 400:  # Need sufficient data (~16+ months)
                return None
            
            # Calculate rolling maximum
            symbol_recent = symbol_recent.copy()
            symbol_recent['rolling_max'] = symbol_recent['close'].expanding().max()
            
            # Calculate drawdown
            symbol_recent['drawdown'] = (symbol_recent['close'] - symbol_recent['rolling_max']) / symbol_recent['rolling_max']
            
            # Get maximum drawdown (most negative value)
            max_drawdown = symbol_recent['drawdown'].min()
            
            # Apply cap to prevent extreme values
            max_drawdown = max(max_drawdown, self.outlier_caps['drawdown_cap'])
            
            # Convert to stability score (inverse relationship)
            # Formula: 1 / (1 - max_drawdown) gives higher scores for smaller drawdowns
            if max_drawdown < 0:
                stability_score = 1.0 / (1.0 - max_drawdown)
            else:
                stability_score = 1.0  # No drawdown case
            
            return stability_score
            
        except Exception as e:
            logger.warning(f"Error calculating max drawdown stability: {e}")
            return None
    
    def _calculate_earnings_surprise_volatility(
        self, 
        symbol: str,
        earnings_surprise_data: pd.DataFrame, 
        target_date: str
    ) -> Optional[float]:
        """
        Calculate earnings surprise volatility (predictability score).
        
        Args:
            symbol (str): Symbol identifier
            earnings_surprise_data (pd.DataFrame): Earnings surprise data
            target_date (str): Target calculation date
            
        Returns:
            Optional[float]: Predictability score (higher = more predictable) or None
        """
        try:
            if earnings_surprise_data.empty:
                return None
            
            # Get symbol's earnings surprise data
            symbol_surprises = earnings_surprise_data[earnings_surprise_data['symbol'] == symbol]
            if len(symbol_surprises) < 8:  # Need at least 8 quarters (2 years)
                return None
            
            # Calculate standard deviation of surprise percentages
            surprise_vol = symbol_surprises['surprise_pct'].std()
            
            # Convert to predictability score (inverse relationship)
            # Formula: 1 / (1 + surprise_vol) gives higher scores for lower volatility
            predictability_score = 1.0 / (1.0 + surprise_vol)
            
            return predictability_score
            
        except Exception as e:
            logger.warning(f"Error calculating earnings surprise volatility for {symbol}: {e}")
            return None
    
    def _generate_quality_flags(self, results: pd.DataFrame) -> List[str]:
        """
        Generate data quality flags for each row.
        
        Args:
            results (pd.DataFrame): Results dataframe with calculated features
            
        Returns:
            List[str]: Quality flags for each row
        """
        flags = []
        for _, row in results.iterrows():
            flag_list = []
            
            # Check for missing momentum data
            momentum_cols = [f'mom_{p}' for p in ['12m', '6m', '3m', '1m']]
            missing_momentum = sum(1 for col in momentum_cols if pd.isna(row[col]))
            if missing_momentum > 2:
                flag_list.append("INSUFFICIENT_MOMENTUM_DATA")
            elif missing_momentum > 0:
                flag_list.append("PARTIAL_MOMENTUM_DATA")
            
            # Check for missing trend data
            trend_cols = ['is_above_200dma', 'is_above_50dma', 'trend_strength']
            missing_trend = sum(1 for col in trend_cols if pd.isna(row[col]))
            if missing_trend > 1:
                flag_list.append("INSUFFICIENT_TREND_DATA")
            
            # Check for missing risk data
            risk_cols = ['idio_vol', 'turnover', 'max_drawdown_12_24m']
            missing_risk = sum(1 for col in risk_cols if pd.isna(row[col]))
            if missing_risk > 1:
                flag_list.append("INSUFFICIENT_RISK_DATA")
            
            # Check for missing relative strength data
            rel_strength_cols = ['sector_rel_strength', 'sizebucket_rel_strength']
            missing_rel_strength = sum(1 for col in rel_strength_cols if pd.isna(row[col]))
            if missing_rel_strength == 2:
                flag_list.append("NO_RELATIVE_STRENGTH_DATA")
            elif missing_rel_strength == 1:
                flag_list.append("PARTIAL_RELATIVE_STRENGTH_DATA")
            
            # Check for missing 52-week high data
            high52w_cols = ['dist_to_52w_high', 'post_break_hold_score']
            missing_high52w = sum(1 for col in high52w_cols if pd.isna(row[col]))
            if missing_high52w == 2:
                flag_list.append("NO_52WEEK_HIGH_DATA")
            
            # Check for extreme values that might indicate data quality issues
            if pd.notna(row.get('vol_adj_momentum_composite')):
                if abs(row['vol_adj_momentum_composite']) > 3.0:
                    flag_list.append("EXTREME_MOMENTUM_VALUE")
            
            if pd.notna(row.get('idio_vol')):
                if row['idio_vol'] > 1.5:
                    flag_list.append("EXTREME_VOLATILITY")
            
            if pd.notna(row.get('turnover')):
                if row['turnover'] > 0.3:
                    flag_list.append("EXTREME_TURNOVER")
            
            flags.append("|".join(flag_list) if flag_list else "OK")
        
        return flags
    
    def _apply_outlier_caps(self, results: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier caps to prevent extreme values."""
        results = results.copy()
        
        # Cap momentum indicators
        momentum_cols = [f'mom_{p}' for p in ['12m', '6m', '3m', '1m']] + ['vol_adj_momentum_composite']
        for col in momentum_cols:
            if col in results.columns:
                results[col] = results[col].clip(
                    lower=self.outlier_caps['annual_return_lower'],
                    upper=self.outlier_caps['annual_return_upper']
                )
        
        # Cap volatility indicators
        vol_cols = ['idio_vol']
        for col in vol_cols:
            if col in results.columns:
                results[col] = results[col].clip(
                    upper=self.outlier_caps['volatility_upper']
                )
        
        # Cap turnover
        if 'turnover' in results.columns:
            results['turnover'] = results['turnover'].clip(
                upper=self.outlier_caps['turnover_upper']
            )
        
        return results
    
    def _save_to_database(self, results: pd.DataFrame, table_name: str) -> None:
        """
        Save results to database with error handling.
        
        Args:
            results (pd.DataFrame): Results to save
            table_name (str): Target table name (schema.table format)
            
        Raises:
            PriceMomentumCalculationError: If database save fails
        """
        try:
            # Validate results before saving
            if results.empty:
                logger.warning("No results to save to database")
                return
            
            # Split schema and table name
            schema_name, table_only = table_name.split('.') if '.' in table_name else (None, table_name)
            full_table_name = f"{schema_name}.{table_only}" if schema_name else table_only

            # Ensure proper data types
            results = self._ensure_proper_dtypes(results)

            # Determine scope for deduplication prior to append
            if 'as_of_date' not in results.columns:
                raise PriceMomentumCalculationError("Results must include 'as_of_date' column for persistence")

            as_of_dates = results['as_of_date'].dropna().unique()
            if len(as_of_dates) != 1:
                raise PriceMomentumCalculationError("Results should contain exactly one as_of_date when saving")

            as_of_date = pd.to_datetime(as_of_dates[0]).date()
            symbols = results['symbol'].dropna().unique().tolist()

            with self.engine.begin() as conn:
                if symbols:
                    placeholders = ", ".join([f":symbol_{idx}" for idx in range(len(symbols))])
                    delete_query = text(
                        f"""
                        DELETE FROM {full_table_name}
                        WHERE as_of_date = :as_of_date
                        AND symbol IN ({placeholders})
                        """
                    )
                    params = {'as_of_date': as_of_date}
                    for idx, symbol in enumerate(symbols):
                        params[f"symbol_{idx}"] = symbol
                    try:
                        conn.execute(delete_query, params)
                    except Exception as exc:
                        logger.warning(f"Skipping pre-delete for {full_table_name}: {exc}")

                results.to_sql(
                    table_only,
                    conn,
                    schema=schema_name,
                    if_exists='append',
                    index=False,
                    method='multi'
                )

            logger.info(f"Successfully saved {len(results)} rows to {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            raise PriceMomentumCalculationError(f"Database save failed: {e}")
    
    def _ensure_proper_dtypes(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure proper data types for database storage.
        
        Args:
            results (pd.DataFrame): Results dataframe
            
        Returns:
            pd.DataFrame: DataFrame with proper data types
        """
        results = results.copy()
        
        # Convert datetime columns
        if 'as_of_date' in results.columns:
            results['as_of_date'] = pd.to_datetime(results['as_of_date'])
        if 'last_updated' in results.columns:
            results['last_updated'] = pd.to_datetime(results['last_updated'])
        
        # Ensure numeric columns are float64
        numeric_cols = [
            'mom_12m', 'mom_6m', 'mom_3m', 'mom_1m',
            'vol_adj_momentum_composite', 'momentum_persistence',
            'is_above_200dma', 'is_above_50dma', 'is_gc_50_200', 'trend_strength',
            'dist_to_52w_high', 'post_break_hold_score',
            'sector_rel_strength', 'sizebucket_rel_strength',
            'idio_vol', 'turnover', 'max_drawdown_12_24m', 'earnings_surprise_vol'
        ]
        
        for col in numeric_cols:
            if col in results.columns:
                results[col] = pd.to_numeric(results[col], errors='coerce').astype('float64')
        
        return results
    
    def validate_input_data(self, price_data: pd.DataFrame) -> bool:
        """
        Validate input price data for required columns and basic quality.
        
        Args:
            price_data (pd.DataFrame): Input price data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'daily_return']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in price_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for empty data
        if price_data.empty:
            logger.error("Price data is empty")
            return False
        
        # Check for required data quality
        total_rows = len(price_data)
        null_counts = price_data[required_columns].isnull().sum()
        
        for col, null_count in null_counts.items():
            if null_count / total_rows > 0.1:  # More than 10% null values
                logger.warning(f"Column {col} has {null_count/total_rows:.1%} null values")
        
        # Check for reasonable value ranges
        if (price_data['close'] <= 0).any():
            logger.warning("Found non-positive close prices")
        
        if (price_data['volume'] < 0).any():
            logger.warning("Found negative volume values")
        
        logger.info(f"Input validation passed for {total_rows} rows across {price_data['symbol'].nunique()} symbols")
        return True


def main():
    """Main function for testing the calculator."""
    try:
        calc = PriceMomentumCalculator()
        
        # Test with a few symbols
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        target_date = '2025-09-10'
        
        results = calc.calculate_all_features(
            target_date=target_date,
            symbols=test_symbols,
            save_to_db=False
        )
        
        print(f"Calculated features for {len(results)} symbols")
        print(results.head())
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
