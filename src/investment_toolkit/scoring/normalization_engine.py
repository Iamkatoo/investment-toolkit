#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normalization Engine - New Scoring System Feature Normalization

This module implements the dual normalization approach for the new 5-pillar scoring system:
1. Cross-sectional normalization: Sector×Size bucket percentile ranking
2. Time-series normalization: Self-history Z-score and trend analysis  
3. Hybrid normalization: Weighted combination of both approaches

Key Features:
- Sector×Size bucket grouping for cross-sectional comparison
- 5-year historical Z-score calculation with outlier capping
- Percentile trimming (2.5-97.5%) for outlier handling
- Winsorization for extreme value management
- Automatic method selection based on indicator characteristics
- Comprehensive data quality validation
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scoring.schema_parser import ScoreSchemaParser

logger = logging.getLogger(__name__)

class NormalizationMethod(Enum):
    """Enumeration of available normalization methods"""
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series" 
    HYBRID = "hybrid"

@dataclass
class NormalizationResult:
    """Container for normalization results"""
    normalized_value: float
    method_used: str
    metadata: Dict[str, Any]
    warnings: List[str]
    
class MarketCapBucket(Enum):
    """Market cap buckets for cross-sectional grouping"""
    LARGE_CAP = "large_cap"      # $10B+
    MID_CAP = "mid_cap"          # $2B-$10B
    SMALL_CAP = "small_cap"      # $300M-$2B
    MICRO_CAP = "micro_cap"      # <$300M

class NormalizationEngine:
    """
    Core normalization engine implementing dual normalization approach.
    
    This engine provides cross-sectional and time-series normalization
    capabilities for stock scoring indicators, with automatic method
    selection and comprehensive data quality handling.
    
    Attributes:
        schema_parser (ScoreSchemaParser): Configuration parser
        cross_sectional_config (Dict): Cross-sectional normalization settings
        time_series_config (Dict): Time-series normalization settings
    """
    
    def __init__(self, schema_path: str = "config/score_schema.yaml"):
        """
        Initialize the normalization engine.
        
        Args:
            schema_path (str): Path to the schema configuration file
        """
        self.schema_parser = ScoreSchemaParser(schema_path)
        
        # Load normalization configuration
        self.cross_sectional_config = self.schema_parser.get_normalization_config("cross_sectional")
        self.time_series_config = self.schema_parser.get_normalization_config("time_series")
        
        # Market cap bucket thresholds (in USD)
        self.market_cap_buckets = {
            MarketCapBucket.LARGE_CAP.value: 10_000_000_000,   # $10B+
            MarketCapBucket.MID_CAP.value: (2_000_000_000, 10_000_000_000),  # $2B-$10B  
            MarketCapBucket.SMALL_CAP.value: (300_000_000, 2_000_000_000),   # $300M-$2B
            MarketCapBucket.MICRO_CAP.value: 300_000_000       # <$300M
        }
        
        logger.info("Normalization engine initialized successfully")

    def assign_market_cap_bucket(self, market_cap: float) -> str:
        """
        Assign market cap bucket based on company size.
        
        Args:
            market_cap (float): Market capitalization in USD
            
        Returns:
            str: Market cap bucket name
        """
        if pd.isna(market_cap) or market_cap <= 0:
            return MarketCapBucket.MICRO_CAP.value
            
        if market_cap >= self.market_cap_buckets[MarketCapBucket.LARGE_CAP.value]:
            return MarketCapBucket.LARGE_CAP.value
        elif market_cap >= self.market_cap_buckets[MarketCapBucket.MID_CAP.value][0]:
            return MarketCapBucket.MID_CAP.value  
        elif market_cap >= self.market_cap_buckets[MarketCapBucket.SMALL_CAP.value][0]:
            return MarketCapBucket.SMALL_CAP.value
        else:
            return MarketCapBucket.MICRO_CAP.value

    def cross_sectional_normalize(self, 
                                df: pd.DataFrame,
                                indicator_column: str,
                                groupby_columns: List[str] = None,
                                trim_percentiles: Tuple[float, float] = (2.5, 97.5),
                                min_observations: int = 10) -> pd.DataFrame:
        """
        Perform cross-sectional normalization with sector×size grouping.
        
        This method calculates percentile ranks within groups defined by
        sector and market cap bucket, with outlier trimming.
        
        Args:
            df (pd.DataFrame): Input dataframe with indicator values
            indicator_column (str): Column name of indicator to normalize
            groupby_columns (List[str]): Columns to group by (default: sector, size_bucket)
            trim_percentiles (Tuple[float, float]): Percentiles for trimming outliers
            min_observations (int): Minimum observations required per group
            
        Returns:
            pd.DataFrame: DataFrame with cross-sectional normalized scores (0-1)
        """
        if groupby_columns is None:
            groupby_columns = ['gics_sector', 'market_cap_bucket']
            
        logger.debug(f"Cross-sectional normalization for {indicator_column}")
        
        result_df = df.copy()
        
        # Add market cap bucket if not present
        if 'market_cap_bucket' in groupby_columns and 'market_cap_bucket' not in result_df.columns:
            if 'market_cap' in result_df.columns:
                result_df['market_cap_bucket'] = result_df['market_cap'].apply(self.assign_market_cap_bucket)
            else:
                logger.warning("market_cap column missing, using single group normalization")
                groupby_columns = ['gics_sector'] if 'gics_sector' in groupby_columns else []
        
        # Ensure required columns exist after processing
        required_cols = groupby_columns + [indicator_column]
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        result_df[f'{indicator_column}_cs_norm'] = np.nan
        result_df[f'{indicator_column}_cs_group_size'] = np.nan
        
        if not groupby_columns:
            # Global normalization if no grouping columns available
            valid_mask = result_df[indicator_column].notna()
            values = result_df.loc[valid_mask, indicator_column]
            
            if len(values) >= min_observations:
                # Apply trimming
                lower_bound = np.percentile(values, trim_percentiles[0])
                upper_bound = np.percentile(values, trim_percentiles[1])
                trimmed_values = np.clip(values, lower_bound, upper_bound)
                
                # Calculate percentile ranks
                percentile_scores = stats.rankdata(trimmed_values, method='average') - 1
                percentile_scores = percentile_scores / (len(percentile_scores) - 1)
                
                result_df.loc[valid_mask, f'{indicator_column}_cs_norm'] = percentile_scores
                result_df.loc[valid_mask, f'{indicator_column}_cs_group_size'] = len(values)
            
            return result_df
        
        # Group-wise normalization
        for group_values, group_df in result_df.groupby(groupby_columns):
            group_idx = group_df.index
            valid_mask = group_df[indicator_column].notna()
            valid_idx = group_idx[valid_mask]
            values = group_df.loc[valid_mask, indicator_column]
            
            if len(values) < min_observations:
                # Fallback to sector-only grouping if available
                if len(groupby_columns) > 1 and 'gics_sector' in groupby_columns:
                    sector = group_values[groupby_columns.index('gics_sector')] if isinstance(group_values, tuple) else group_values
                    sector_data = result_df[result_df['gics_sector'] == sector]
                    sector_values = sector_data[sector_data[indicator_column].notna()][indicator_column]
                    
                    if len(sector_values) >= min_observations:
                        values = sector_values
                        valid_idx = sector_data[sector_data[indicator_column].notna()].index
                    else:
                        logger.warning(f"Insufficient data for group {group_values}, skipping normalization")
                        continue
                else:
                    logger.warning(f"Insufficient data for group {group_values} (n={len(values)}), skipping")
                    continue
            
            # Apply percentile trimming
            lower_bound = np.percentile(values, trim_percentiles[0])
            upper_bound = np.percentile(values, trim_percentiles[1])
            trimmed_values = np.clip(values, lower_bound, upper_bound)
            
            # Calculate percentile ranks within group
            if len(np.unique(trimmed_values)) == 1:
                # All values are the same after trimming
                percentile_scores = np.full(len(trimmed_values), 0.5)
            else:
                percentile_scores = stats.rankdata(trimmed_values, method='average') - 1
                percentile_scores = percentile_scores / (len(percentile_scores) - 1)
            
            # Store results
            result_df.loc[valid_idx, f'{indicator_column}_cs_norm'] = percentile_scores
            result_df.loc[valid_idx, f'{indicator_column}_cs_group_size'] = len(values)
            
        logger.debug(f"Cross-sectional normalization complete for {indicator_column}")
        return result_df

    def time_series_normalize(self,
                            df: pd.DataFrame, 
                            indicator_column: str,
                            symbol_column: str = 'symbol',
                            date_column: str = 'as_of_date',
                            lookback_years: int = 5,
                            min_historical_points: int = 20,
                            z_score_caps: Tuple[float, float] = (-3.0, 3.0)) -> pd.DataFrame:
        """
        Perform time-series normalization using rolling historical statistics.
        
        This method calculates Z-scores relative to each stock's 5-year history,
        with outlier capping and trend analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe with historical indicator values
            indicator_column (str): Column name of indicator to normalize
            symbol_column (str): Column name for stock symbols
            date_column (str): Column name for dates
            lookback_years (int): Years of history to use for statistics
            min_historical_points (int): Minimum historical data points required
            z_score_caps (Tuple[float, float]): Z-score capping bounds
            
        Returns:
            pd.DataFrame: DataFrame with time-series normalized scores
        """
        logger.debug(f"Time-series normalization for {indicator_column}")
        
        # Validate required columns
        required_cols = [symbol_column, date_column, indicator_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        result_df = df.copy()
        result_df[f'{indicator_column}_ts_norm'] = np.nan
        result_df[f'{indicator_column}_ts_trend'] = np.nan
        result_df[f'{indicator_column}_ts_history_points'] = np.nan
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
            result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        # Sort by symbol and date
        result_df = result_df.sort_values([symbol_column, date_column])
        
        # Process each symbol separately
        for symbol in result_df[symbol_column].unique():
            symbol_data = result_df[result_df[symbol_column] == symbol].copy()
            
            for idx, row in symbol_data.iterrows():
                current_date = row[date_column]
                current_value = row[indicator_column]
                
                if pd.isna(current_value):
                    continue
                    
                # Get historical data (excluding current date)
                start_date = current_date - pd.DateOffset(years=lookback_years)
                historical_mask = (
                    (symbol_data[date_column] >= start_date) &
                    (symbol_data[date_column] < current_date) &
                    symbol_data[indicator_column].notna()
                )
                
                historical_values = symbol_data.loc[historical_mask, indicator_column]
                
                if len(historical_values) < min_historical_points:
                    logger.debug(f"Insufficient history for {symbol} at {current_date}: {len(historical_values)} points")
                    continue
                
                # Calculate Z-score
                hist_mean = historical_values.mean()
                hist_std = historical_values.std()
                
                if hist_std == 0 or pd.isna(hist_std):
                    z_score = 0.0  # No variation in history
                else:
                    z_score = (current_value - hist_mean) / hist_std
                    
                # Apply Z-score capping
                z_score_capped = np.clip(z_score, z_score_caps[0], z_score_caps[1])
                
                # Calculate trend slope if sufficient data
                trend_slope = np.nan
                if len(historical_values) >= 8:  # Minimum for trend calculation
                    historical_dates = symbol_data.loc[historical_mask, date_column]
                    # Convert dates to numeric (days since first observation)
                    date_numeric = (historical_dates - historical_dates.min()).dt.days
                    
                    if len(np.unique(date_numeric)) > 1:  # Need variation in dates
                        try:
                            slope, _, r_value, _, _ = stats.linregress(date_numeric, historical_values)
                            # Only use trend if R² >= 0.3 (reasonably reliable trend)
                            if r_value ** 2 >= 0.3:
                                # Annualize the trend (slope per year)
                                trend_slope = slope * 365.25
                        except Exception as e:
                            logger.debug(f"Trend calculation failed for {symbol}: {e}")
                
                # Store results
                result_df.loc[idx, f'{indicator_column}_ts_norm'] = z_score_capped
                result_df.loc[idx, f'{indicator_column}_ts_trend'] = trend_slope
                result_df.loc[idx, f'{indicator_column}_ts_history_points'] = len(historical_values)
        
        logger.debug(f"Time-series normalization complete for {indicator_column}")
        return result_df

    def apply_winsorization(self, 
                           series: pd.Series, 
                           lower_percentile: float = 1.0,
                           upper_percentile: float = 99.0) -> pd.Series:
        """
        Apply winsorization to handle extreme outliers.
        
        Args:
            series (pd.Series): Input data series
            lower_percentile (float): Lower percentile for winsorization
            upper_percentile (float): Upper percentile for winsorization
            
        Returns:
            pd.Series: Winsorized series
        """
        if series.empty or series.isna().all():
            return series
            
        valid_values = series.dropna()
        if len(valid_values) == 0:
            return series
            
        lower_bound = np.percentile(valid_values, lower_percentile)
        upper_bound = np.percentile(valid_values, upper_percentile)
        
        winsorized = series.clip(lower=lower_bound, upper=upper_bound)
        
        n_clipped = ((series < lower_bound) | (series > upper_bound)).sum()
        if n_clipped > 0:
            logger.debug(f"Winsorized {n_clipped} extreme values ({n_clipped/len(series)*100:.1f}%)")
            
        return winsorized

    def detect_outliers(self, 
                       df: pd.DataFrame, 
                       indicator_column: str,
                       method: str = 'iqr',
                       iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in the data using various methods.
        
        Args:
            df (pd.DataFrame): Input dataframe
            indicator_column (str): Column to analyze for outliers
            method (str): Outlier detection method ('iqr', 'zscore')
            iqr_multiplier (float): Multiplier for IQR method
            
        Returns:
            pd.DataFrame: DataFrame with outlier flags added
        """
        result_df = df.copy()
        outlier_column = f'{indicator_column}_outlier_flag'
        result_df[outlier_column] = False
        
        values = result_df[indicator_column].dropna()
        if len(values) == 0:
            return result_df
            
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            outlier_mask = (
                (result_df[indicator_column] < lower_bound) |
                (result_df[indicator_column] > upper_bound)
            )
            
        elif method == 'zscore':
            mean_val = values.mean()
            std_val = values.std()
            z_scores = np.abs((result_df[indicator_column] - mean_val) / std_val)
            outlier_mask = z_scores > 3.0
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        result_df[outlier_column] = outlier_mask
        
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            logger.debug(f"Detected {n_outliers} outliers for {indicator_column} ({n_outliers/len(result_df)*100:.1f}%)")
        
        return result_df

    def normalize_indicator(self,
                           df: pd.DataFrame,
                           indicator_name: str, 
                           indicator_column: str,
                           method: str = "auto") -> pd.DataFrame:
        """
        Main entry point for indicator normalization.
        
        Automatically selects the appropriate normalization method based on
        indicator characteristics and schema configuration.
        
        Args:
            df (pd.DataFrame): Input dataframe
            indicator_name (str): Name of indicator (for schema lookup)
            indicator_column (str): Column name in dataframe
            method (str): Normalization method ('auto', 'cross_sectional', 'time_series', 'hybrid')
            
        Returns:
            pd.DataFrame: DataFrame with normalized indicator
        """
        logger.debug(f"Normalizing indicator: {indicator_name}")
        
        # Get normalization method from schema if auto
        if method == "auto":
            method = self._get_normalization_method(indicator_name)
        
        result_df = df.copy()
        
        # Apply outlier detection first
        result_df = self.detect_outliers(result_df, indicator_column)
        
        # Apply winsorization
        result_df[f'{indicator_column}_winsorized'] = self.apply_winsorization(
            result_df[indicator_column]
        )
        
        # Apply appropriate normalization method
        if method == NormalizationMethod.CROSS_SECTIONAL.value:
            result_df = self.cross_sectional_normalize(result_df, f'{indicator_column}_winsorized')
            result_df[f'{indicator_column}_normalized'] = result_df[f'{indicator_column}_winsorized_cs_norm']
            
        elif method == NormalizationMethod.TIME_SERIES.value:
            result_df = self.time_series_normalize(result_df, f'{indicator_column}_winsorized')
            # Convert Z-scores to 0-1 scale using sigmoid-like transformation
            ts_scores = result_df[f'{indicator_column}_winsorized_ts_norm']
            normalized_ts = 1 / (1 + np.exp(-ts_scores))  # Sigmoid transformation
            result_df[f'{indicator_column}_normalized'] = normalized_ts
            
        elif method == NormalizationMethod.HYBRID.value:
            # Apply both methods
            result_df = self.cross_sectional_normalize(result_df, f'{indicator_column}_winsorized')
            result_df = self.time_series_normalize(result_df, f'{indicator_column}_winsorized')
            
            # Get hybrid weights (default 60% cross-sectional, 40% time-series)
            cs_weight, ts_weight = self._get_hybrid_weights(indicator_name)
            
            cs_scores = result_df[f'{indicator_column}_winsorized_cs_norm'] 
            ts_scores = result_df[f'{indicator_column}_winsorized_ts_norm']
            ts_scores_norm = 1 / (1 + np.exp(-ts_scores))  # Sigmoid transformation
            
            # Combine scores with weights
            hybrid_scores = cs_weight * cs_scores + ts_weight * ts_scores_norm
            result_df[f'{indicator_column}_normalized'] = hybrid_scores
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Add metadata
        result_df[f'{indicator_column}_norm_method'] = method
        
        logger.debug(f"Indicator normalization complete: {indicator_name}")
        return result_df

    def _get_normalization_method(self, indicator_name: str) -> str:
        """
        Determine normalization method for indicator from schema.
        
        Args:
            indicator_name (str): Name of the indicator
            
        Returns:
            str: Normalization method to use
        """
        # Get indicator configuration from schema
        try:
            indicator_config = self.schema_parser.get_indicator_config(indicator_name)
            return indicator_config.get('normalization_method', 'cross_sectional')
        except:
            # Default to cross-sectional if not found in schema
            logger.debug(f"Using default cross-sectional normalization for {indicator_name}")
            return NormalizationMethod.CROSS_SECTIONAL.value

    def _get_hybrid_weights(self, indicator_name: str) -> Tuple[float, float]:
        """
        Get hybrid normalization weights for indicator.
        
        Args:
            indicator_name (str): Name of the indicator
            
        Returns:
            Tuple[float, float]: (cross_sectional_weight, time_series_weight)
        """
        try:
            indicator_config = self.schema_parser.get_indicator_config(indicator_name)
            if 'weight_ratio' in indicator_config:
                weights = indicator_config['weight_ratio']
                return weights[0], weights[1]
        except:
            pass
            
        # Default hybrid weights
        return 0.6, 0.4

    def batch_normalize_indicators(self,
                                 df: pd.DataFrame,
                                 indicator_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Normalize multiple indicators in batch.
        
        Args:
            df (pd.DataFrame): Input dataframe
            indicator_mapping (Dict[str, str]): Mapping of indicator_name -> column_name
            
        Returns:
            pd.DataFrame: DataFrame with all indicators normalized
        """
        result_df = df.copy()
        
        for indicator_name, column_name in indicator_mapping.items():
            if column_name in result_df.columns:
                try:
                    result_df = self.normalize_indicator(result_df, indicator_name, column_name)
                except Exception as e:
                    logger.error(f"Failed to normalize {indicator_name}: {e}")
                    
        return result_df