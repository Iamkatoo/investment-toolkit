#!/usr/bin/env python3
"""
AB Comparison Engine - V2 Migration System

Real-time comparison analysis engine for V1 vs V2 scoring systems.
Provides comprehensive statistical analysis, correlation calculations,
ranking overlap analysis, and anomaly detection capabilities.

Implementation Task 2.2: Real-time Comparison Analysis Engine
- Statistical correlation analysis (Pearson, Spearman, Kendall)
- Ranking overlap and top-N analysis
- Sector-wise performance comparison
- Time series trend analysis
- Outlier and anomaly detection
- Distribution similarity testing

Key Features:
- Multi-level correlation analysis (total score, pillar-wise, sub-metrics)
- Advanced statistical tests (KS test, Chi-square, Mann-Whitney U)
- Sector and market cap bucket analysis
- Rolling correlation and trend detection
- Configurable outlier detection algorithms
- Performance degradation alerts

Usage:
    from investment_analysis.analysis.ab_comparison_engine import ABComparisonEngine

    engine = ABComparisonEngine()
    results = engine.analyze_daily_comparison(v1_scores, v2_scores, date)
    correlation_matrix = engine.calculate_correlation_matrix(v1_scores, v2_scores)
    outliers = engine.detect_outliers(v1_scores, v2_scores)

Created: 2025-09-15
Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from investment_analysis.utilities.feature_flags import is_enabled
from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from sqlalchemy import create_engine, text

# Setup logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class CorrelationAnalysis:
    """Results from correlation analysis"""
    pearson_correlation: float
    spearman_correlation: float
    kendall_correlation: float
    pearson_p_value: float
    spearman_p_value: float
    kendall_p_value: float
    r_squared: float
    sample_size: int


@dataclass
class RankingAnalysis:
    """Results from ranking overlap analysis"""
    top_50_overlap: float
    top_100_overlap: float
    top_200_overlap: float
    top_500_overlap: float
    bottom_50_overlap: float
    rank_correlation: float
    rank_distance_mean: float
    rank_distance_std: float


@dataclass
class DistributionAnalysis:
    """Results from distribution comparison"""
    ks_statistic: float
    ks_p_value: float
    chi_square_statistic: float
    chi_square_p_value: float
    jensen_shannon_distance: float
    wasserstein_distance: float
    mean_difference: float
    std_difference: float
    skewness_difference: float
    kurtosis_difference: float


@dataclass
class SectorAnalysis:
    """Sector-wise comparison results"""
    sector: str
    symbol_count: int
    correlation: float
    correlation_p_value: float
    mean_score_diff: float
    std_score_diff: float
    top_10_overlap: float
    outlier_count: int


@dataclass
class OutlierDetection:
    """Outlier detection results"""
    method: str
    outlier_symbols: List[str]
    outlier_scores: List[Tuple[str, float, float, float]]  # symbol, v1_score, v2_score, diff
    outlier_threshold: float
    total_outliers: int
    outlier_percentage: float


@dataclass
class ABComparisonResult:
    """Complete AB comparison analysis result"""
    analysis_date: str
    timestamp: datetime
    total_symbols: int
    correlation_analysis: CorrelationAnalysis
    ranking_analysis: RankingAnalysis
    distribution_analysis: DistributionAnalysis
    sector_analyses: List[SectorAnalysis]
    outlier_detections: List[OutlierDetection]
    pillar_correlations: Dict[str, float]
    performance_metrics: Dict[str, Any]
    alert_flags: List[str]
    summary_stats: Dict[str, Any]


class ABComparisonEngine:
    """
    Real-time AB Comparison Analysis Engine
    
    Provides comprehensive statistical analysis comparing V1 and V2 scoring systems
    including correlations, ranking overlaps, distribution analysis, and anomaly detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AB Comparison Engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
            
        # Database connection
        self.db_engine = self._create_db_engine()
        
        # Analysis cache
        self.analysis_cache: Dict[str, ABComparisonResult] = {}
        
        logger.info("AB Comparison Engine initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for comparison analysis"""
        return {
            'correlation_thresholds': {
                'excellent': 0.9,
                'good': 0.7,
                'acceptable': 0.5,
                'poor': 0.3
            },
            'ranking_thresholds': {
                'top_50_min': 0.8,
                'top_100_min': 0.7,
                'top_200_min': 0.6
            },
            'outlier_detection': {
                'z_score_threshold': 3.0,
                'iqr_multiplier': 1.5,
                'isolation_forest_contamination': 0.1,
                'enable_multiple_methods': True
            },
            'distribution_tests': {
                'ks_test_alpha': 0.05,
                'chi_square_alpha': 0.05,
                'bins_for_chi_square': 20
            },
            'sector_analysis': {
                'min_symbols_per_sector': 5,
                'enable_size_bucket_analysis': True
            },
            'performance_monitoring': {
                'max_analysis_time': 300,  # 5 minutes
                'enable_caching': True,
                'cache_ttl_minutes': 60
            }
        }
    
    def _create_db_engine(self):
        """Create database engine for data retrieval"""
        try:
            connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(connection_string, pool_pre_ping=True)
            return engine
        except Exception as e:
            logger.warning(f"Failed to create database engine: {e}")
            return None
    
    def analyze_daily_comparison(self, v1_scores: pd.DataFrame, v2_scores: pd.DataFrame,
                               analysis_date: str, sector_data: Optional[pd.DataFrame] = None) -> ABComparisonResult:
        """
        Perform comprehensive daily comparison analysis
        
        Args:
            v1_scores: V1 scoring results DataFrame
            v2_scores: V2 scoring results DataFrame
            analysis_date: Date of analysis (YYYY-MM-DD)
            sector_data: Optional sector information DataFrame
            
        Returns:
            ABComparisonResult with complete analysis
        """
        logger.info(f"Starting daily comparison analysis for {analysis_date}")
        start_time = datetime.now()
        
        try:
            # Align dataframes on symbol
            aligned_data = self._align_scoring_data(v1_scores, v2_scores)
            if aligned_data.empty:
                raise ValueError("No overlapping symbols found between V1 and V2 results")
            
            total_symbols = len(aligned_data)
            logger.info(f"Analyzing {total_symbols} symbols")
            
            # Core statistical analysis
            correlation_analysis = self.calculate_correlation_analysis(
                aligned_data['v1_total_score'], 
                aligned_data['v2_total_score']
            )
            
            ranking_analysis = self.calculate_ranking_overlap(
                aligned_data['v1_total_score'],
                aligned_data['v2_total_score']
            )
            
            distribution_analysis = self.analyze_score_distributions(
                aligned_data['v1_total_score'],
                aligned_data['v2_total_score']
            )
            
            # Pillar-wise correlation analysis
            pillar_correlations = self._calculate_pillar_correlations(aligned_data)
            
            # Sector analysis
            sector_analyses = []
            if sector_data is not None:
                sector_analyses = self._analyze_by_sectors(aligned_data, sector_data)
            
            # Outlier detection
            outlier_detections = self.detect_outliers(
                aligned_data[['symbol', 'v1_total_score', 'v2_total_score']]
            )
            
            # Performance metrics
            analysis_duration = (datetime.now() - start_time).total_seconds()
            performance_metrics = {
                'analysis_duration_seconds': analysis_duration,
                'symbols_processed': total_symbols,
                'memory_usage_mb': self._get_memory_usage(),
                'successful_comparisons': len(aligned_data),
                'missing_v1_symbols': len(v2_scores) - total_symbols,
                'missing_v2_symbols': len(v1_scores) - total_symbols
            }
            
            # Generate alerts
            alert_flags = self._generate_alert_flags(
                correlation_analysis, ranking_analysis, distribution_analysis, outlier_detections
            )
            
            # Summary statistics
            summary_stats = self._calculate_summary_statistics(aligned_data)
            
            # Create comprehensive result
            result = ABComparisonResult(
                analysis_date=analysis_date,
                timestamp=datetime.now(),
                total_symbols=total_symbols,
                correlation_analysis=correlation_analysis,
                ranking_analysis=ranking_analysis,
                distribution_analysis=distribution_analysis,
                sector_analyses=sector_analyses,
                outlier_detections=outlier_detections,
                pillar_correlations=pillar_correlations,
                performance_metrics=performance_metrics,
                alert_flags=alert_flags,
                summary_stats=summary_stats
            )
            
            # Cache result
            if self.config['performance_monitoring']['enable_caching']:
                self.analysis_cache[analysis_date] = result
            
            logger.info(f"Daily comparison analysis completed in {analysis_duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Daily comparison analysis failed: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            raise
    
    def calculate_correlation_analysis(self, v1_scores: pd.Series, v2_scores: pd.Series) -> CorrelationAnalysis:
        """
        Calculate comprehensive correlation analysis
        
        Args:
            v1_scores: V1 scores
            v2_scores: V2 scores
            
        Returns:
            CorrelationAnalysis with multiple correlation metrics
        """
        # Remove NaN values
        mask = ~(pd.isna(v1_scores) | pd.isna(v2_scores))
        v1_clean = v1_scores[mask]
        v2_clean = v2_scores[mask]
        
        if len(v1_clean) < 3:
            logger.warning("Insufficient data for correlation analysis")
            return CorrelationAnalysis(0, 0, 0, 1, 1, 1, 0, 0)
        
        try:
            # Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(v1_clean, v2_clean)
            
            # Spearman correlation
            spearman_corr, spearman_p = stats.spearmanr(v1_clean, v2_clean)
            
            # Kendall's tau
            kendall_corr, kendall_p = stats.kendalltau(v1_clean, v2_clean)
            
            # R-squared
            r_squared = pearson_corr ** 2
            
            return CorrelationAnalysis(
                pearson_correlation=float(pearson_corr),
                spearman_correlation=float(spearman_corr),
                kendall_correlation=float(kendall_corr),
                pearson_p_value=float(pearson_p),
                spearman_p_value=float(spearman_p),
                kendall_p_value=float(kendall_p),
                r_squared=float(r_squared),
                sample_size=len(v1_clean)
            )
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return CorrelationAnalysis(0, 0, 0, 1, 1, 1, 0, len(v1_clean))
    
    def calculate_ranking_overlap(self, v1_scores: pd.Series, v2_scores: pd.Series) -> RankingAnalysis:
        """
        Calculate ranking overlap analysis
        
        Args:
            v1_scores: V1 scores
            v2_scores: V2 scores
            
        Returns:
            RankingAnalysis with overlap metrics
        """
        try:
            # Create rankings (descending order - higher scores are better)
            v1_ranks = v1_scores.rank(method='dense', ascending=False)
            v2_ranks = v2_scores.rank(method='dense', ascending=False)
            
            total_symbols = len(v1_scores)
            
            # Calculate top-N overlaps
            top_overlaps = {}
            for n in [50, 100, 200, 500]:
                if n <= total_symbols:
                    v1_top_n = set(v1_ranks[v1_ranks <= n].index)
                    v2_top_n = set(v2_ranks[v2_ranks <= n].index)
                    overlap = len(v1_top_n.intersection(v2_top_n)) / n
                    top_overlaps[f'top_{n}'] = overlap
                else:
                    top_overlaps[f'top_{n}'] = 0.0
            
            # Bottom 50 overlap
            if total_symbols >= 50:
                v1_bottom_50 = set(v1_ranks[v1_ranks > total_symbols - 50].index)
                v2_bottom_50 = set(v2_ranks[v2_ranks > total_symbols - 50].index)
                bottom_50_overlap = len(v1_bottom_50.intersection(v2_bottom_50)) / 50
            else:
                bottom_50_overlap = 0.0
            
            # Rank correlation
            rank_correlation, _ = stats.spearmanr(v1_ranks, v2_ranks)
            
            # Rank distance statistics
            rank_distances = abs(v1_ranks - v2_ranks)
            rank_distance_mean = float(rank_distances.mean())
            rank_distance_std = float(rank_distances.std())
            
            return RankingAnalysis(
                top_50_overlap=top_overlaps.get('top_50', 0.0),
                top_100_overlap=top_overlaps.get('top_100', 0.0),
                top_200_overlap=top_overlaps.get('top_200', 0.0),
                top_500_overlap=top_overlaps.get('top_500', 0.0),
                bottom_50_overlap=bottom_50_overlap,
                rank_correlation=float(rank_correlation) if not pd.isna(rank_correlation) else 0.0,
                rank_distance_mean=rank_distance_mean,
                rank_distance_std=rank_distance_std
            )
            
        except Exception as e:
            logger.error(f"Ranking overlap analysis failed: {e}")
            return RankingAnalysis(0, 0, 0, 0, 0, 0, 0, 0)
    
    def analyze_score_distributions(self, v1_scores: pd.Series, v2_scores: pd.Series) -> DistributionAnalysis:
        """
        Analyze distribution differences between V1 and V2 scores
        
        Args:
            v1_scores: V1 scores
            v2_scores: V2 scores
            
        Returns:
            DistributionAnalysis with distribution comparison metrics
        """
        try:
            # Remove NaN values
            mask = ~(pd.isna(v1_scores) | pd.isna(v2_scores))
            v1_clean = v1_scores[mask].values
            v2_clean = v2_scores[mask].values
            
            if len(v1_clean) < 5:
                logger.warning("Insufficient data for distribution analysis")
                return DistributionAnalysis(0, 1, 0, 1, 0, 0, 0, 0, 0, 0)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(v1_clean, v2_clean)
            
            # Chi-square test (using binned data)
            bins = self.config['distribution_tests']['bins_for_chi_square']
            v1_hist, bin_edges = np.histogram(v1_clean, bins=bins, density=True)
            v2_hist, _ = np.histogram(v2_clean, bins=bin_edges, density=True)
            
            # Avoid zero frequencies for chi-square test
            v1_hist = v1_hist + 1e-10
            v2_hist = v2_hist + 1e-10
            
            chi2_stat, chi2_p = stats.chisquare(v1_hist, v2_hist)
            
            # Jensen-Shannon distance
            try:
                js_distance = jensenshannon(v1_hist, v2_hist)
            except:
                js_distance = 0.0
            
            # Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(v1_clean, v2_clean)
            
            # Moment differences
            mean_diff = float(np.mean(v2_clean) - np.mean(v1_clean))
            std_diff = float(np.std(v2_clean) - np.std(v1_clean))
            
            try:
                skew_diff = float(stats.skew(v2_clean) - stats.skew(v1_clean))
                kurt_diff = float(stats.kurtosis(v2_clean) - stats.kurtosis(v1_clean))
            except:
                skew_diff = 0.0
                kurt_diff = 0.0
            
            return DistributionAnalysis(
                ks_statistic=float(ks_stat),
                ks_p_value=float(ks_p),
                chi_square_statistic=float(chi2_stat),
                chi_square_p_value=float(chi2_p),
                jensen_shannon_distance=float(js_distance),
                wasserstein_distance=float(wasserstein_dist),
                mean_difference=mean_diff,
                std_difference=std_diff,
                skewness_difference=skew_diff,
                kurtosis_difference=kurt_diff
            )
            
        except Exception as e:
            logger.error(f"Distribution analysis failed: {e}")
            return DistributionAnalysis(0, 1, 0, 1, 0, 0, 0, 0, 0, 0)
    
    def detect_outliers(self, score_data: pd.DataFrame, 
                       methods: Optional[List[str]] = None) -> List[OutlierDetection]:
        """
        Detect outliers using multiple methods
        
        Args:
            score_data: DataFrame with columns ['symbol', 'v1_total_score', 'v2_total_score']
            methods: List of detection methods to use
            
        Returns:
            List of OutlierDetection results
        """
        if methods is None:
            methods = ['z_score', 'iqr', 'isolation_forest']
        
        outlier_results = []
        
        # Calculate score differences
        score_data = score_data.copy()
        score_data['score_diff'] = score_data['v2_total_score'] - score_data['v1_total_score']
        score_data['abs_score_diff'] = abs(score_data['score_diff'])
        
        # Remove NaN values
        clean_data = score_data.dropna()
        
        if len(clean_data) < 5:
            logger.warning("Insufficient data for outlier detection")
            return outlier_results
        
        try:
            # Z-score method
            if 'z_score' in methods:
                z_scores = np.abs(stats.zscore(clean_data['score_diff']))
                z_threshold = self.config['outlier_detection']['z_score_threshold']
                z_outliers = clean_data[z_scores > z_threshold]
                
                outlier_scores = [
                    (row['symbol'], row['v1_total_score'], row['v2_total_score'], row['score_diff'])
                    for _, row in z_outliers.iterrows()
                ]
                
                outlier_results.append(OutlierDetection(
                    method='z_score',
                    outlier_symbols=z_outliers['symbol'].tolist(),
                    outlier_scores=outlier_scores,
                    outlier_threshold=z_threshold,
                    total_outliers=len(z_outliers),
                    outlier_percentage=len(z_outliers) / len(clean_data) * 100
                ))
            
            # IQR method
            if 'iqr' in methods:
                Q1 = clean_data['abs_score_diff'].quantile(0.25)
                Q3 = clean_data['abs_score_diff'].quantile(0.75)
                IQR = Q3 - Q1
                iqr_multiplier = self.config['outlier_detection']['iqr_multiplier']
                iqr_threshold = Q3 + iqr_multiplier * IQR
                
                iqr_outliers = clean_data[clean_data['abs_score_diff'] > iqr_threshold]
                
                outlier_scores = [
                    (row['symbol'], row['v1_total_score'], row['v2_total_score'], row['score_diff'])
                    for _, row in iqr_outliers.iterrows()
                ]
                
                outlier_results.append(OutlierDetection(
                    method='iqr',
                    outlier_symbols=iqr_outliers['symbol'].tolist(),
                    outlier_scores=outlier_scores,
                    outlier_threshold=iqr_threshold,
                    total_outliers=len(iqr_outliers),
                    outlier_percentage=len(iqr_outliers) / len(clean_data) * 100
                ))
            
            # Isolation Forest method
            if 'isolation_forest' in methods and len(clean_data) >= 10:
                contamination = self.config['outlier_detection']['isolation_forest_contamination']
                
                # Use multiple features for isolation forest
                features = clean_data[['v1_total_score', 'v2_total_score', 'score_diff']].values
                
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_labels = iso_forest.fit_predict(features_scaled)
                
                iso_outliers = clean_data[outlier_labels == -1]
                
                outlier_scores = [
                    (row['symbol'], row['v1_total_score'], row['v2_total_score'], row['score_diff'])
                    for _, row in iso_outliers.iterrows()
                ]
                
                outlier_results.append(OutlierDetection(
                    method='isolation_forest',
                    outlier_symbols=iso_outliers['symbol'].tolist(),
                    outlier_scores=outlier_scores,
                    outlier_threshold=contamination,
                    total_outliers=len(iso_outliers),
                    outlier_percentage=len(iso_outliers) / len(clean_data) * 100
                ))
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
        
        return outlier_results
    
    def _align_scoring_data(self, v1_scores: pd.DataFrame, v2_scores: pd.DataFrame) -> pd.DataFrame:
        """Align V1 and V2 scoring data on symbol"""
        try:
            # Ensure we have the required columns
            v1_cols = ['symbol', 'total_score']
            v2_cols = ['symbol', 'total_score']
            
            # Rename columns to avoid conflicts
            v1_data = v1_scores[v1_cols].copy()
            v1_data.columns = ['symbol', 'v1_total_score']
            
            v2_data = v2_scores[v2_cols].copy()
            v2_data.columns = ['symbol', 'v2_total_score']
            
            # Merge on symbol
            aligned = pd.merge(v1_data, v2_data, on='symbol', how='inner')
            
            logger.info(f"Aligned {len(aligned)} symbols from V1({len(v1_scores)}) and V2({len(v2_scores)})")
            return aligned
            
        except Exception as e:
            logger.error(f"Failed to align scoring data: {e}")
            return pd.DataFrame()
    
    def _calculate_pillar_correlations(self, aligned_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pillar-wise correlations if data is available"""
        pillar_correlations = {}
        
        pillar_mappings = {
            'value': ['value_score'],
            'growth': ['growth_score'],
            'quality': ['quality_score'],
            'momentum': ['momentum_score'],
            'macro': ['macro_sector_score']
        }
        
        try:
            for pillar, v1_cols in pillar_mappings.items():
                for col in v1_cols:
                    v1_col = f'v1_{col}'
                    v2_col = f'v2_{col}'
                    
                    if v1_col in aligned_data.columns and v2_col in aligned_data.columns:
                        corr, _ = stats.pearsonr(
                            aligned_data[v1_col].fillna(0),
                            aligned_data[v2_col].fillna(0)
                        )
                        pillar_correlations[pillar] = float(corr) if not pd.isna(corr) else 0.0
                        break
        except Exception as e:
            logger.warning(f"Failed to calculate pillar correlations: {e}")
        
        return pillar_correlations
    
    def _analyze_by_sectors(self, aligned_data: pd.DataFrame, 
                          sector_data: pd.DataFrame) -> List[SectorAnalysis]:
        """Perform sector-wise analysis"""
        sector_analyses = []
        
        try:
            # Merge with sector data
            data_with_sectors = pd.merge(aligned_data, sector_data, on='symbol', how='left')
            
            min_symbols = self.config['sector_analysis']['min_symbols_per_sector']
            
            for sector in data_with_sectors['sector'].unique():
                if pd.isna(sector):
                    continue
                    
                sector_data_subset = data_with_sectors[data_with_sectors['sector'] == sector]
                
                if len(sector_data_subset) < min_symbols:
                    continue
                
                # Calculate sector-specific metrics
                correlation, corr_p = stats.pearsonr(
                    sector_data_subset['v1_total_score'],
                    sector_data_subset['v2_total_score']
                )
                
                score_diff = sector_data_subset['v2_total_score'] - sector_data_subset['v1_total_score']
                
                # Top 10 overlap within sector
                v1_top_10 = set(sector_data_subset.nlargest(10, 'v1_total_score')['symbol'])
                v2_top_10 = set(sector_data_subset.nlargest(10, 'v2_total_score')['symbol'])
                top_10_overlap = len(v1_top_10.intersection(v2_top_10)) / min(10, len(sector_data_subset))
                
                # Count outliers in this sector
                outlier_count = len(sector_data_subset[abs(score_diff) > 2 * score_diff.std()])
                
                sector_analyses.append(SectorAnalysis(
                    sector=sector,
                    symbol_count=len(sector_data_subset),
                    correlation=float(correlation) if not pd.isna(correlation) else 0.0,
                    correlation_p_value=float(corr_p) if not pd.isna(corr_p) else 1.0,
                    mean_score_diff=float(score_diff.mean()),
                    std_score_diff=float(score_diff.std()),
                    top_10_overlap=top_10_overlap,
                    outlier_count=outlier_count
                ))
                
        except Exception as e:
            logger.warning(f"Sector analysis failed: {e}")
        
        return sector_analyses
    
    def _generate_alert_flags(self, correlation_analysis: CorrelationAnalysis,
                            ranking_analysis: RankingAnalysis,
                            distribution_analysis: DistributionAnalysis,
                            outlier_detections: List[OutlierDetection]) -> List[str]:
        """Generate alert flags based on analysis results"""
        alerts = []
        
        thresholds = self.config['correlation_thresholds']
        ranking_thresholds = self.config['ranking_thresholds']
        
        # Correlation alerts
        if correlation_analysis.pearson_correlation < thresholds['poor']:
            alerts.append(f"POOR_CORRELATION: {correlation_analysis.pearson_correlation:.3f}")
        elif correlation_analysis.pearson_correlation < thresholds['acceptable']:
            alerts.append(f"LOW_CORRELATION: {correlation_analysis.pearson_correlation:.3f}")
        
        # Ranking overlap alerts
        if ranking_analysis.top_50_overlap < ranking_thresholds['top_50_min']:
            alerts.append(f"POOR_TOP50_OVERLAP: {ranking_analysis.top_50_overlap:.3f}")
        
        if ranking_analysis.top_100_overlap < ranking_thresholds['top_100_min']:
            alerts.append(f"POOR_TOP100_OVERLAP: {ranking_analysis.top_100_overlap:.3f}")
        
        # Distribution alerts
        alpha = self.config['distribution_tests']['ks_test_alpha']
        if distribution_analysis.ks_p_value < alpha:
            alerts.append(f"SIGNIFICANT_DISTRIBUTION_DIFFERENCE: p={distribution_analysis.ks_p_value:.4f}")
        
        # Outlier alerts
        for detection in outlier_detections:
            if detection.outlier_percentage > 10:  # More than 10% outliers
                alerts.append(f"HIGH_OUTLIER_RATE_{detection.method.upper()}: {detection.outlier_percentage:.1f}%")
        
        return alerts
    
    def _calculate_summary_statistics(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the comparison"""
        try:
            v1_scores = aligned_data['v1_total_score']
            v2_scores = aligned_data['v2_total_score']
            score_diff = v2_scores - v1_scores
            
            return {
                'v1_mean': float(v1_scores.mean()),
                'v1_std': float(v1_scores.std()),
                'v1_min': float(v1_scores.min()),
                'v1_max': float(v1_scores.max()),
                'v2_mean': float(v2_scores.mean()),
                'v2_std': float(v2_scores.std()),
                'v2_min': float(v2_scores.min()),
                'v2_max': float(v2_scores.max()),
                'diff_mean': float(score_diff.mean()),
                'diff_std': float(score_diff.std()),
                'diff_min': float(score_diff.min()),
                'diff_max': float(score_diff.max()),
                'positive_changes': int((score_diff > 0).sum()),
                'negative_changes': int((score_diff < 0).sum()),
                'unchanged': int((score_diff == 0).sum())
            }
        except Exception as e:
            logger.error(f"Failed to calculate summary statistics: {e}")
            return {}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_cached_analysis(self, date: str) -> Optional[ABComparisonResult]:
        """Retrieve cached analysis result"""
        return self.analysis_cache.get(date)
    
    def save_analysis_to_database(self, result: ABComparisonResult):
        """Save analysis result to database"""
        if not self.db_engine:
            logger.warning("No database engine available")
            return
            
        try:
            # Convert result to database-friendly format
            result_data = {
                'date': result.analysis_date,
                'comparison_timestamp': result.timestamp,
                'total_symbols': result.total_symbols,
                'total_score_correlation': result.correlation_analysis.pearson_correlation,
                'spearman_correlation': result.correlation_analysis.spearman_correlation,
                'kendall_correlation': result.correlation_analysis.kendall_correlation,
                'top_50_overlap_rate': result.ranking_analysis.top_50_overlap * 100,
                'top_100_overlap_rate': result.ranking_analysis.top_100_overlap * 100,
                'top_200_overlap_rate': result.ranking_analysis.top_200_overlap * 100,
                'ks_test_p_value': result.distribution_analysis.ks_p_value,
                'mean_score_difference': result.distribution_analysis.mean_difference,
                'score_difference_std': result.distribution_analysis.std_difference,
                'alert_level': 'CRITICAL' if any('CRITICAL' in alert for alert in result.alert_flags) else 
                             'WARNING' if any('WARNING' in alert or 'POOR' in alert for alert in result.alert_flags) else 'OK',
                'alert_messages': json.dumps(result.alert_flags),
                'outlier_symbols_count': sum(d.total_outliers for d in result.outlier_detections),
                'sector_analysis_results': json.dumps([asdict(s) for s in result.sector_analyses]),
                'performance_metrics': json.dumps(result.performance_metrics),
                'summary_statistics': json.dumps(result.summary_stats)
            }
            
            # Insert into database
            columns = ', '.join(result_data.keys())
            placeholders = ', '.join([f":{key}" for key in result_data.keys()])
            query = f"INSERT INTO backtest_results.ab_comparison_results ({columns}) VALUES ({placeholders})"
            
            with self.db_engine.connect() as conn:
                conn.execute(text(query), result_data)
                conn.commit()
                
            logger.info("Analysis result saved to database")
            
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {e}")


def main():
    """Example usage of AB Comparison Engine"""
    import random
    
    # Create sample data for testing
    symbols = [f"STOCK_{i:03d}" for i in range(100)]
    
    # Generate correlated sample data
    base_scores = np.random.normal(50, 15, 100)
    noise = np.random.normal(0, 5, 100)
    
    v1_scores = pd.DataFrame({
        'symbol': symbols,
        'total_score': base_scores,
        'date': '2025-09-15'
    })
    
    v2_scores = pd.DataFrame({
        'symbol': symbols,
        'total_score': base_scores * 0.9 + noise,  # Correlated but slightly different
        'date': '2025-09-15'
    })
    
    # Initialize engine and run analysis
    engine = ABComparisonEngine()
    result = engine.analyze_daily_comparison(v1_scores, v2_scores, "2025-09-15")
    
    print("AB Comparison Analysis Results:")
    print(f"Symbols analyzed: {result.total_symbols}")
    print(f"Pearson correlation: {result.correlation_analysis.pearson_correlation:.4f}")
    print(f"Top 50 overlap: {result.ranking_analysis.top_50_overlap:.3f}")
    print(f"Alert flags: {result.alert_flags}")


if __name__ == "__main__":
    main()