#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test Suite for Task 5: Normalization & Pillar Scoring

This test suite validates the complete Task 5 implementation:
1. Cross-sectional and time-series normalization
2. Pillar score calculation with weight redistribution
3. Soft caps and hard gates application
4. Macro adjustments and final score calculation
5. End-to-end batch processing

Run with: python src/scoring/test_task5_integration.py
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from investment_toolkit.scoring.normalization_engine import NormalizationEngine
from investment_toolkit.scoring.pillar_scoring import PillarScoringEngine
from investment_toolkit.scoring.schema_parser import ScoreSchemaParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Task5IntegrationTester:
    """
    Integration test suite for Task 5 implementation.
    """
    
    def __init__(self):
        """Initialize test suite components"""
        self.schema_parser = ScoreSchemaParser()
        self.normalization_engine = NormalizationEngine()
        self.pillar_engine = PillarScoringEngine()
        
        # Test data
        self.sample_data = self._create_sample_data()
        self.sample_indicators = self._create_sample_indicators()
        self.sample_macro = self._create_sample_macro_data()
        
        logger.info("Task 5 Integration Tester initialized")

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample stock data for testing"""
        np.random.seed(42)  # For reproducible tests
        
        n_stocks = 100
        sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer'] * 20
        
        data = {
            'symbol': [f'TEST{i:03d}' for i in range(n_stocks)],
            'as_of_date': pd.to_datetime('2025-09-11'),
            'gics_sector': sectors,
            'market_cap': np.random.lognormal(mean=9, sigma=1.5, size=n_stocks) * 1e6,  # $1M to $100B range
            
            # Raw indicator values (before normalization)
            'ttm_eps_yoy': np.random.normal(0.15, 0.25, n_stocks),  # 15% ± 25% growth
            'ttm_revenue_yoy': np.random.normal(0.12, 0.20, n_stocks),  # 12% ± 20% growth
            'roic_minus_wacc': np.random.normal(0.05, 0.10, n_stocks),  # 5% ± 10%
            'value_composite': np.random.uniform(0.2, 0.9, n_stocks),  # Already normalized-like
            'vol_adj_momentum_composite': np.random.normal(0.5, 0.3, n_stocks),
            'idio_vol': np.random.lognormal(mean=-2, sigma=0.5, size=n_stocks),  # 0.1 to 1.0 range
            
            # Quality indicators
            'operating_accruals': np.random.normal(0.05, 0.08, n_stocks),
            'cfo_to_net_income': np.random.normal(1.2, 0.4, n_stocks),
            'roe': np.random.normal(0.12, 0.15, n_stocks),  # Some will be negative
            'pe_ratio': np.random.lognormal(mean=3, sigma=0.8, size=n_stocks),  # Mix of reasonable and extreme
            
            # Financial health indicators
            'net_debt_to_ebitda': np.random.lognormal(mean=1, sigma=1, size=n_stocks),
            'interest_coverage_ttm': np.random.lognormal(mean=1.5, sigma=1, size=n_stocks),
            'operating_cash_flow': np.random.normal(100, 200, n_stocks),  # Mix of positive/negative
            'free_cash_flow': np.random.normal(80, 150, n_stocks),
        }
        
        df = pd.DataFrame(data)
        
        # Add some extreme/problem cases for testing flags
        # Negative ROE cases
        df.loc[df.index[:5], 'roe'] = np.random.uniform(-0.2, -0.05, 5)
        
        # Extreme PE cases
        df.loc[df.index[10:12], 'pe_ratio'] = np.random.uniform(120, 200, 2)  # Very high PE
        df.loc[df.index[12:14], 'pe_ratio'] = np.random.uniform(-50, -10, 2)  # Negative PE
        
        # High volatility cases
        df.loc[df.index[15:18], 'idio_vol'] = np.random.uniform(0.9, 1.5, 3)  # Very high vol
        
        # Financial distress cases
        df.loc[df.index[20:22], 'net_debt_to_ebitda'] = np.random.uniform(9, 15, 2)
        df.loc[df.index[20:22], 'interest_coverage_ttm'] = np.random.uniform(0.5, 1.2, 2)
        
        # High accruals cases
        df.loc[df.index[25:27], 'operating_accruals'] = np.random.uniform(0.12, 0.20, 2)
        
        return df

    def _create_sample_indicators(self) -> Dict[str, str]:
        """Create indicator name to column mapping"""
        return {
            'ttm_eps_yoy': 'ttm_eps_yoy',
            'ttm_revenue_yoy': 'ttm_revenue_yoy', 
            'roic_minus_wacc': 'roic_minus_wacc',
            'value_composite': 'value_composite',
            'vol_adj_momentum_composite': 'vol_adj_momentum_composite',
            'idiosyncratic_volatility': 'idio_vol',
            'operating_accruals': 'operating_accruals',
            'cfo_to_net_income': 'cfo_to_net_income'
        }

    def _create_sample_macro_data(self) -> Dict[str, float]:
        """Create sample macro indicator data"""
        return {
            'ism_manufacturing': 51.5,    # Neutral
            'high_yield_oas': 0.065,      # Normal
            'vix_close': 22.5             # Normal
        }

    def test_schema_parser(self) -> bool:
        """Test schema parser functionality"""
        logger.info("Testing schema parser...")
        
        try:
            # Test basic schema loading
            assert self.schema_parser.schema is not None
            assert len(self.schema_parser.pillars) == 5
            
            # Test pillar config retrieval
            value_config = self.schema_parser.get_pillar_config('value')
            assert value_config['total_weight'] == 20
            assert 'sub_indicators' in value_config
            
            # Test indicator config retrieval
            ttm_eps_config = self.schema_parser.get_indicator_config('ttm_eps_yoy')
            assert 'weight' in ttm_eps_config
            assert 'direction' in ttm_eps_config
            
            # Test normalization config
            norm_config = self.schema_parser.get_normalization_config('ttm_eps_yoy')
            assert norm_config is not None
            
            logger.info("✅ Schema parser tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema parser test failed: {e}")
            return False

    def test_normalization_engine(self) -> bool:
        """Test normalization engine functionality"""
        logger.info("Testing normalization engine...")
        
        try:
            # Test cross-sectional normalization
            df_cs = self.normalization_engine.cross_sectional_normalize(
                self.sample_data, 'ttm_eps_yoy'
            )
            assert 'ttm_eps_yoy_cs_norm' in df_cs.columns
            assert df_cs['ttm_eps_yoy_cs_norm'].notna().sum() > 50  # Most should be normalized
            
            # Check that normalized values are in [0, 1] range
            normalized_values = df_cs['ttm_eps_yoy_cs_norm'].dropna()
            assert normalized_values.min() >= 0
            assert normalized_values.max() <= 1
            
            # Test time-series normalization (limited test due to single date)
            df_ts = self.normalization_engine.time_series_normalize(
                self.sample_data, 'roic_minus_wacc'
            )
            assert 'roic_minus_wacc_ts_norm' in df_ts.columns
            
            # Test outlier detection
            df_outliers = self.normalization_engine.detect_outliers(
                self.sample_data, 'pe_ratio'
            )
            assert 'pe_ratio_outlier_flag' in df_outliers.columns
            assert df_outliers['pe_ratio_outlier_flag'].sum() > 0  # Should detect some outliers
            
            # Test winsorization
            winsorized = self.normalization_engine.apply_winsorization(
                self.sample_data['pe_ratio']
            )
            assert len(winsorized) == len(self.sample_data)
            assert winsorized.max() < self.sample_data['pe_ratio'].max()  # Should reduce extremes
            
            # Test full indicator normalization
            df_normalized = self.normalization_engine.normalize_indicator(
                self.sample_data, 'ttm_eps_yoy', 'ttm_eps_yoy'
            )
            assert 'ttm_eps_yoy_normalized' in df_normalized.columns
            
            logger.info("✅ Normalization engine tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Normalization engine test failed: {e}")
            return False

    def test_pillar_scoring_engine(self) -> bool:
        """Test pillar scoring engine functionality"""
        logger.info("Testing pillar scoring engine...")
        
        try:
            # Create normalized indicator values for one stock
            test_indicators = {
                'ttm_eps_yoy': 0.7,                # Good growth
                'ttm_revenue_yoy': 0.6,             # Decent growth
                'roic_minus_wacc': 0.8,             # Strong value creation
                'value_composite': 0.3,             # Undervalued (low is better, but we handle direction in engine)
                'vol_adj_momentum_composite': 0.65, # Good momentum
                'idiosyncratic_volatility': 0.4,    # Moderate risk
                'operating_accruals': 0.3,          # Reasonable quality
                'cfo_to_net_income': 0.7            # Good cash quality
            }
            
            # Test individual pillar calculation
            growth_score = self.pillar_engine.calculate_pillar_score('growth', test_indicators)
            assert 0 <= growth_score <= 20
            assert growth_score > 0  # Should have some score with good indicators
            
            # Test detailed pillar score
            growth_detail = self.pillar_engine.calculate_pillar_score(
                'growth', test_indicators, return_details=True
            )
            assert hasattr(growth_detail, 'raw_score')
            assert hasattr(growth_detail, 'adjusted_score')
            assert growth_detail.used_indicators > 0
            
            # Test all pillars
            all_pillars = {}
            for pillar in ['value', 'growth', 'quality', 'momentum', 'risk']:
                score = self.pillar_engine.calculate_pillar_score(pillar, test_indicators)
                all_pillars[pillar] = score
                assert 0 <= score <= (10 if pillar == 'risk' else 20)
            
            # Test soft caps
            problem_indicators = test_indicators.copy()
            problem_indicators.update({
                'roe': -0.1,                    # Negative ROE
                'operating_accruals': 0.15,     # High accruals
                'pe_ratio': 150                 # Extreme PE
            })
            
            soft_capped = self.pillar_engine.apply_soft_caps(all_pillars, problem_indicators)
            assert soft_capped['growth'] <= all_pillars['growth']  # Should be capped due to negative ROE
            
            # Test hard gates
            distress_indicators = problem_indicators.copy()
            distress_indicators.update({
                'net_debt_to_ebitda': 10,
                'interest_coverage_ttm': 1.0,
                'operating_cash_flow': -50,
                'free_cash_flow': -30
            })
            
            hard_gated, red_flags = self.pillar_engine.apply_hard_gates(
                all_pillars, distress_indicators
            )
            assert red_flags.has_critical_flags or red_flags.has_high_flags
            
            # Test macro adjustments
            macro_adjusted, macro_result = self.pillar_engine.apply_macro_adjustments(
                all_pillars, self.sample_macro
            )
            assert len(macro_result.adjustments) == 5  # All 5 pillars
            
            # Test final score calculation
            final_score = self.pillar_engine.calculate_final_score(test_indicators)
            assert 0 <= final_score <= 100 or np.isnan(final_score)
            
            # Test final score with breakdown
            breakdown = self.pillar_engine.calculate_final_score(
                test_indicators, return_breakdown=True
            )
            assert 'final_score' in breakdown
            assert 'pillar_scores' in breakdown
            assert 'red_flags' in breakdown
            
            logger.info("✅ Pillar scoring engine tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pillar scoring engine test failed: {e}")
            return False

    def test_batch_processing(self) -> bool:
        """Test end-to-end batch processing"""
        logger.info("Testing batch processing...")
        
        try:
            # Step 1: Normalize indicators
            df_processed = self.sample_data.copy()
            
            for indicator_name, column_name in self.sample_indicators.items():
                df_processed = self.normalization_engine.normalize_indicator(
                    df_processed, indicator_name, column_name
                )
            
            # Step 2: Prepare indicator mapping for scoring
            normalized_columns = {}
            for indicator_name, column_name in self.sample_indicators.items():
                normalized_col = f"{column_name}_normalized"
                if normalized_col in df_processed.columns:
                    normalized_columns[indicator_name] = normalized_col
            
            # Step 3: Calculate batch scores
            macro_columns = {
                'ism_manufacturing': 'ism_manufacturing',
                'high_yield_oas': 'high_yield_oas', 
                'vix_close': 'vix_close'
            }
            
            # Add macro data to dataframe
            for macro_name, value in self.sample_macro.items():
                df_processed[macro_name] = value
            
            # Calculate scores
            scored_df = self.pillar_engine.batch_calculate_scores(
                df_processed,
                normalized_columns,
                macro_data_columns=macro_columns
            )
            
            # Validate results
            assert 'final_score' in scored_df.columns
            assert 'value_score' in scored_df.columns
            assert 'growth_score' in scored_df.columns
            assert 'quality_score' in scored_df.columns
            assert 'momentum_score' in scored_df.columns
            assert 'risk_score' in scored_df.columns
            assert 'red_flag_status' in scored_df.columns
            
            # Check score distributions
            valid_scores = scored_df['final_score'].dropna()
            assert len(valid_scores) > 0
            assert valid_scores.min() >= 0
            assert valid_scores.max() <= 100
            
            # Check that we have some red flags detected
            red_flag_counts = scored_df['red_flag_status'].value_counts()
            assert 'none' in red_flag_counts  # Most should be fine
            
            # Check pillar score ranges
            for pillar_col in ['value_score', 'growth_score', 'quality_score', 'momentum_score']:
                valid_pillar = scored_df[pillar_col].dropna()
                if len(valid_pillar) > 0:
                    assert valid_pillar.min() >= 0
                    assert valid_pillar.max() <= 20
            
            # Risk pillar should be 0-10
            valid_risk = scored_df['risk_score'].dropna()
            if len(valid_risk) > 0:
                assert valid_risk.min() >= 0
                assert valid_risk.max() <= 10
            
            logger.info("✅ Batch processing tests passed")
            logger.info(f"   Processed {len(scored_df)} stocks")
            logger.info(f"   Valid scores: {len(valid_scores)}/{len(scored_df)}")
            logger.info(f"   Score range: [{valid_scores.min():.1f}, {valid_scores.max():.1f}]")
            logger.info(f"   Red flag distribution: {dict(red_flag_counts)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing test failed: {e}")
            return False

    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling"""
        logger.info("Testing edge cases...")
        
        try:
            # Test with all missing indicators
            empty_indicators = {}
            score = self.pillar_engine.calculate_pillar_score('growth', empty_indicators)
            assert score == 0.0
            
            # Test with some missing indicators
            partial_indicators = {'ttm_eps_yoy': 0.7}
            score = self.pillar_engine.calculate_pillar_score('growth', partial_indicators)
            assert score > 0  # Should redistribute weights
            
            # Test normalization with insufficient data
            small_df = self.sample_data.head(3)
            normalized = self.normalization_engine.cross_sectional_normalize(
                small_df, 'ttm_eps_yoy', min_observations=5
            )
            # Should handle gracefully
            
            # Test with extreme outliers
            extreme_data = self.sample_data.copy()
            extreme_data.loc[0, 'pe_ratio'] = 10000  # Extreme outlier
            winsorized = self.normalization_engine.apply_winsorization(extreme_data['pe_ratio'])
            assert winsorized.iloc[0] < 10000  # Should be winsorized
            
            logger.info("✅ Edge case tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Edge case test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        logger.info("=" * 60)
        logger.info("STARTING TASK 5 INTEGRATION TESTS")
        logger.info("=" * 60)
        
        tests = [
            ('Schema Parser', self.test_schema_parser),
            ('Normalization Engine', self.test_normalization_engine),
            ('Pillar Scoring Engine', self.test_pillar_scoring_engine),
            ('Batch Processing', self.test_batch_processing),
            ('Edge Cases', self.test_edge_cases)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} tests ---")
            result = test_func()
            results.append((test_name, result))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1
        
        overall_result = passed == len(tests)
        logger.info(f"\nOVERALL: {passed}/{len(tests)} tests passed")
        
        if overall_result:
            logger.info("🎉 ALL TASK 5 INTEGRATION TESTS PASSED! 🎉")
        else:
            logger.error("❌ SOME TESTS FAILED")
        
        return overall_result

def main():
    """Main test execution function"""
    try:
        tester = Task5IntegrationTester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()