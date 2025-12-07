#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for Quality/Value Features Calculator

Tests for the QualityValueCalculator class covering:
- Quality pillar features (7 indicators)
- Value pillar features (4 indicators) 
- Edge case handling (negative values, zero denominators)
- Data quality validation
- Outlier capping functionality
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from investment_analysis.scoring.quality_value_features import QualityValueCalculator, QualityValueCalculationError


class TestQualityValueCalculator(unittest.TestCase):
    """Test cases for QualityValueCalculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock database engine
        self.mock_engine = Mock()
        self.calculator = QualityValueCalculator(engine=self.mock_engine)
        
        # Sample test data
        self.sample_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'as_of_date': ['2025-09-11'] * 4,
            
            # Income statement data
            'revenue': [365000, 198000, 280000, 96000],
            'gross_profit': [170000, 135000, 175000, 19000],
            'operating_income': [114000, 83000, 75000, 8000],
            'net_income': [97000, 72000, 61000, 5500],
            'interest_expense': [2600, 2000, 0, 640],
            'income_tax_expense': [13000, 10000, 12000, 1500],
            'eps': [6.0, 9.5, 5.8, 1.8],
            'shares_outstanding': [16000, 7500, 10500, 3000],
            'depreciation_and_amortization': [11000, 14000, 15000, 3000],
            
            # Balance sheet data
            'total_assets': [351000, 411000, 365000, 106000],
            'total_stockholders_equity': [63000, 183000, 251000, 30000],
            'long_term_debt': [95000, 58000, 13000, 9500],
            'short_term_debt': [9900, 6000, 2000, 1500],
            'cash_and_cash_equivalents': [29000, 34000, 120000, 5500],
            
            # Cash flow data
            'net_cash_provided_by_operating_activities': [110000, 87000, 85000, 7500],
            'free_cash_flow': [95000, 65000, 67000, 5000],
            'dividends_paid': [14000, 18000, 0, 0],
            'common_stock_repurchased': [90000, 15000, 50000, 0],
            'common_stock_issued': [1000, 500, 2000, 8000],
            
            # Market data
            'market_cap': [3000000, 2800000, 1800000, 800000],
            'enterprise_value': [3070000, 2830000, 1695000, 806000],
            'close_price': [185.0, 370.0, 171.0, 267.0]
        })
    
    def test_initialization(self):
        """Test calculator initialization"""
        self.assertIsNotNone(self.calculator.engine)
        self.assertEqual(self.calculator.data_quality_threshold, 0.5)
        self.assertIn('roic_upper', self.calculator.outlier_caps)
        self.assertIn('market_median_wacc', self.calculator.wacc_fallbacks)
    
    def test_calculate_roic_minus_wacc(self):
        """Test ROIC minus WACC calculation"""
        result = self.calculator._calculate_roic_minus_wacc(self.sample_data)
        
        # Check that result is a Series with correct length
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Check that ROIC calculations are reasonable
        # AAPL should have positive ROIC-WACC
        self.assertGreater(result.iloc[0], 0)
        
        # Check for no infinite values
        self.assertFalse(np.isinf(result).any())
    
    def test_calculate_gross_profitability(self):
        """Test gross profitability calculation"""
        result = self.calculator._calculate_gross_profitability(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # AAPL: 170000 / 351000 ≈ 0.484
        self.assertAlmostEqual(result.iloc[0], 170000/351000, places=3)
        
        # All values should be positive ratios
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())
    
    def test_calculate_cfo_to_net_income(self):
        """Test CFO to net income ratio calculation"""
        result = self.calculator._calculate_cfo_to_net_income(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # AAPL: 110000 / 97000 ≈ 1.134
        self.assertAlmostEqual(result.iloc[0], 110000/97000, places=3)
        
        # Values should be reasonable (typically 0.5 to 2.0)
        valid_values = result.dropna()
        self.assertTrue((valid_values > 0).all())
    
    def test_calculate_operating_accruals(self):
        """Test operating accruals calculation"""
        result = self.calculator._calculate_operating_accruals(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # AAPL: (97000 - 110000) / 351000 ≈ -0.037
        expected = (97000 - 110000) / 351000
        self.assertAlmostEqual(result.iloc[0], expected, places=4)
        
        # Accruals can be positive or negative
        self.assertFalse(np.isinf(result).any())
    
    def test_calculate_net_debt_to_ebitda(self):
        """Test net debt to EBITDA calculation"""
        result = self.calculator._calculate_net_debt_to_ebitda(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # AAPL calculation
        ebitda_aapl = 114000 + 11000  # operating_income + depreciation
        net_debt_aapl = (95000 + 9900) - 29000  # total_debt - cash
        expected_aapl = net_debt_aapl / ebitda_aapl
        self.assertAlmostEqual(result.iloc[0], expected_aapl, places=3)
        
        # Should handle negative EBITDA (return NaN)
        test_data = self.sample_data.copy()
        test_data.loc[0, 'operating_income'] = -50000
        result_negative = self.calculator._calculate_net_debt_to_ebitda(test_data)
        self.assertTrue(np.isnan(result_negative.iloc[0]))
    
    def test_calculate_interest_coverage(self):
        """Test interest coverage ratio calculation"""
        result = self.calculator._calculate_interest_coverage(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # AAPL: 114000 / 2600 ≈ 43.85
        self.assertAlmostEqual(result.iloc[0], 114000/2600, places=2)
        
        # GOOGL has no interest expense, should get high coverage (999.0)
        self.assertEqual(result.iloc[2], 999.0)
        
        # All valid values should be positive
        valid_values = result.dropna()
        self.assertTrue((valid_values > 0).all())
    
    def test_calculate_margin_consistency(self):
        """Test margin consistency calculation (placeholder)"""
        result = self.calculator._calculate_margin_consistency(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Should return valid consistency scores
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0.1).all())
        self.assertTrue((valid_values <= 50.0).all())
    
    def test_calculate_value_composite(self):
        """Test value composite score calculation"""
        result = self.calculator._calculate_value_composite(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Composite scores should be between 0 and 1
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())
        self.assertTrue((valid_values <= 1).all())
    
    def test_calculate_valuation_deviation_5y(self):
        """Test 5-year valuation deviation calculation (placeholder)"""
        result = self.calculator._calculate_valuation_deviation_5y(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Should return reasonable deviation percentages
        valid_values = result.dropna()
        self.assertTrue((valid_values >= -90).all())  # Not too negative
        self.assertTrue((valid_values <= 500).all())  # Not too positive
    
    def test_calculate_total_shareholder_yield(self):
        """Test total shareholder yield calculation"""
        result = self.calculator._calculate_total_shareholder_yield(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # AAPL: (14000 + 90000) / 3000000 * 100 ≈ 3.47%
        expected_aapl = (14000 + 90000) / 3000000 * 100
        self.assertAlmostEqual(result.iloc[0], expected_aapl, places=2)
        
        # All values should be non-negative percentages
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())
    
    def test_calculate_dilution_penalty(self):
        """Test dilution penalty calculation"""
        result = self.calculator._calculate_dilution_penalty(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # AAPL: -1000 / 3000000 * 100 ≈ -0.033%
        expected_aapl = -1000 / 3000000 * 100
        self.assertAlmostEqual(result.iloc[0], expected_aapl, places=4)
        
        # TSLA has high dilution (8000 shares), should be more negative
        self.assertLess(result.iloc[3], result.iloc[0])
    
    def test_calculate_quality_features(self):
        """Test quality features calculation"""
        result = self.calculator.calculate_quality_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Check that all quality columns exist
        expected_columns = [
            'symbol', 'as_of_date', 'roic_minus_wacc', 'gross_profitability',
            'cfo_to_net_income', 'operating_accruals', 'net_debt_to_ebitda',
            'interest_coverage_ttm', 'margin_cv_12q'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check symbol preservation
        pd.testing.assert_series_equal(result['symbol'], self.sample_data['symbol'])
    
    def test_calculate_value_features(self):
        """Test value features calculation"""
        result = self.calculator.calculate_value_features(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Check that all value columns exist
        expected_columns = [
            'symbol', 'as_of_date', 'value_composite', 'valuation_deviation_5y',
            'total_shareholder_yield', 'dilution_penalty'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check symbol preservation
        pd.testing.assert_series_equal(result['symbol'], self.sample_data['symbol'])
    
    def test_apply_outlier_caps(self):
        """Test outlier capping functionality"""
        # Create test data with extreme values
        test_data = pd.DataFrame({
            'symbol': ['TEST'],
            'as_of_date': ['2025-09-11'],
            'roic_minus_wacc': [200.0],  # Exceeds upper cap of 100
            'interest_coverage_ttm': [1000.0],  # Exceeds upper cap of 100
            'valuation_deviation_5y': [-95.0],  # Below lower cap of -90
            'total_shareholder_yield': [80.0]  # Exceeds upper cap of 50
        })
        
        result = self.calculator._apply_outlier_caps(test_data)
        
        # Check that values were capped
        self.assertEqual(result['roic_minus_wacc'].iloc[0], 100.0)
        self.assertEqual(result['interest_coverage_ttm'].iloc[0], 100.0)
        self.assertEqual(result['valuation_deviation_5y'].iloc[0], -90.0)
        self.assertEqual(result['total_shareholder_yield'].iloc[0], 50.0)
    
    def test_validate_data_quality(self):
        """Test data quality validation"""
        # Create test data with varying completeness
        test_data = pd.DataFrame({
            'symbol': ['GOOD', 'BAD', 'MARGINAL'],
            'as_of_date': ['2025-09-11'] * 3,
            'feature1': [1.0, np.nan, 1.0],
            'feature2': [2.0, np.nan, np.nan],
            'feature3': [3.0, np.nan, 3.0],
            'feature4': [4.0, np.nan, np.nan]
        })
        
        result = self.calculator._validate_data_quality(test_data)
        
        # Should keep records with sufficient data quality
        # GOOD: 4/4 = 100% > 50% threshold
        # MARGINAL: 2/4 = 50% = 50% threshold  
        # BAD: 0/4 = 0% < 50% threshold
        self.assertGreaterEqual(len(result), 2)  # At least GOOD and MARGINAL
        self.assertIn('GOOD', result['symbol'].values)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        quality_result = self.calculator.calculate_quality_features(empty_df)
        value_result = self.calculator.calculate_value_features(empty_df)
        
        self.assertTrue(quality_result.empty)
        self.assertTrue(value_result.empty)
        
        # Test with all-zero/null data
        null_data = pd.DataFrame({
            'symbol': ['NULL'],
            'as_of_date': ['2025-09-11'],
            'revenue': [0],
            'total_assets': [0],
            'net_income': [0],
            'market_cap': [0]
        })
        
        # Should handle gracefully without errors
        quality_result = self.calculator.calculate_quality_features(null_data)
        value_result = self.calculator.calculate_value_features(null_data)
        
        self.assertEqual(len(quality_result), 1)
        self.assertEqual(len(value_result), 1)
    
    @patch('src.scoring.quality_value_features.pd.read_sql')
    def test_get_financial_data(self, mock_read_sql):
        """Test financial data retrieval"""
        # Mock database response
        mock_read_sql.return_value = self.sample_data
        
        result = self.calculator._get_financial_data('2025-09-11', ['AAPL'])
        
        # Check that SQL query was called
        mock_read_sql.assert_called_once()
        
        # Check that data was returned correctly
        self.assertEqual(len(result), len(self.sample_data))
        self.assertIn('symbol', result.columns)
    
    @patch('src.scoring.quality_value_features.pd.read_sql')
    def test_calculate_all_features_integration(self, mock_read_sql):
        """Test full integration of all features calculation"""
        # Mock database response
        mock_read_sql.return_value = self.sample_data
        
        result = self.calculator.calculate_all_features('2025-09-11', ['AAPL', 'MSFT'])
        
        # Check that result contains all expected columns
        expected_columns = [
            'symbol', 'as_of_date',
            # Quality features
            'roic_minus_wacc', 'gross_profitability', 'cfo_to_net_income',
            'operating_accruals', 'net_debt_to_ebitda', 'interest_coverage_ttm',
            'margin_cv_12q',
            # Value features
            'value_composite', 'valuation_deviation_5y', 'total_shareholder_yield',
            'dilution_penalty'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Missing column: {col}")
        
        self.assertEqual(len(result), len(self.sample_data))
    
    def test_wacc_proxy(self):
        """Test WACC proxy functionality"""
        result = self.calculator._get_wacc_proxy(self.sample_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Should return market median WACC for all records
        expected_wacc = self.calculator.wacc_fallbacks['market_median_wacc']
        self.assertTrue((result == expected_wacc).all())


class TestQualityValueCalculatorErrors(unittest.TestCase):
    """Test error handling in QualityValueCalculator"""
    
    def test_database_connection_error(self):
        """Test database connection error handling"""
        with patch('src.scoring.quality_value_features.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = Exception("Connection failed")
            
            with self.assertRaises(QualityValueCalculationError):
                QualityValueCalculator()
    
    @patch('src.scoring.quality_value_features.pd.read_sql')
    def test_data_retrieval_error(self, mock_read_sql):
        """Test data retrieval error handling"""
        mock_engine = Mock()
        calculator = QualityValueCalculator(engine=mock_engine)
        
        mock_read_sql.side_effect = Exception("SQL error")
        
        with self.assertRaises(QualityValueCalculationError):
            calculator._get_financial_data('2025-09-11', ['AAPL'])


if __name__ == '__main__':
    unittest.main()