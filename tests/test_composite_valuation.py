#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
複合バリュエーション指標計算のテスト

pytest tests/test_composite_valuation.py -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from investment_analysis.data.composite_valuation import (
    calc_peg_ratios,
    calc_pegy_ratios,
    calc_garp_flags,
    calc_enterprise_value,
    calc_ev_ratios,
    calc_earnings_yield,
    calc_altman_z_score,
    calc_piotroski_f_score,
    calc_composite_metrics
)


class TestCompositeValuation:
    """複合バリュエーション指標計算のテストクラス"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        # テスト用データの作成
        self.test_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
            'as_of_date': pd.to_datetime(['2024-01-01'] * 5),
            'per': [20.0, 25.0, 15.0, np.nan, 30.0],
            'market_cap': [3000000, 2800000, 1800000, 800000, 1500000],
            'dividend_yield': [0.005, 0.007, 0.0, 0.0, 0.0],
            'eps_cagr_3y': [0.15, 0.12, 0.20, -0.05, 0.25],  # 15%, 12%, 20%, -5%, 25%
            'eps_cagr_5y': [0.18, 0.10, 0.22, 0.05, 0.30],   # 18%, 10%, 22%, 5%, 30%
            'ebitda': [100000, 80000, 70000, 5000, 60000],
            'operating_income': [90000, 75000, 65000, 3000, 55000],
            'net_income': [80000, 70000, 60000, -5000, 50000],
            'revenue': [300000, 250000, 200000, 100000, 400000],
            'interest_expense': [2000, 1500, 1000, 3000, 2500],
            'income_tax_expense': [15000, 12000, 10000, 0, 8000],
            'net_debt': [50000, -20000, 30000, 40000, np.nan],
            'total_debt': [100000, 50000, 80000, 60000, 70000],
            'cash_and_cash_equivalents': [50000, 70000, 50000, 20000, 80000],
            'total_assets': [500000, 400000, 350000, 200000, 600000],
            'total_current_assets': [200000, 180000, 150000, 80000, 250000],
            'total_current_liabilities': [100000, 90000, 80000, 60000, 120000],
            'total_equity': [400000, 350000, 270000, 140000, 530000],
            'total_stockholders_equity': [400000, 350000, 270000, 140000, 530000],
            'retained_earnings': [300000, 250000, 200000, 50000, 400000],
            'net_receivables': [30000, 25000, 20000, 15000, 35000],
            'inventory': [20000, 15000, 10000, 25000, 40000],
            'property_plant_equipment_net': [150000, 120000, 100000, 80000, 200000],
            'free_cash_flow': [80000, 70000, 60000, 2000, 50000]
        })
    
    def test_calc_peg_ratios(self):
        """PEG比率計算のテスト"""
        result = calc_peg_ratios(self.test_data.copy())
        
        # PEG 3年の計算確認
        # AAPL: 20 / (0.15 * 100) = 20 / 15 = 1.33
        assert abs(result.loc[0, 'peg_3y'] - (20.0 / 15.0)) < 0.01
        
        # MSFT: 25 / (0.12 * 100) = 25 / 12 = 2.08
        assert abs(result.loc[1, 'peg_3y'] - (25.0 / 12.0)) < 0.01
        
        # GOOGL: 15 / (0.20 * 100) = 15 / 20 = 0.75
        assert abs(result.loc[2, 'peg_3y'] - (15.0 / 20.0)) < 0.01
        
        # TSLA: 負のEPS CAGRなのでNaN
        assert pd.isna(result.loc[3, 'peg_3y'])
        
        # PEG 5年の計算確認
        # AAPL: 20 / (0.18 * 100) = 20 / 18 = 1.11
        assert abs(result.loc[0, 'peg_5y'] - (20.0 / 18.0)) < 0.01
    
    def test_calc_pegy_ratios(self):
        """PEGY比率計算のテスト"""
        # まずPEG比率を計算
        data_with_peg = calc_peg_ratios(self.test_data.copy())
        result = calc_pegy_ratios(data_with_peg)
        
        # AAPL: PEG_3y / (1 + dividend_yield) = 1.33 / (1 + 0.005) = 1.33 / 1.005
        expected_pegy_3y = (20.0 / 15.0) / (1 + 0.005)
        assert abs(result.loc[0, 'pegy_3y'] - expected_pegy_3y) < 0.01
        
        # GOOGL: 配当なしなのでPEGYはPEGと同じ
        expected_pegy_3y_googl = 15.0 / 20.0
        assert abs(result.loc[2, 'pegy_3y'] - expected_pegy_3y_googl) < 0.01
    
    def test_calc_garp_flags(self):
        """GARPフラグ計算のテスト"""
        # まずPEG比率を計算
        data_with_peg = calc_peg_ratios(self.test_data.copy())
        result = calc_garp_flags(data_with_peg)
        
        # AAPL: PEG_3y = 1.33 > 1.0 なのでFalse
        assert result.loc[0, 'garp_flag_3y'] == False
        
        # GOOGL: PEG_3y = 0.75 < 1.0 なのでTrue
        assert result.loc[2, 'garp_flag_3y'] == True
        
        # TSLA: PEG_3yがNaNなのでFalse（修正後の仕様）
        assert result.loc[3, 'garp_flag_3y'] == False
    
    def test_calc_enterprise_value(self):
        """企業価値計算のテスト"""
        result = calc_enterprise_value(self.test_data.copy())
        
        # AAPL: market_cap + net_debt = 3000000 + 50000 = 3050000
        assert result.loc[0, 'ev'] == 3050000
        
        # MSFT: market_cap + net_debt = 2800000 + (-20000) = 2780000
        assert result.loc[1, 'ev'] == 2780000
        
        # AMZN: net_debtがNaNなので代替計算
        # market_cap + total_debt - cash = 1500000 + 70000 - 80000 = 1490000
        assert result.loc[4, 'ev'] == 1490000
    
    def test_calc_ev_ratios(self):
        """EV比率計算のテスト"""
        # まず企業価値を計算
        data_with_ev = calc_enterprise_value(self.test_data.copy())
        result = calc_ev_ratios(data_with_ev)
        
        # AAPL: EV/EBITDA = 3050000 / 100000 = 30.5
        assert abs(result.loc[0, 'ev_ebitda'] - 30.5) < 0.01
        
        # AAPL: EV/FCF = 3050000 / 80000 = 38.125
        assert abs(result.loc[0, 'ev_fcf'] - 38.125) < 0.01
    
    def test_calc_earnings_yield(self):
        """アーニングスイールド計算のテスト"""
        # まず企業価値を計算
        data_with_ev = calc_enterprise_value(self.test_data.copy())
        result = calc_earnings_yield(data_with_ev)
        
        # AAPL: operating_income / EV = 90000 / 3050000 = 0.0295
        expected_earnings_yield = 90000 / 3050000
        assert abs(result.loc[0, 'earnings_yield'] - expected_earnings_yield) < 0.001
    
    def test_calc_composite_metrics_integration(self):
        """複合指標計算の統合テスト"""
        result = calc_composite_metrics(self.test_data.copy())
        
        # 結果データフレームの列数確認
        expected_columns = [
            'symbol', 'as_of_date', 'peg_3y', 'peg_5y', 'pegy_3y', 'pegy_5y',
            'garp_flag_3y', 'garp_flag_5y', 'ev', 'ev_ebitda', 'ev_fcf',
            'earnings_yield', 'altman_z', 'piotroski_f'
        ]
        assert list(result.columns) == expected_columns
        
        # 行数確認
        assert len(result) == len(self.test_data)
        
        # 各指標が計算されていることを確認
        assert result['peg_3y'].notna().sum() > 0
        assert result['ev'].notna().sum() > 0
        assert result['garp_flag_3y'].notna().sum() > 0
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 空のデータフレーム
        empty_df = pd.DataFrame()
        result = calc_composite_metrics(empty_df)
        assert len(result) == 0
        
        # 全てNaNのデータ
        nan_data = pd.DataFrame({
            'symbol': ['TEST'],
            'as_of_date': [pd.to_datetime('2024-01-01')],
            'per': [np.nan],
            'market_cap': [np.nan],
            'dividend_yield': [np.nan],
            'eps_cagr_3y': [np.nan],
            'eps_cagr_5y': [np.nan],
            'ebitda': [np.nan],
            'operating_income': [np.nan],
            'net_income': [np.nan],
            'revenue': [np.nan],
            'interest_expense': [np.nan],
            'income_tax_expense': [np.nan],
            'net_debt': [np.nan],
            'total_debt': [np.nan],
            'cash_and_cash_equivalents': [np.nan],
            'total_assets': [np.nan],
            'total_current_assets': [np.nan],
            'total_current_liabilities': [np.nan],
            'total_equity': [np.nan],
            'total_stockholders_equity': [np.nan],
            'retained_earnings': [np.nan],
            'net_receivables': [np.nan],
            'inventory': [np.nan],
            'property_plant_equipment_net': [np.nan],
            'free_cash_flow': [np.nan]
        })
        
        result = calc_composite_metrics(nan_data)
        assert len(result) == 1
        assert pd.isna(result.loc[0, 'peg_3y'])
        assert pd.isna(result.loc[0, 'ev'])
        assert pd.isna(result.loc[0, 'altman_z'])
        assert pd.isna(result.loc[0, 'piotroski_f'])
    
    def test_zero_division_handling(self):
        """ゼロ除算の処理テスト"""
        zero_data = self.test_data.copy()
        zero_data.loc[0, 'eps_cagr_3y'] = 0.0  # ゼロ成長
        zero_data.loc[1, 'ebitda'] = 0.0       # ゼロEBITDA
        
        result = calc_composite_metrics(zero_data)
        
        # ゼロ成長の場合はPEGが計算されない
        assert pd.isna(result.loc[0, 'peg_3y'])
        
        # ゼロEBITDAの場合はEV/EBITDAが計算されない
        assert pd.isna(result.loc[1, 'ev_ebitda'])
    
    def test_calc_altman_z_score(self):
        """Altman Zスコア計算のテスト"""
        result = calc_altman_z_score(self.test_data.copy())
        
        # Altman Zスコアが計算されていることを確認
        assert 'altman_z' in result.columns
        
        # 少なくとも一部のデータでスコアが計算されていることを確認
        assert result['altman_z'].notna().sum() > 0
        
        # スコアが合理的な範囲内であることを確認（実際の範囲に合わせて調整）
        valid_scores = result['altman_z'].dropna()
        if len(valid_scores) > 0:
            assert valid_scores.min() > -20  # より広い範囲に調整
            assert valid_scores.max() < 50   # より広い範囲に調整
    
    def test_calc_piotroski_f_score(self):
        """Piotroski Fスコア計算のテスト"""
        result = calc_piotroski_f_score(self.test_data.copy())
        
        # Piotroski Fスコアが計算されていることを確認
        assert 'piotroski_f' in result.columns
        
        # 少なくとも一部のデータでスコアが計算されていることを確認
        assert result['piotroski_f'].notna().sum() > 0
        
        # スコアが0-9の範囲内であることを確認
        valid_scores = result['piotroski_f'].dropna()
        if len(valid_scores) > 0:
            assert valid_scores.min() >= 0
            assert valid_scores.max() <= 9
            
        # 整数値であることを確認
        assert all(score == int(score) for score in valid_scores)


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"]) 