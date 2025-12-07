#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Suite for Scoring Validation and Audit System - Phase 3 Task 6

This module provides comprehensive testing for the validation system including:
- Unit tests for all validation functions
- Integration tests with database
- Alert system testing
- HTML generation testing
- Error handling validation

Usage:
    python src/scoring/test_validation.py
    pytest src/scoring/test_validation.py -v
    
Test Coverage:
    - Daily validation KPIs
    - Monthly evaluation metrics
    - Alert generation logic
    - HTML report generation
    - Database integration
    - Error scenarios
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import warnings

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# テスト対象のモジュールをインポート
try:
    from investment_toolkit.scoring.validation import (
        ScoringValidator, AlertLevel, ValidationAlert,
        DailyValidationResults, MonthlyValidationResults
    )
except ImportError as e:
    print(f"Error importing validation module: {e}")
    sys.exit(1)


class TestScoringValidator(unittest.TestCase):
    """ScoringValidator クラスのテストケース"""
    
    def setUp(self):
        """テスト用のセットアップ"""
        self.validator = ScoringValidator()
        self.test_date = datetime(2025, 9, 11)
        self.test_month = datetime(2025, 9, 1)
        
        # テスト用のモックデータ
        self.mock_daily_results = DailyValidationResults(
            date=self.test_date,
            red_flag_analysis={
                "total_stocks": 50,
                "red_flag_stocks": 8,
                "red_flag_rate": 16.0,
                "common_red_flags": "財務健全性;成長率"
            },
            pillar_distribution={
                "value_pillar": {"avg": 65.2, "std": 12.1},
                "growth_pillar": {"avg": 70.8, "std": 15.3},
                "quality_pillar": {"avg": 62.5, "std": 11.7},
                "momentum_pillar": {"avg": 68.9, "std": 14.2},
                "risk_pillar": {"avg": 55.3, "std": 9.8},
                "total_score": {"min": 45.2, "max": 89.7, "avg": 67.3, "std": 11.5}
            },
            sector_analysis={
                "sector_distribution": [
                    {"sector": "Technology", "stock_count": 12, "percentage": 24.0, "avg_score": 72.1, "alert_level": "NORMAL"},
                    {"sector": "Healthcare", "stock_count": 8, "percentage": 16.0, "avg_score": 68.5, "alert_level": "NORMAL"}
                ],
                "max_percentage": 24.0,
                "dominant_sector": "Technology",
                "sector_count": 8,
                "concentration_risk": "LOW"
            },
            driver_analysis={
                "driver_distribution": [
                    {"driver": "Growth", "stock_count": 15, "percentage": 30.0},
                    {"driver": "Quality", "stock_count": 12, "percentage": 24.0},
                    {"driver": "Momentum", "stock_count": 10, "percentage": 20.0}
                ],
                "max_percentage": 30.0,
                "dominant_driver": "Growth",
                "monopoly_risk": "LOW"
            },
            score_statistics={
                "total_stocks": 2500,
                "min_score": 15.2,
                "max_score": 95.8,
                "avg_score": 50.0,
                "std_score": 18.5,
                "top50_range": 44.5,
                "compression_risk": "LOW"
            },
            alerts=[]
        )
    
    def test_validator_initialization(self):
        """Validator の初期化テスト"""
        # 正常な初期化
        validator = ScoringValidator()
        self.assertIsNotNone(validator)
        self.assertIsNotNone(validator.daily_alert_conditions)
        self.assertIsNotNone(validator.monthly_alert_conditions)
        
        # 閾値の確認
        self.assertIn("red_flag_contamination", validator.daily_alert_conditions)
        self.assertIn("warning_threshold", validator.daily_alert_conditions["red_flag_contamination"])
    
    def test_alert_level_enum(self):
        """AlertLevel enum のテスト"""
        self.assertEqual(AlertLevel.INFO.value, "info")
        self.assertEqual(AlertLevel.WARNING.value, "warning")
        self.assertEqual(AlertLevel.CRITICAL.value, "critical")
        self.assertEqual(AlertLevel.EMERGENCY.value, "emergency")
    
    def test_validation_alert_dataclass(self):
        """ValidationAlert データクラスのテスト"""
        alert = ValidationAlert(
            timestamp=datetime.now(),
            alert_type="test_alert",
            level=AlertLevel.WARNING,
            message="Test message",
            value=25.0,
            threshold=20.0,
            recommended_action="Test action",
            dashboard_section="test_section"
        )
        
        self.assertEqual(alert.alert_type, "test_alert")
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.value, 25.0)
        self.assertEqual(alert.threshold, 20.0)
    
    @patch('src.scoring.validation.get_connection')
    def test_analyze_red_flags_success(self, mock_connection):
        """赤旗分析の正常ケーステスト"""
        # モックデータベース接続のセットアップ
        mock_conn = Mock()
        mock_connection.return_value.__enter__.return_value = mock_conn
        
        # モック結果のセットアップ
        mock_result = Mock()
        mock_result.total_stocks = 50
        mock_result.red_flag_stocks = 8
        mock_result.red_flag_rate = 16.0
        mock_result.common_red_flags = "財務健全性;成長率"
        
        mock_conn.execute.return_value.fetchone.return_value = mock_result
        
        # テスト実行
        result = self.validator._analyze_red_flags(mock_conn, self.test_date)
        
        # 結果の検証
        self.assertEqual(result["total_stocks"], 50)
        self.assertEqual(result["red_flag_stocks"], 8)
        self.assertEqual(result["red_flag_rate"], 16.0)
        self.assertIn("common_red_flags", result)
    
    @patch('src.scoring.validation.get_connection')
    def test_analyze_red_flags_no_data(self, mock_connection):
        """赤旗分析のデータなしケーステスト"""
        mock_conn = Mock()
        mock_connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None
        
        result = self.validator._analyze_red_flags(mock_conn, self.test_date)
        
        # デフォルト値の確認
        self.assertEqual(result["total_stocks"], 0)
        self.assertEqual(result["red_flag_stocks"], 0)
        self.assertEqual(result["red_flag_rate"], 0.0)
    
    @patch('src.scoring.validation.get_connection')
    def test_analyze_red_flags_database_error(self, mock_connection):
        """赤旗分析のデータベースエラーテスト"""
        mock_conn = Mock()
        mock_connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.side_effect = Exception("Database connection failed")
        
        result = self.validator._analyze_red_flags(mock_conn, self.test_date)
        
        # エラーハンドリングの確認
        self.assertIn("error", result)
        self.assertEqual(result["total_stocks"], 0)
    
    def test_check_threshold_alert_warning(self):
        """閾値アラートの警告レベルテスト"""
        alerts = self.validator._check_threshold_alert(
            "red_flag_contamination", 25.0, "Test message", "test_metric", "test_section"
        )
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, AlertLevel.WARNING)
        self.assertEqual(alerts[0].value, 25.0)
    
    def test_check_threshold_alert_critical(self):
        """閾値アラートの重大レベルテスト"""
        alerts = self.validator._check_threshold_alert(
            "red_flag_contamination", 40.0, "Critical message", "test_metric", "test_section"
        )
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, AlertLevel.CRITICAL)
    
    def test_check_threshold_alert_emergency(self):
        """閾値アラートの緊急レベルテスト"""
        alerts = self.validator._check_threshold_alert(
            "red_flag_contamination", 55.0, "Emergency message", "test_metric", "test_section"
        )
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, AlertLevel.EMERGENCY)
    
    def test_check_threshold_alert_no_alert(self):
        """閾値アラートのアラートなしテスト"""
        alerts = self.validator._check_threshold_alert(
            "red_flag_contamination", 15.0, "Safe message", "test_metric", "test_section"
        )
        
        self.assertEqual(len(alerts), 0)
    
    def test_check_threshold_alert_reverse(self):
        """逆転閾値（スコア圧縮）のテスト"""
        # 低い値で緊急アラート（reverse_threshold=True）
        alerts = self.validator._check_threshold_alert(
            "score_compression", 1.0, "Compression message", "score_range", "test_section", 
            reverse_threshold=True
        )
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, AlertLevel.EMERGENCY)
    
    def test_generate_daily_alerts(self):
        """日次アラート生成のテスト"""
        alerts = self.validator._generate_daily_alerts(self.mock_daily_results)
        
        # アラートリストが生成されることを確認
        self.assertIsInstance(alerts, list)
        
        # 現在のテストデータでは閾値を超えていないのでアラートは少ない
        for alert in alerts:
            self.assertIsInstance(alert, ValidationAlert)
            self.assertIn(alert.level, [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY])
    
    def test_generate_daily_alerts_high_values(self):
        """高い値での日次アラート生成テスト"""
        # 閾値を超える値でテストデータを変更
        high_risk_results = self.mock_daily_results
        high_risk_results.red_flag_analysis["red_flag_rate"] = 45.0  # 緊急レベル
        high_risk_results.sector_analysis["max_percentage"] = 40.0   # 緊急レベル
        
        alerts = self.validator._generate_daily_alerts(high_risk_results)
        
        # 緊急アラートが生成されることを確認
        emergency_alerts = [a for a in alerts if a.level == AlertLevel.EMERGENCY]
        self.assertGreater(len(emergency_alerts), 0)
    
    def test_get_alert_css_class(self):
        """CSS クラス取得のテスト"""
        # 正常値
        css_class = self.validator._get_alert_css_class(15.0, "red_flag_contamination")
        self.assertEqual(css_class, "text-success")
        
        # 警告レベル
        css_class = self.validator._get_alert_css_class(25.0, "red_flag_contamination")
        self.assertEqual(css_class, "text-warning")
        
        # 重大レベル
        css_class = self.validator._get_alert_css_class(40.0, "red_flag_contamination")
        self.assertEqual(css_class, "text-danger")
        
        # 緊急レベル
        css_class = self.validator._get_alert_css_class(55.0, "red_flag_contamination")
        self.assertEqual(css_class, "text-danger font-weight-bold")
    
    def test_generate_alert_html_list_no_alerts(self):
        """アラートなしのHTML リスト生成テスト"""
        html = self.validator._generate_alert_html_list([])
        
        self.assertIn("問題ありません", html)
        self.assertIn("alert-success", html)
    
    def test_generate_alert_html_list_with_alerts(self):
        """アラートありのHTML リスト生成テスト"""
        test_alerts = [
            ValidationAlert(
                timestamp=datetime.now(),
                alert_type="test",
                level=AlertLevel.WARNING,
                message="Test warning",
                value=25.0,
                threshold=20.0,
                recommended_action="Check system",
                dashboard_section="test"
            )
        ]
        
        html = self.validator._generate_alert_html_list(test_alerts)
        
        self.assertIn("Test warning", html)
        self.assertIn("Check system", html)
        self.assertIn("⚠️", html)  # 警告アイコン
    
    def test_generate_daily_html_section(self):
        """日次HTML セクション生成のテスト"""
        html = self.validator.generate_daily_html_section(self.mock_daily_results)
        
        # 基本構造の確認
        self.assertIn("📊 スコアリング品質チェック", html)
        self.assertIn("赤旗混入率", html)
        self.assertIn("セクター分散", html)
        self.assertIn("16.0%", html)  # 赤旗混入率
        self.assertIn("Technology", html)  # 支配的セクター
        
        # ピラー情報の確認
        self.assertIn("Value:", html)
        self.assertIn("Growth:", html)
        self.assertIn("65.2", html)  # Value平均スコア
    
    def test_log_validation_results(self):
        """検証結果ログ記録のテスト"""
        with patch('src.scoring.validation.logger') as mock_logger:
            self.validator.log_validation_results(self.mock_daily_results)
            
            # ログ出力が呼ばれることを確認
            self.assertTrue(mock_logger.info.called)
            
            # ログ内容の確認
            logged_messages = [call.args[0] for call in mock_logger.info.call_args_list]
            self.assertTrue(any("Red flag rate: 16.0%" in msg for msg in logged_messages))
    
    @patch('src.scoring.validation.get_connection')
    def test_run_daily_validation_integration(self, mock_connection):
        """日次検証の統合テスト"""
        # モックデータベース接続のセットアップ
        mock_conn = Mock()
        mock_connection.return_value.__enter__.return_value = mock_conn
        
        # 各分析メソッドの戻り値をモック
        with patch.object(self.validator, '_analyze_red_flags', return_value={"red_flag_rate": 15.0}), \
             patch.object(self.validator, '_analyze_pillar_distribution', return_value={"avg_score": 65.0}), \
             patch.object(self.validator, '_analyze_sector_concentration', return_value={"max_percentage": 20.0}), \
             patch.object(self.validator, '_analyze_score_drivers', return_value={"max_percentage": 25.0}), \
             patch.object(self.validator, '_calculate_score_statistics', return_value={"top50_range": 15.0}):
            
            result = self.validator.run_daily_validation("2025-09-11")
            
            # 結果の型確認
            self.assertIsInstance(result, DailyValidationResults)
            self.assertEqual(result.date, datetime(2025, 9, 11))
            self.assertIsInstance(result.alerts, list)


class TestValidationIntegration(unittest.TestCase):
    """統合テストケース"""
    
    @patch('src.scoring.validation.get_connection')
    def test_database_integration_mock(self, mock_connection):
        """データベース統合のモックテスト"""
        validator = ScoringValidator()
        mock_conn = Mock()
        mock_connection.return_value.__enter__.return_value = mock_conn
        
        # SQLクエリが実行されることを確認
        mock_conn.execute.return_value.fetchone.return_value = Mock(
            total_stocks=50, red_flag_stocks=5, red_flag_rate=10.0, common_red_flags=""
        )
        
        result = validator._analyze_red_flags(mock_conn, datetime(2025, 9, 11))
        
        # データベースクエリが呼び出されたことを確認
        self.assertTrue(mock_conn.execute.called)
        self.assertEqual(result["red_flag_rate"], 10.0)
    
    def test_html_output_quality(self):
        """HTML 出力品質のテスト"""
        validator = ScoringValidator()
        
        # テスト用の結果データ
        test_results = DailyValidationResults(
            date=datetime(2025, 9, 11),
            red_flag_analysis={"red_flag_rate": 15.0, "red_flag_stocks": 7, "total_stocks": 50},
            pillar_distribution={"value_pillar": {"avg": 65.0}, "growth_pillar": {"avg": 70.0}},
            sector_analysis={"max_percentage": 25.0, "dominant_sector": "Technology", "sector_count": 8},
            driver_analysis={"max_percentage": 30.0, "dominant_driver": "Growth"},
            score_statistics={"top50_range": 20.0},
            alerts=[]
        )
        
        html = validator.generate_daily_html_section(test_results)
        
        # HTML の妥当性チェック
        self.assertIn('<div class="validation-section">', html)
        self.assertIn('</div>', html)
        self.assertNotIn('<script>', html)  # XSS 対策
        
        # 数値が正しくフォーマットされていることを確認
        self.assertIn('15.0%', html)
        self.assertIn('25.0%', html)


class TestErrorHandling(unittest.TestCase):
    """エラーハンドリングのテストケース"""
    
    def setUp(self):
        self.validator = ScoringValidator()
    
    @patch('src.scoring.validation.get_connection')
    def test_database_connection_failure(self, mock_connection):
        """データベース接続失敗のテスト"""
        mock_connection.side_effect = Exception("Connection failed")
        
        # エラーが適切にハンドリングされることを確認
        with self.assertRaises(Exception):
            self.validator.run_daily_validation("2025-09-11")
    
    @patch('src.scoring.validation.get_connection')
    def test_sql_query_error(self, mock_connection):
        """SQLクエリエラーのテスト"""
        mock_conn = Mock()
        mock_connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.side_effect = Exception("SQL error")
        
        result = self.validator._analyze_red_flags(mock_conn, datetime(2025, 9, 11))
        
        # エラー情報が結果に含まれることを確認
        self.assertIn("error", result)
        self.assertEqual(result["total_stocks"], 0)
    
    def test_invalid_date_format(self):
        """無効な日付フォーマットのテスト"""
        with patch.object(self.validator, '_analyze_red_flags'):
            with self.assertRaises(ValueError):
                self.validator.run_daily_validation("invalid-date")
    
    def test_missing_schema_handling(self):
        """スキーマファイル不存在時のハンドリングテスト"""
        validator = ScoringValidator(schema_path="/nonexistent/path.yaml")
        
        # スキーマなしでも動作することを確認
        self.assertIsNotNone(validator)
    
    def test_alert_generation_error_handling(self):
        """アラート生成時のエラーハンドリング"""
        # 無効なデータでアラート生成
        invalid_results = DailyValidationResults(
            date=datetime(2025, 9, 11),
            red_flag_analysis={},  # 空の辞書
            pillar_distribution={},
            sector_analysis={},
            driver_analysis={},
            score_statistics={},
            alerts=[]
        )
        
        # エラーが発生しても処理が続行されることを確認
        alerts = self.validator._generate_daily_alerts(invalid_results)
        self.assertIsInstance(alerts, list)


def run_performance_tests():
    """パフォーマンステスト（手動実行）"""
    print("\n🚀 パフォーマンステスト実行中...")
    
    validator = ScoringValidator()
    
    # 大量データでのテスト
    start_time = datetime.now()
    
    # モックデータでの処理時間測定
    with patch.object(validator, '_analyze_red_flags', return_value={"red_flag_rate": 15.0}), \
         patch.object(validator, '_analyze_pillar_distribution', return_value={}), \
         patch.object(validator, '_analyze_sector_concentration', return_value={}), \
         patch.object(validator, '_analyze_score_drivers', return_value={}), \
         patch.object(validator, '_calculate_score_statistics', return_value={}):
        
        try:
            result = validator.run_daily_validation("2025-09-11")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"  ✅ 日次検証処理時間: {processing_time:.3f}秒")
            
            if processing_time > 30:  # 30秒以上は警告
                print(f"  ⚠️ 処理時間が長すぎます: {processing_time:.3f}秒")
            else:
                print(f"  ✅ 処理時間良好: {processing_time:.3f}秒")
                
        except Exception as e:
            print(f"  ❌ パフォーマンステストエラー: {e}")


def main():
    """メインテスト実行関数"""
    print("🔍 スコアリング検証システムのテスト実行")
    print("="*60)
    
    # 警告を抑制
    warnings.filterwarnings("ignore", category=ResourceWarning)
    
    # テストスイートの作成
    test_suite = unittest.TestSuite()
    
    # テストクラスを追加
    test_classes = [
        TestScoringValidator,
        TestValidationIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    # 結果サマリー
    print("\n" + "="*60)
    print(f"🧪 テスト実行完了")
    print(f"   実行: {test_result.testsRun} テスト")
    print(f"   成功: {test_result.testsRun - len(test_result.failures) - len(test_result.errors)}")
    print(f"   失敗: {len(test_result.failures)}")
    print(f"   エラー: {len(test_result.errors)}")
    
    # パフォーマンステストの実行
    if "--performance" in sys.argv:
        run_performance_tests()
    
    # 失敗した場合の詳細表示
    if test_result.failures or test_result.errors:
        print("\n❌ 失敗したテスト:")
        for test, traceback in test_result.failures + test_result.errors:
            print(f"   {test}: {traceback.split(chr(10))[0]}")
        return False
    else:
        print("\n✅ すべてのテストが成功しました！")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)