#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scoring Validation and Audit System - Phase 3 Task 6

This module implements comprehensive validation and audit capabilities for the new 5-pillar scoring system.
Provides daily monitoring KPIs, monthly evaluation metrics, alert system, and report integration.

Key Features:
- Daily KPIs: Red flag contamination, pillar distribution, sector bias, driver analysis
- Monthly KPIs: Decile backtest, pillar contribution, stability metrics, turnover analysis
- Alert System: 4-level severity (Info/Warning/Critical/Emergency) with automated detection
- Dashboard Integration: Seamless integration with existing HTML reporting system
- Statistical Analysis: Correlation analysis, outlier detection, performance evaluation

Usage:
    validator = ScoringValidator()
    daily_results = validator.run_daily_validation(target_date="2025-09-11")
    monthly_results = validator.run_monthly_evaluation(month="2025-09")
    alerts = validator.generate_alerts(daily_results)
    
References:
    - Specification: docs/validation_checks.md
    - Implementation Roadmap: implementation_roadmap.md (Task 6)
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
from dataclasses import dataclass
from enum import Enum
import json
import warnings

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# プロジェクト内のモジュールをインポート
try:
    from investment_analysis.utilities.config import get_connection, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
except ImportError as e:
    print(f"Warning: Could not import database config: {e}")

try:
    from investment_analysis.scoring.schema_parser import ScoreSchemaParser
except ImportError as e:
    print(f"Warning: Could not import schema parser: {e}")

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """アラートレベルの定義"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ValidationAlert:
    """検証アラートのデータクラス"""
    timestamp: datetime
    alert_type: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    recommended_action: str
    dashboard_section: str

@dataclass
class DailyValidationResults:
    """日次検証結果のデータクラス"""
    date: datetime
    red_flag_analysis: Dict[str, Any]
    pillar_distribution: Dict[str, Any]
    sector_analysis: Dict[str, Any]
    driver_analysis: Dict[str, Any]
    score_statistics: Dict[str, Any]
    alerts: List[ValidationAlert]

@dataclass
class MonthlyValidationResults:
    """月次検証結果のデータクラス"""
    month: datetime
    decile_backtest: Dict[str, Any]
    pillar_contribution: Dict[str, Any]
    stability_metrics: Dict[str, Any]
    turnover_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    alerts: List[ValidationAlert]

class ScoringValidator:
    """
    スコアリングシステムの検証・監査を行うメインクラス
    
    機能:
    - 日次検証KPIの計算
    - 月次評価指標の算出
    - アラート生成・管理
    - レポート統合機能
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        検証システムの初期化
        
        Args:
            schema_path: スキーマファイルのパス (デフォルトは config/score_schema.yaml)
        """
        self.logger = logging.getLogger(__name__)
        
        # スキーマパーサーの初期化
        if schema_path is None:
            schema_path = str(project_root / "config" / "score_schema.yaml")
        
        try:
            self.schema_parser = ScoreSchemaParser()
            self.schema = self.schema_parser.load_schema(schema_path)
        except Exception as e:
            self.logger.warning(f"Could not load schema: {e}")
            self.schema_parser = None
            self.schema = {}
        
        # アラート条件の定義
        self.daily_alert_conditions = {
            "red_flag_contamination": {
                "warning_threshold": 20.0,
                "critical_threshold": 35.0,
                "emergency_threshold": 50.0
            },
            "sector_concentration": {
                "warning_threshold": 25.0,
                "critical_threshold": 35.0,
                "emergency_threshold": 45.0
            },
            "driver_monopoly": {
                "warning_threshold": 35.0,
                "critical_threshold": 45.0,
                "emergency_threshold": 55.0
            },
            "score_compression": {
                "warning_threshold": 10.0,
                "critical_threshold": 5.0,
                "emergency_threshold": 2.0
            }
        }
        
        self.monthly_alert_conditions = {
            "backtest_failure": {
                "warning_threshold": 0.5,
                "critical_threshold": -0.5,
                "emergency_threshold": -2.0
            },
            "turnover_instability": {
                "warning_threshold": 40.0,
                "critical_threshold": 60.0,
                "emergency_threshold": 80.0
            },
            "correlation_breakdown": {
                "warning_threshold": 0.1,
                "critical_threshold": -0.05,
                "emergency_threshold": -0.2
            }
        }

    def run_daily_validation(self, target_date: str) -> DailyValidationResults:
        """
        日次検証を実行
        
        Args:
            target_date: 検証対象日 (YYYY-MM-DD)
            
        Returns:
            DailyValidationResults: 日次検証結果
        """
        try:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            self.logger.info(f"Running daily validation for {target_date}")
            
            with get_connection() as conn:
                # 各検証項目を実行
                red_flag_analysis = self._analyze_red_flags(conn, date_obj)
                pillar_distribution = self._analyze_pillar_distribution(conn, date_obj)
                sector_analysis = self._analyze_sector_concentration(conn, date_obj)
                driver_analysis = self._analyze_score_drivers(conn, date_obj)
                score_statistics = self._calculate_score_statistics(conn, date_obj)
                
                # 検証結果をまとめる
                results = DailyValidationResults(
                    date=date_obj,
                    red_flag_analysis=red_flag_analysis,
                    pillar_distribution=pillar_distribution,
                    sector_analysis=sector_analysis,
                    driver_analysis=driver_analysis,
                    score_statistics=score_statistics,
                    alerts=[]
                )
                
                # アラートを生成
                alerts = self._generate_daily_alerts(results)
                results.alerts = alerts
                
                self.logger.info(f"Daily validation completed. {len(alerts)} alerts generated.")
                return results
                
        except Exception as e:
            self.logger.error(f"Daily validation failed: {e}")
            raise

    def run_monthly_evaluation(self, month: str) -> MonthlyValidationResults:
        """
        月次評価を実行
        
        Args:
            month: 評価対象月 (YYYY-MM)
            
        Returns:
            MonthlyValidationResults: 月次評価結果
        """
        try:
            month_obj = datetime.strptime(month + "-01", "%Y-%m-%d")
            self.logger.info(f"Running monthly evaluation for {month}")
            
            with get_connection() as conn:
                # 各評価項目を実行
                decile_backtest = self._perform_decile_backtest(conn, month_obj)
                pillar_contribution = self._analyze_pillar_contribution(conn, month_obj)
                stability_metrics = self._calculate_stability_metrics(conn, month_obj)
                turnover_analysis = self._analyze_turnover(conn, month_obj)
                correlation_analysis = self._analyze_correlations(conn, month_obj)
                
                # 評価結果をまとめる
                results = MonthlyValidationResults(
                    month=month_obj,
                    decile_backtest=decile_backtest,
                    pillar_contribution=pillar_contribution,
                    stability_metrics=stability_metrics,
                    turnover_analysis=turnover_analysis,
                    correlation_analysis=correlation_analysis,
                    alerts=[]
                )
                
                # アラートを生成
                alerts = self._generate_monthly_alerts(results)
                results.alerts = alerts
                
                self.logger.info(f"Monthly evaluation completed. {len(alerts)} alerts generated.")
                return results
                
        except Exception as e:
            self.logger.error(f"Monthly evaluation failed: {e}")
            raise

    def _analyze_red_flags(self, conn, date: datetime) -> Dict[str, Any]:
        """上位50銘柄の赤旗混入率を分析"""
        try:
            query = text("""
                WITH top_50_daily AS (
                    SELECT symbol, total_score, red_flags, sector
                    FROM daily_scores ds
                    LEFT JOIN stock_info si ON ds.symbol = si.symbol 
                    WHERE ds.date = :target_date 
                    ORDER BY total_score DESC 
                    LIMIT 50
                )
                SELECT 
                    COUNT(*) as total_stocks,
                    COUNT(CASE WHEN red_flags IS NOT NULL AND red_flags != '' THEN 1 END) as red_flag_stocks,
                    ROUND(COUNT(CASE WHEN red_flags IS NOT NULL AND red_flags != '' THEN 1 END) * 100.0 / COUNT(*), 2) as red_flag_rate,
                    STRING_AGG(DISTINCT CASE WHEN red_flags IS NOT NULL AND red_flags != '' 
                              THEN red_flags END, '; ') as common_red_flags
                FROM top_50_daily
            """)
            
            result = conn.execute(query, {"target_date": date.date()}).fetchone()
            
            if result:
                return {
                    "total_stocks": result.total_stocks or 0,
                    "red_flag_stocks": result.red_flag_stocks or 0,
                    "red_flag_rate": result.red_flag_rate or 0.0,
                    "common_red_flags": result.common_red_flags or ""
                }
            else:
                return {
                    "total_stocks": 0,
                    "red_flag_stocks": 0,
                    "red_flag_rate": 0.0,
                    "common_red_flags": ""
                }
                
        except Exception as e:
            self.logger.error(f"Red flag analysis failed: {e}")
            return {
                "total_stocks": 0,
                "red_flag_stocks": 0,
                "red_flag_rate": 0.0,
                "common_red_flags": "",
                "error": str(e)
            }

    def _analyze_pillar_distribution(self, conn, date: datetime) -> Dict[str, Any]:
        """上位50銘柄のピラー別スコア分布を分析"""
        try:
            query = text("""
                SELECT 
                    AVG(value_score) as avg_value, STDDEV(value_score) as std_value,
                    AVG(growth_score) as avg_growth, STDDEV(growth_score) as std_growth,
                    AVG(quality_score) as avg_quality, STDDEV(quality_score) as std_quality,
                    AVG(momentum_score) as avg_momentum, STDDEV(momentum_score) as std_momentum,
                    AVG(risk_score) as avg_risk, STDDEV(risk_score) as std_risk,
                    MIN(total_score) as min_total, MAX(total_score) as max_total,
                    AVG(total_score) as avg_total, STDDEV(total_score) as std_total
                FROM (
                    SELECT 
                        COALESCE(value_score, 0) as value_score,
                        COALESCE(growth_score, 0) as growth_score,
                        COALESCE(quality_score, 0) as quality_score,
                        COALESCE(momentum_score, 0) as momentum_score,
                        COALESCE(risk_score, 0) as risk_score,
                        COALESCE(total_score, 0) as total_score
                    FROM daily_scores 
                    WHERE date = :target_date 
                    ORDER BY total_score DESC 
                    LIMIT 50
                ) top_50
            """)
            
            result = conn.execute(query, {"target_date": date.date()}).fetchone()
            
            if result:
                return {
                    "value_pillar": {"avg": result.avg_value or 0, "std": result.std_value or 0},
                    "growth_pillar": {"avg": result.avg_growth or 0, "std": result.std_growth or 0},
                    "quality_pillar": {"avg": result.avg_quality or 0, "std": result.std_quality or 0},
                    "momentum_pillar": {"avg": result.avg_momentum or 0, "std": result.std_momentum or 0},
                    "risk_pillar": {"avg": result.avg_risk or 0, "std": result.std_risk or 0},
                    "total_score": {"min": result.min_total or 0, "max": result.max_total or 0,
                                  "avg": result.avg_total or 0, "std": result.std_total or 0}
                }
            else:
                return {
                    "value_pillar": {"avg": 0, "std": 0},
                    "growth_pillar": {"avg": 0, "std": 0},
                    "quality_pillar": {"avg": 0, "std": 0},
                    "momentum_pillar": {"avg": 0, "std": 0},
                    "risk_pillar": {"avg": 0, "std": 0},
                    "total_score": {"min": 0, "max": 0, "avg": 0, "std": 0}
                }
                
        except Exception as e:
            self.logger.error(f"Pillar distribution analysis failed: {e}")
            return {
                "error": str(e),
                "value_pillar": {"avg": 0, "std": 0},
                "growth_pillar": {"avg": 0, "std": 0},
                "quality_pillar": {"avg": 0, "std": 0},
                "momentum_pillar": {"avg": 0, "std": 0},
                "risk_pillar": {"avg": 0, "std": 0},
                "total_score": {"min": 0, "max": 0, "avg": 0, "std": 0}
            }

    def _analyze_sector_concentration(self, conn, date: datetime) -> Dict[str, Any]:
        """上位50銘柄のセクター偏重を分析"""
        try:
            query = text("""
                WITH sector_distribution AS (
                    SELECT 
                        COALESCE(si.sector, 'Unknown') as sector,
                        COUNT(*) as stock_count,
                        ROUND(COUNT(*) * 100.0 / 50, 2) as percentage,
                        AVG(ds.total_score) as avg_score
                    FROM (
                        SELECT symbol, total_score
                        FROM daily_scores 
                        WHERE date = :target_date 
                        ORDER BY total_score DESC 
                        LIMIT 50
                    ) ds
                    LEFT JOIN stock_info si ON ds.symbol = si.symbol
                    GROUP BY si.sector
                )
                SELECT 
                    sector,
                    stock_count,
                    percentage,
                    avg_score,
                    CASE 
                        WHEN percentage > 35 THEN 'EMERGENCY'
                        WHEN percentage > 25 THEN 'CRITICAL'
                        WHEN percentage > 20 THEN 'WARNING' 
                        ELSE 'NORMAL'
                    END as alert_level
                FROM sector_distribution
                ORDER BY stock_count DESC
            """)
            
            results = conn.execute(query, {"target_date": date.date()}).fetchall()
            
            sector_data = []
            max_percentage = 0
            dominant_sector = "Unknown"
            
            for row in results:
                sector_info = {
                    "sector": row.sector,
                    "stock_count": row.stock_count,
                    "percentage": row.percentage,
                    "avg_score": row.avg_score or 0,
                    "alert_level": row.alert_level
                }
                sector_data.append(sector_info)
                
                if row.percentage > max_percentage:
                    max_percentage = row.percentage
                    dominant_sector = row.sector
            
            return {
                "sector_distribution": sector_data,
                "max_percentage": max_percentage,
                "dominant_sector": dominant_sector,
                "sector_count": len(sector_data),
                "concentration_risk": "HIGH" if max_percentage > 30 else "MEDIUM" if max_percentage > 20 else "LOW"
            }
            
        except Exception as e:
            self.logger.error(f"Sector concentration analysis failed: {e}")
            return {
                "error": str(e),
                "sector_distribution": [],
                "max_percentage": 0,
                "dominant_sector": "Unknown",
                "sector_count": 0,
                "concentration_risk": "UNKNOWN"
            }

    def _analyze_score_drivers(self, conn, date: datetime) -> Dict[str, Any]:
        """上位50銘柄のスコアドライバーを分析"""
        try:
            query = text("""
                WITH driver_analysis AS (
                    SELECT 
                        symbol,
                        COALESCE(value_score, 0) as value_score,
                        COALESCE(growth_score, 0) as growth_score,
                        COALESCE(quality_score, 0) as quality_score,
                        COALESCE(momentum_score, 0) as momentum_score,
                        COALESCE(risk_score, 0) as risk_score,
                        CASE 
                            WHEN COALESCE(value_score, 0) = GREATEST(
                                COALESCE(value_score, 0), COALESCE(growth_score, 0), 
                                COALESCE(quality_score, 0), COALESCE(momentum_score, 0), 
                                COALESCE(risk_score, 0)) THEN 'Value'
                            WHEN COALESCE(growth_score, 0) = GREATEST(
                                COALESCE(value_score, 0), COALESCE(growth_score, 0), 
                                COALESCE(quality_score, 0), COALESCE(momentum_score, 0), 
                                COALESCE(risk_score, 0)) THEN 'Growth'
                            WHEN COALESCE(quality_score, 0) = GREATEST(
                                COALESCE(value_score, 0), COALESCE(growth_score, 0), 
                                COALESCE(quality_score, 0), COALESCE(momentum_score, 0), 
                                COALESCE(risk_score, 0)) THEN 'Quality'
                            WHEN COALESCE(momentum_score, 0) = GREATEST(
                                COALESCE(value_score, 0), COALESCE(growth_score, 0), 
                                COALESCE(quality_score, 0), COALESCE(momentum_score, 0), 
                                COALESCE(risk_score, 0)) THEN 'Momentum'
                            ELSE 'Risk'
                        END as primary_driver
                    FROM (
                        SELECT * FROM daily_scores 
                        WHERE date = :target_date 
                        ORDER BY total_score DESC 
                        LIMIT 50
                    ) top_50
                )
                SELECT 
                    primary_driver,
                    COUNT(*) as stock_count,
                    ROUND(COUNT(*) * 100.0 / 50, 2) as percentage
                FROM driver_analysis
                GROUP BY primary_driver
                ORDER BY stock_count DESC
            """)
            
            results = conn.execute(query, {"target_date": date.date()}).fetchall()
            
            driver_data = []
            max_percentage = 0
            dominant_driver = "Unknown"
            
            for row in results:
                driver_info = {
                    "driver": row.primary_driver,
                    "stock_count": row.stock_count,
                    "percentage": row.percentage
                }
                driver_data.append(driver_info)
                
                if row.percentage > max_percentage:
                    max_percentage = row.percentage
                    dominant_driver = row.primary_driver
            
            return {
                "driver_distribution": driver_data,
                "max_percentage": max_percentage,
                "dominant_driver": dominant_driver,
                "monopoly_risk": "HIGH" if max_percentage > 40 else "MEDIUM" if max_percentage > 30 else "LOW"
            }
            
        except Exception as e:
            self.logger.error(f"Score driver analysis failed: {e}")
            return {
                "error": str(e),
                "driver_distribution": [],
                "max_percentage": 0,
                "dominant_driver": "Unknown",
                "monopoly_risk": "UNKNOWN"
            }

    def _calculate_score_statistics(self, conn, date: datetime) -> Dict[str, Any]:
        """スコア統計を計算"""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as total_stocks,
                    MIN(total_score) as min_score,
                    MAX(total_score) as max_score,
                    AVG(total_score) as avg_score,
                    STDDEV(total_score) as std_score,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_score) as q1_score,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_score) as median_score,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_score) as q3_score,
                    (SELECT MAX(total_score) - MIN(total_score) 
                     FROM (SELECT total_score FROM daily_scores 
                           WHERE date = :target_date ORDER BY total_score DESC LIMIT 50) top_50) as top50_range
                FROM daily_scores 
                WHERE date = :target_date AND total_score IS NOT NULL
            """)
            
            result = conn.execute(query, {"target_date": date.date()}).fetchone()
            
            if result:
                return {
                    "total_stocks": result.total_stocks or 0,
                    "min_score": result.min_score or 0,
                    "max_score": result.max_score or 0,
                    "avg_score": result.avg_score or 0,
                    "std_score": result.std_score or 0,
                    "q1_score": result.q1_score or 0,
                    "median_score": result.median_score or 0,
                    "q3_score": result.q3_score or 0,
                    "top50_range": result.top50_range or 0,
                    "compression_risk": "HIGH" if (result.top50_range or 0) < 5 else "MEDIUM" if (result.top50_range or 0) < 10 else "LOW"
                }
            else:
                return {
                    "total_stocks": 0,
                    "min_score": 0,
                    "max_score": 0,
                    "avg_score": 0,
                    "std_score": 0,
                    "q1_score": 0,
                    "median_score": 0,
                    "q3_score": 0,
                    "top50_range": 0,
                    "compression_risk": "UNKNOWN"
                }
                
        except Exception as e:
            self.logger.error(f"Score statistics calculation failed: {e}")
            return {
                "error": str(e),
                "total_stocks": 0,
                "min_score": 0,
                "max_score": 0,
                "avg_score": 0,
                "std_score": 0,
                "q1_score": 0,
                "median_score": 0,
                "q3_score": 0,
                "top50_range": 0,
                "compression_risk": "UNKNOWN"
            }

    def _generate_daily_alerts(self, results: DailyValidationResults) -> List[ValidationAlert]:
        """日次検証結果からアラートを生成"""
        alerts = []
        
        try:
            # 赤旗混入率アラート
            red_flag_rate = results.red_flag_analysis.get("red_flag_rate", 0)
            alerts.extend(self._check_threshold_alert(
                "red_flag_contamination", red_flag_rate, 
                f"上位50銘柄の赤旗混入率が {red_flag_rate}% です",
                "赤旗混入率", "validation_quality"
            ))
            
            # セクター偏重アラート
            max_sector_pct = results.sector_analysis.get("max_percentage", 0)
            dominant_sector = results.sector_analysis.get("dominant_sector", "Unknown")
            alerts.extend(self._check_threshold_alert(
                "sector_concentration", max_sector_pct,
                f"セクター偏重を検出: {dominant_sector}が{max_sector_pct}%を占めています",
                "セクター偏重", "sector_analysis"
            ))
            
            # ドライバー偏重アラート
            max_driver_pct = results.driver_analysis.get("max_percentage", 0)
            dominant_driver = results.driver_analysis.get("dominant_driver", "Unknown")
            alerts.extend(self._check_threshold_alert(
                "driver_monopoly", max_driver_pct,
                f"ドライバー偏重: {dominant_driver}が{max_driver_pct}%の銘柄で主要因子",
                "ドライバー偏重", "driver_analysis"
            ))
            
            # スコア圧縮アラート
            top50_range = results.score_statistics.get("top50_range", 0)
            alerts.extend(self._check_threshold_alert(
                "score_compression", top50_range,
                f"スコア圧縮を検出: 上位50の点差が{top50_range}点のみ",
                "スコア圧縮", "score_distribution", 
                reverse_threshold=True  # 低い値ほど問題
            ))
            
        except Exception as e:
            self.logger.error(f"Daily alert generation failed: {e}")
            alerts.append(ValidationAlert(
                timestamp=datetime.now(),
                alert_type="system_error",
                level=AlertLevel.CRITICAL,
                message=f"アラート生成システムエラー: {str(e)}",
                value=0,
                threshold=0,
                recommended_action="システム管理者に連絡してください",
                dashboard_section="system_status"
            ))
        
        return alerts

    def _check_threshold_alert(self, alert_type: str, value: float, message: str, 
                             metric_name: str, section: str, reverse_threshold: bool = False) -> List[ValidationAlert]:
        """閾値ベースのアラートチェック"""
        alerts = []
        
        if alert_type not in self.daily_alert_conditions:
            return alerts
        
        conditions = self.daily_alert_conditions[alert_type]
        warning_threshold = conditions["warning_threshold"]
        critical_threshold = conditions["critical_threshold"]
        emergency_threshold = conditions["emergency_threshold"]
        
        # 推奨アクションの定義
        action_map = {
            "red_flag_contamination": {
                AlertLevel.WARNING: "赤旗ルールの見直しを検討してください",
                AlertLevel.CRITICAL: "スコアリングロジックの確認が必要です",
                AlertLevel.EMERGENCY: "緊急: 即座にシステムの見直しが必要です"
            },
            "sector_concentration": {
                AlertLevel.WARNING: "セクター分散の改善を検討してください",
                AlertLevel.CRITICAL: "セクター偏重の原因分析が必要です", 
                AlertLevel.EMERGENCY: "緊急: セクターバランス調整が必要です"
            },
            "driver_monopoly": {
                AlertLevel.WARNING: "ピラーバランスの確認を推奨します",
                AlertLevel.CRITICAL: "ピラー重み配分の見直しが必要です",
                AlertLevel.EMERGENCY: "緊急: ピラー設計の根本見直しが必要です"
            },
            "score_compression": {
                AlertLevel.WARNING: "スコア分散の改善を検討してください",
                AlertLevel.CRITICAL: "正規化手法の見直しが必要です",
                AlertLevel.EMERGENCY: "緊急: スコア計算ロジックの確認が必要です"
            }
        }
        
        # 閾値判定（通常は値が大きいほど問題、reverse_thresholdの場合は逆）
        if not reverse_threshold:
            if value >= emergency_threshold:
                level = AlertLevel.EMERGENCY
            elif value >= critical_threshold:
                level = AlertLevel.CRITICAL
            elif value >= warning_threshold:
                level = AlertLevel.WARNING
            else:
                return alerts
        else:
            if value <= emergency_threshold:
                level = AlertLevel.EMERGENCY
            elif value <= critical_threshold:
                level = AlertLevel.CRITICAL
            elif value <= warning_threshold:
                level = AlertLevel.WARNING
            else:
                return alerts
        
        alert = ValidationAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            level=level,
            message=message,
            value=value,
            threshold=warning_threshold if level == AlertLevel.WARNING 
                     else critical_threshold if level == AlertLevel.CRITICAL 
                     else emergency_threshold,
            recommended_action=action_map.get(alert_type, {}).get(level, "担当者にご相談ください"),
            dashboard_section=section
        )
        
        alerts.append(alert)
        return alerts

    def _perform_decile_backtest(self, conn, month: datetime) -> Dict[str, Any]:
        """デシル・バックテスト分析を実行"""
        try:
            # 実装は簡略化 - 実際の株価リターンデータが必要
            query = text("""
                SELECT 
                    'decile_backtest' as analysis_type,
                    'placeholder' as status
            """)
            
            result = conn.execute(query).fetchone()
            
            return {
                "status": "placeholder_implementation",
                "top_decile_return": 0.0,
                "bottom_decile_return": 0.0,
                "excess_return_diff": 0.0,
                "information_ratio": 0.0,
                "win_rate": 0.0,
                "note": "実装には株価リターンデータの統合が必要"
            }
            
        except Exception as e:
            self.logger.error(f"Decile backtest failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _analyze_pillar_contribution(self, conn, month: datetime) -> Dict[str, Any]:
        """ピラー寄与度分析を実行"""
        try:
            # 実装は簡略化 - 実際のリターンデータとの相関分析が必要
            return {
                "status": "placeholder_implementation",
                "value_correlation": 0.0,
                "growth_correlation": 0.0,
                "quality_correlation": 0.0,
                "momentum_correlation": 0.0,
                "risk_correlation": 0.0,
                "note": "実装にはリターンデータとの相関分析が必要"
            }
            
        except Exception as e:
            self.logger.error(f"Pillar contribution analysis failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _calculate_stability_metrics(self, conn, month: datetime) -> Dict[str, Any]:
        """安定性指標を計算"""
        try:
            # スコア分散の推移分析
            query = text("""
                SELECT 
                    AVG(total_score) as avg_total_score,
                    STDDEV(total_score) as score_stddev,
                    MIN(total_score) as min_score,
                    MAX(total_score) as max_score,
                    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY total_score) as p90_score,
                    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY total_score) as p10_score
                FROM daily_scores 
                WHERE DATE_TRUNC('month', date) = DATE_TRUNC('month', :target_month::date)
                  AND total_score IS NOT NULL
            """)
            
            result = conn.execute(query, {"target_month": month}).fetchone()
            
            if result:
                return {
                    "avg_total_score": result.avg_total_score or 0,
                    "score_stddev": result.score_stddev or 0,
                    "min_score": result.min_score or 0,
                    "max_score": result.max_score or 0,
                    "p90_score": result.p90_score or 0,
                    "p10_score": result.p10_score or 0,
                    "score_range": (result.max_score or 0) - (result.min_score or 0),
                    "stability_score": "GOOD" if (result.score_stddev or 0) < 15 else "MEDIUM" if (result.score_stddev or 0) < 25 else "POOR"
                }
            else:
                return {
                    "avg_total_score": 0,
                    "score_stddev": 0,
                    "min_score": 0,
                    "max_score": 0,
                    "p90_score": 0,
                    "p10_score": 0,
                    "score_range": 0,
                    "stability_score": "UNKNOWN"
                }
                
        except Exception as e:
            self.logger.error(f"Stability metrics calculation failed: {e}")
            return {
                "error": str(e),
                "stability_score": "ERROR"
            }

    def _analyze_turnover(self, conn, month: datetime) -> Dict[str, Any]:
        """入れ替わり率を分析"""
        try:
            # 実装は簡略化 - 月末間の上位銘柄比較が必要
            return {
                "status": "placeholder_implementation",
                "turnover_rate": 0.0,
                "new_entries": 0,
                "dropped_stocks": 0,
                "stability_rating": "UNKNOWN",
                "note": "実装には月次の上位銘柄履歴データが必要"
            }
            
        except Exception as e:
            self.logger.error(f"Turnover analysis failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def _analyze_correlations(self, conn, month: datetime) -> Dict[str, Any]:
        """相関分析を実行"""
        try:
            # ピラー間相関の分析
            query = text("""
                SELECT 
                    CORR(value_score, growth_score) as value_growth_corr,
                    CORR(value_score, quality_score) as value_quality_corr,
                    CORR(value_score, momentum_score) as value_momentum_corr,
                    CORR(growth_score, quality_score) as growth_quality_corr,
                    CORR(growth_score, momentum_score) as growth_momentum_corr,
                    CORR(quality_score, momentum_score) as quality_momentum_corr
                FROM daily_scores
                WHERE DATE_TRUNC('month', date) = DATE_TRUNC('month', :target_month::date)
                  AND value_score IS NOT NULL 
                  AND growth_score IS NOT NULL
                  AND quality_score IS NOT NULL
                  AND momentum_score IS NOT NULL
            """)
            
            result = conn.execute(query, {"target_month": month}).fetchone()
            
            if result:
                correlations = {
                    "value_growth": result.value_growth_corr or 0,
                    "value_quality": result.value_quality_corr or 0,
                    "value_momentum": result.value_momentum_corr or 0,
                    "growth_quality": result.growth_quality_corr or 0,
                    "growth_momentum": result.growth_momentum_corr or 0,
                    "quality_momentum": result.quality_momentum_corr or 0
                }
                
                # 相関の健全性チェック
                avg_correlation = np.mean(list(correlations.values()))
                max_correlation = max(correlations.values())
                
                return {
                    "pillar_correlations": correlations,
                    "avg_correlation": avg_correlation,
                    "max_correlation": max_correlation,
                    "correlation_health": "GOOD" if max_correlation < 0.7 and avg_correlation < 0.3 
                                        else "MEDIUM" if max_correlation < 0.85 
                                        else "POOR"
                }
            else:
                return {
                    "pillar_correlations": {},
                    "avg_correlation": 0,
                    "max_correlation": 0,
                    "correlation_health": "UNKNOWN"
                }
                
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            return {
                "error": str(e),
                "correlation_health": "ERROR"
            }

    def _generate_monthly_alerts(self, results: MonthlyValidationResults) -> List[ValidationAlert]:
        """月次評価結果からアラートを生成"""
        alerts = []
        
        try:
            # 実装は簡略化 - 実際のバックテストデータが必要
            # プレースホルダーとして基本的な安定性チェックのみ実装
            
            stability = results.stability_metrics.get("stability_score", "UNKNOWN")
            if stability == "POOR":
                alerts.append(ValidationAlert(
                    timestamp=datetime.now(),
                    alert_type="stability_concern",
                    level=AlertLevel.WARNING,
                    message=f"月次安定性指標が低下しています: {stability}",
                    value=0,
                    threshold=0,
                    recommended_action="スコア変動の要因分析を実施してください",
                    dashboard_section="stability_metrics"
                ))
                
        except Exception as e:
            self.logger.error(f"Monthly alert generation failed: {e}")
            
        return alerts

    def generate_daily_html_section(self, results: DailyValidationResults) -> str:
        """日次検証結果のHTML セクションを生成"""
        try:
            alert_count = len(results.alerts)
            critical_alerts = [a for a in results.alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
            
            html_content = f"""
            <div class="validation-section">
                <h3>📊 スコアリング品質チェック ({results.date.strftime('%Y-%m-%d')})</h3>
                
                <div class="validation-metrics row">
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h4>赤旗混入率</h4>
                            <span class="metric-value {self._get_alert_css_class(results.red_flag_analysis.get('red_flag_rate', 0), 'red_flag_contamination')}">
                                {results.red_flag_analysis.get('red_flag_rate', 0):.1f}%
                            </span>
                            <small>上位50銘柄中 {results.red_flag_analysis.get('red_flag_stocks', 0)} 銘柄</small>
                        </div>
                    </div>
                    
                    <div class="col-md-3">  
                        <div class="metric-card">
                            <h4>セクター分散</h4>
                            <span class="metric-value">
                                {results.sector_analysis.get('sector_count', 0)} セクター
                            </span>
                            <small>最大: {results.sector_analysis.get('dominant_sector', 'Unknown')} ({results.sector_analysis.get('max_percentage', 0):.1f}%)</small>
                        </div>
                    </div>
                    
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h4>スコア範囲</h4>
                            <span class="metric-value">
                                {results.score_statistics.get('top50_range', 0):.1f}pt
                            </span>
                            <small>上位50銘柄の点差</small>
                        </div>
                    </div>
                    
                    <div class="col-md-3">
                        <div class="alert-summary">
                            <h4>⚠️ アラート状況</h4>
                            <div class="alert-count">
                                <span class="badge badge-{('danger' if critical_alerts else 'warning' if alert_count else 'success')}">{alert_count} 件</span>
                                {('<small>重要: ' + str(len(critical_alerts)) + ' 件</small>') if critical_alerts else ''}
                            </div>
                        </div>
                    </div>
                </div>
                
                {self._generate_alert_html_list(results.alerts)}
                
                <div class="validation-details">
                    <h4>詳細分析</h4>
                    <div class="row">
                        <div class="col-md-6">
                            <h5>ピラー別平均スコア</h5>
                            <ul class="pillar-scores">
                                <li>Value: {results.pillar_distribution.get('value_pillar', {}).get('avg', 0):.1f}点</li>
                                <li>Growth: {results.pillar_distribution.get('growth_pillar', {}).get('avg', 0):.1f}点</li>
                                <li>Quality: {results.pillar_distribution.get('quality_pillar', {}).get('avg', 0):.1f}点</li>
                                <li>Momentum: {results.pillar_distribution.get('momentum_pillar', {}).get('avg', 0):.1f}点</li>
                                <li>Risk: {results.pillar_distribution.get('risk_pillar', {}).get('avg', 0):.1f}点</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h5>主要ドライバー分布</h5>
                            <ul class="driver-distribution">
                            {"".join([f"<li>{driver['driver']}: {driver['stock_count']}銘柄 ({driver['percentage']:.1f}%)</li>" 
                                     for driver in results.driver_analysis.get('driver_distribution', [])])}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            """
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"HTML generation failed: {e}")
            return f'<div class="error">検証レポート生成エラー: {str(e)}</div>'

    def _get_alert_css_class(self, value: float, alert_type: str) -> str:
        """アラートレベルに応じたCSSクラスを取得"""
        if alert_type not in self.daily_alert_conditions:
            return "text-success"
        
        conditions = self.daily_alert_conditions[alert_type]
        
        if value >= conditions.get("emergency_threshold", 999):
            return "text-danger font-weight-bold"
        elif value >= conditions.get("critical_threshold", 999):
            return "text-danger"
        elif value >= conditions.get("warning_threshold", 999):
            return "text-warning"
        else:
            return "text-success"

    def _generate_alert_html_list(self, alerts: List[ValidationAlert]) -> str:
        """アラートリストのHTML を生成"""
        if not alerts:
            return '<div class="alert alert-success">✅ 本日は問題ありません</div>'
        
        html_parts = ['<div class="alerts-section"><h5>📋 検出された問題</h5><ul class="alert-list">']
        
        for alert in alerts:
            css_class = {
                AlertLevel.INFO: "text-info",
                AlertLevel.WARNING: "text-warning", 
                AlertLevel.CRITICAL: "text-danger",
                AlertLevel.EMERGENCY: "text-danger font-weight-bold"
            }.get(alert.level, "text-secondary")
            
            icon = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨", 
                AlertLevel.EMERGENCY: "🔥"
            }.get(alert.level, "❓")
            
            html_parts.append(f"""
                <li class="alert-item {css_class}">
                    <strong>{icon} {alert.message}</strong><br>
                    <small>推奨アクション: {alert.recommended_action}</small>
                </li>
            """)
        
        html_parts.append('</ul></div>')
        return ''.join(html_parts)

    def log_validation_results(self, results: DailyValidationResults) -> None:
        """検証結果をログに記録"""
        try:
            self.logger.info(f"Daily validation results for {results.date.strftime('%Y-%m-%d')}:")
            self.logger.info(f"  Red flag rate: {results.red_flag_analysis.get('red_flag_rate', 0):.1f}%")
            self.logger.info(f"  Sector concentration: {results.sector_analysis.get('max_percentage', 0):.1f}%")
            self.logger.info(f"  Score range: {results.score_statistics.get('top50_range', 0):.1f} points")
            self.logger.info(f"  Alerts generated: {len(results.alerts)}")
            
            if results.alerts:
                for alert in results.alerts:
                    self.logger.warning(f"  ALERT [{alert.level.value}]: {alert.message}")
                    
        except Exception as e:
            self.logger.error(f"Failed to log validation results: {e}")


def main():
    """テスト実行用のメイン関数"""
    try:
        print("🔍 スコアリング検証システムのテスト実行")
        
        validator = ScoringValidator()
        
        # 日次検証のテスト実行
        target_date = "2025-09-11"
        print(f"\n📅 日次検証実行: {target_date}")
        
        daily_results = validator.run_daily_validation(target_date)
        
        print(f"✅ 日次検証完了:")
        print(f"   赤旗混入率: {daily_results.red_flag_analysis.get('red_flag_rate', 0):.1f}%")
        print(f"   セクター偏重: {daily_results.sector_analysis.get('max_percentage', 0):.1f}% ({daily_results.sector_analysis.get('dominant_sector', 'Unknown')})")
        print(f"   アラート数: {len(daily_results.alerts)}")
        
        # アラートの表示
        if daily_results.alerts:
            print(f"\n⚠️  検出されたアラート:")
            for alert in daily_results.alerts:
                print(f"   [{alert.level.value.upper()}] {alert.message}")
        
        # HTMLセクション生成のテスト
        html_section = validator.generate_daily_html_section(daily_results)
        print(f"\n📄 HTML セクション生成完了 ({len(html_section)} 文字)")
        
        print(f"\n✅ 検証システムのテスト実行完了")
        
    except Exception as e:
        print(f"❌ テスト実行失敗: {e}")
        raise


if __name__ == "__main__":
    main()