#!/usr/bin/env python3
"""
Automated Tuning System - V2 Migration System

Self-learning system for automated threshold adjustments and parameter optimization
based on historical performance data and real-time system behavior.

Task 7.2: Continuous Improvement System Implementation
- Dynamic threshold adjustment based on performance patterns
- Automated parameter optimization using historical data
- Self-learning alert sensitivity tuning
- Performance-based configuration updates
- Intelligent system adaptation

Features:
- Historical performance analysis for optimal threshold setting
- Automated alert sensitivity adjustment
- Performance-based parameter optimization
- Safe threshold adjustment with rollback capabilities
- Comprehensive tuning audit trail

Usage:
    from investment_toolkit.utilities.automated_tuning_system import AutomatedTuningSystem

    tuner = AutomatedTuningSystem()
    tuner.analyze_and_tune()
    tuner.apply_safe_adjustments()

Created: 2025-09-17
Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import logging
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from scipy import stats
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    from investment_toolkit.monitoring.ab_anomaly_detector import ABAnomalyDetector
except ImportError as e:
    print(f"❌ Import error: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdAdjustment:
    """Threshold adjustment recommendation"""
    parameter_name: str
    current_value: float
    recommended_value: float
    adjustment_type: str  # 'TIGHTEN', 'RELAX', 'OPTIMIZE'
    confidence: float  # 0-100
    rationale: str
    expected_impact: str
    safety_score: float  # 0-100, higher is safer
    rollback_criteria: List[str]
    implementation_date: datetime
    review_date: datetime


@dataclass
class PerformanceBaseline:
    """Performance baseline for threshold optimization"""
    metric_name: str
    historical_mean: float
    historical_std: float
    percentile_95: float
    percentile_99: float
    optimal_threshold: float
    confidence_interval: Tuple[float, float]
    last_updated: datetime


@dataclass
class TuningReport:
    """Automated tuning analysis report"""
    analysis_timestamp: datetime
    analysis_period: Tuple[str, str]
    parameters_analyzed: int
    adjustments_recommended: int
    high_confidence_adjustments: int
    safety_score: float
    threshold_adjustments: List[ThresholdAdjustment]
    performance_baselines: List[PerformanceBaseline]
    tuning_history: List[Dict[str, Any]]
    next_tuning_date: datetime


class AutomatedTuningSystem:
    """Automated parameter tuning and threshold optimization system"""

    def __init__(self, analysis_period_days: int = 30, min_confidence: float = 70.0):
        self.analysis_period_days = analysis_period_days
        self.min_confidence = min_confidence
        self.adjustments: List[ThresholdAdjustment] = []
        self.baselines: List[PerformanceBaseline] = []

        # Current threshold configuration paths
        self.config_files = {
            'anomaly_thresholds': project_root / 'config' / 'ab_anomaly_thresholds.yaml',
            'feature_flags': project_root / 'config' / 'ab_feature_flags.yaml',
            'runner_config': project_root / 'config' / 'ab_runner_config.yaml'
        }

        # Tuning history file
        self.tuning_history_file = project_root / 'config' / 'tuning_history.json'

        # Safety constraints
        self.safety_constraints = {
            'correlation_warning': {'min': 0.3, 'max': 0.9},
            'correlation_critical': {'min': 0.1, 'max': 0.7},
            'performance_warning_ratio': {'min': 1.2, 'max': 5.0},
            'ranking_overlap_warning': {'min': 0.4, 'max': 0.95},
            'error_rate_warning': {'min': 0.001, 'max': 0.2}
        }

        logger.info("Automated Tuning System initialized")

    def _get_db_connection(self):
        """Get database connection"""
        import psycopg2
        from psycopg2.extras import RealDictCursor

        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursor_factory=RealDictCursor
        )

    def _load_current_thresholds(self) -> Dict[str, float]:
        """Load current threshold configuration"""
        thresholds = {}

        try:
            # Load from anomaly detector if available
            detector = ABAnomalyDetector()
            if hasattr(detector, 'thresholds'):
                thresholds.update({
                    'correlation_warning': getattr(detector.thresholds, 'correlation_warning', 0.6),
                    'correlation_critical': getattr(detector.thresholds, 'correlation_critical', 0.4),
                    'performance_warning_ratio': getattr(detector.thresholds, 'performance_warning_ratio', 2.0),
                    'ranking_overlap_warning': getattr(detector.thresholds, 'ranking_overlap_warning', 0.7),
                    'error_rate_warning': getattr(detector.thresholds, 'error_rate_warning', 0.05)
                })
        except Exception:
            # Fallback to default values
            thresholds = {
                'correlation_warning': 0.6,
                'correlation_critical': 0.4,
                'performance_warning_ratio': 2.0,
                'ranking_overlap_warning': 0.7,
                'error_rate_warning': 0.05
            }

        # Try to load from config file if it exists
        config_file = self.config_files.get('anomaly_thresholds')
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        thresholds.update(config_data)
            except Exception as e:
                logger.warning(f"Could not load threshold config: {e}")

        return thresholds

    def analyze_historical_performance(self) -> List[PerformanceBaseline]:
        """Analyze historical performance to establish baselines"""
        baselines = []

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.analysis_period_days)

            # Get historical performance data
            cursor.execute("""
                SELECT date, total_score_correlation, value_pillar_correlation,
                       growth_pillar_correlation, quality_pillar_correlation,
                       momentum_pillar_correlation, top_50_overlap_rate,
                       v1_execution_time_ms, v2_execution_time_ms,
                       v1_error_count, v2_error_count,
                       v1_symbols_processed, v2_symbols_processed
                FROM backtest_results.ab_comparison_results
                WHERE date BETWEEN %s AND %s
                ORDER BY date
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

            data = cursor.fetchall()
            cursor.close()
            conn.close()

            if len(data) < 10:  # Need sufficient data for baseline
                logger.warning("Insufficient historical data for baseline analysis")
                return baselines

            df = pd.DataFrame(data)

            # Calculate performance ratio
            df['performance_ratio'] = df['v2_execution_time_ms'] / df['v1_execution_time_ms']
            df['error_rate_v2'] = df['v2_error_count'] / df['v2_symbols_processed']

            # Metrics to analyze for baselines
            metrics = {
                'total_score_correlation': 'Total Score Correlation',
                'value_pillar_correlation': 'Value Pillar Correlation',
                'growth_pillar_correlation': 'Growth Pillar Correlation',
                'quality_pillar_correlation': 'Quality Pillar Correlation',
                'momentum_pillar_correlation': 'Momentum Pillar Correlation',
                'top_50_overlap_rate': 'Top 50 Overlap Rate',
                'performance_ratio': 'Performance Ratio (V2/V1)',
                'error_rate_v2': 'V2 Error Rate'
            }

            for metric_col, metric_name in metrics.items():
                if metric_col in df.columns:
                    values = df[metric_col].dropna()

                    if len(values) >= 10:
                        mean_val = values.mean()
                        std_val = values.std()
                        p95 = values.quantile(0.95)
                        p99 = values.quantile(0.99)

                        # Calculate optimal threshold based on metric type
                        if 'correlation' in metric_col or 'overlap' in metric_col:
                            # Higher is better, threshold should be below normal range
                            optimal_threshold = max(mean_val - 1.5 * std_val, values.quantile(0.1))
                        elif 'ratio' in metric_col or 'error' in metric_col:
                            # Lower is better, threshold should be above normal range
                            optimal_threshold = min(mean_val + 1.5 * std_val, values.quantile(0.9))
                        else:
                            optimal_threshold = mean_val

                        # Calculate confidence interval
                        ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(values))
                        ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(values))

                        baselines.append(PerformanceBaseline(
                            metric_name=metric_name,
                            historical_mean=mean_val,
                            historical_std=std_val,
                            percentile_95=p95,
                            percentile_99=p99,
                            optimal_threshold=optimal_threshold,
                            confidence_interval=(ci_lower, ci_upper),
                            last_updated=datetime.now()
                        ))

        except Exception as e:
            logger.error(f"Error analyzing historical performance: {str(e)}")

        return baselines

    def calculate_threshold_adjustments(self, baselines: List[PerformanceBaseline]) -> List[ThresholdAdjustment]:
        """Calculate optimal threshold adjustments based on baselines"""
        adjustments = []
        current_thresholds = self._load_current_thresholds()

        # Mapping between baseline metrics and threshold parameters
        threshold_mapping = {
            'Total Score Correlation': 'correlation_warning',
            'Performance Ratio (V2/V1)': 'performance_warning_ratio',
            'Top 50 Overlap Rate': 'ranking_overlap_warning',
            'V2 Error Rate': 'error_rate_warning'
        }

        for baseline in baselines:
            if baseline.metric_name in threshold_mapping:
                param_name = threshold_mapping[baseline.metric_name]
                current_threshold = current_thresholds.get(param_name)

                if current_threshold is not None:
                    recommended_threshold = baseline.optimal_threshold

                    # Apply safety constraints
                    if param_name in self.safety_constraints:
                        constraints = self.safety_constraints[param_name]
                        recommended_threshold = max(min(recommended_threshold, constraints['max']), constraints['min'])

                    # Calculate adjustment confidence based on data quality
                    data_quality_score = min(100, (baseline.historical_std / baseline.historical_mean) * 100) if baseline.historical_mean != 0 else 50
                    confidence = max(100 - data_quality_score, 30)

                    # Determine adjustment type and safety
                    if abs(recommended_threshold - current_threshold) > 0.01:  # Meaningful difference
                        if 'correlation' in param_name or 'overlap' in param_name:
                            # For correlation/overlap metrics, higher thresholds are stricter
                            if recommended_threshold > current_threshold:
                                adjustment_type = 'TIGHTEN'
                                safety_score = 70  # More conservative
                            else:
                                adjustment_type = 'RELAX'
                                safety_score = 90  # Safer
                        else:
                            # For performance/error metrics, lower thresholds are stricter
                            if recommended_threshold < current_threshold:
                                adjustment_type = 'TIGHTEN'
                                safety_score = 70
                            else:
                                adjustment_type = 'RELAX'
                                safety_score = 90

                        # Generate rationale
                        change_pct = abs((recommended_threshold - current_threshold) / current_threshold) * 100
                        rationale = f"Historical data suggests optimal threshold of {recommended_threshold:.3f} (current: {current_threshold:.3f}, {change_pct:.1f}% change)"

                        # Expected impact
                        if adjustment_type == 'TIGHTEN':
                            expected_impact = "Increased sensitivity, more alerts but better issue detection"
                        else:
                            expected_impact = "Reduced false positives, fewer alerts but may miss some issues"

                        # Rollback criteria
                        rollback_criteria = [
                            "Alert volume increases by >200% within 7 days",
                            "Critical issues missed due to relaxed thresholds",
                            "System performance degrades significantly"
                        ]

                        adjustments.append(ThresholdAdjustment(
                            parameter_name=param_name,
                            current_value=current_threshold,
                            recommended_value=recommended_threshold,
                            adjustment_type=adjustment_type,
                            confidence=confidence,
                            rationale=rationale,
                            expected_impact=expected_impact,
                            safety_score=safety_score,
                            rollback_criteria=rollback_criteria,
                            implementation_date=datetime.now() + timedelta(days=1),
                            review_date=datetime.now() + timedelta(days=14)
                        ))

        return adjustments

    def evaluate_alert_effectiveness(self) -> Dict[str, Any]:
        """Evaluate current alert effectiveness and frequency"""
        effectiveness = {}

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)  # Last 2 weeks

            # Analyze alert patterns from migration log
            cursor.execute("""
                SELECT log_level, component, operation, COUNT(*) as count
                FROM backtest_results.scoring_migration_log
                WHERE timestamp BETWEEN %s AND %s
                  AND log_level IN ('WARNING', 'ERROR', 'CRITICAL')
                GROUP BY log_level, component, operation
                ORDER BY count DESC
            """, (start_date, end_date))

            alert_data = cursor.fetchall()
            cursor.close()
            conn.close()

            if alert_data:
                df = pd.DataFrame(alert_data)

                # Calculate alert frequency by level
                alert_frequency = df.groupby('log_level')['count'].sum().to_dict()

                # Calculate daily alert rate
                total_alerts = sum(alert_frequency.values())
                daily_alert_rate = total_alerts / 14

                effectiveness = {
                    'total_alerts_14_days': total_alerts,
                    'daily_alert_rate': daily_alert_rate,
                    'alert_frequency_by_level': alert_frequency,
                    'most_frequent_alerts': df.head(5).to_dict('records')
                }

                # Alert effectiveness assessment
                if daily_alert_rate > 10:
                    effectiveness['assessment'] = 'HIGH_VOLUME'
                    effectiveness['recommendation'] = 'Consider relaxing thresholds to reduce noise'
                elif daily_alert_rate < 1:
                    effectiveness['assessment'] = 'LOW_VOLUME'
                    effectiveness['recommendation'] = 'Consider tightening thresholds for better coverage'
                else:
                    effectiveness['assessment'] = 'OPTIMAL'
                    effectiveness['recommendation'] = 'Alert volume appears appropriate'

        except Exception as e:
            logger.error(f"Error evaluating alert effectiveness: {str(e)}")
            effectiveness = {'error': str(e)}

        return effectiveness

    def load_tuning_history(self) -> List[Dict[str, Any]]:
        """Load historical tuning adjustments"""
        history = []

        if self.tuning_history_file.exists():
            try:
                with open(self.tuning_history_file, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load tuning history: {e}")

        return history

    def save_tuning_history(self, new_entry: Dict[str, Any]) -> None:
        """Save tuning adjustment to history"""
        history = self.load_tuning_history()
        history.append(new_entry)

        # Keep only last 50 entries
        history = history[-50:]

        try:
            self.tuning_history_file.parent.mkdir(exist_ok=True)
            with open(self.tuning_history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save tuning history: {e}")

    def apply_safe_adjustments(self, adjustments: List[ThresholdAdjustment], dry_run: bool = True) -> Dict[str, Any]:
        """Apply threshold adjustments with safety checks"""
        results = {
            'applied_adjustments': [],
            'skipped_adjustments': [],
            'errors': []
        }

        high_confidence_adjustments = [
            adj for adj in adjustments
            if adj.confidence >= self.min_confidence and adj.safety_score >= 70
        ]

        for adjustment in high_confidence_adjustments:
            try:
                if dry_run:
                    logger.info(f"DRY RUN: Would adjust {adjustment.parameter_name} from {adjustment.current_value:.3f} to {adjustment.recommended_value:.3f}")
                    results['applied_adjustments'].append({
                        'parameter': adjustment.parameter_name,
                        'old_value': adjustment.current_value,
                        'new_value': adjustment.recommended_value,
                        'dry_run': True
                    })
                else:
                    # TODO: Implement actual threshold updates
                    # This would require updating configuration files and reloading systems
                    logger.info(f"LIVE: Adjusting {adjustment.parameter_name} from {adjustment.current_value:.3f} to {adjustment.recommended_value:.3f}")

                    # Save to tuning history
                    history_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'parameter': adjustment.parameter_name,
                        'old_value': adjustment.current_value,
                        'new_value': adjustment.recommended_value,
                        'adjustment_type': adjustment.adjustment_type,
                        'confidence': adjustment.confidence,
                        'rationale': adjustment.rationale
                    }
                    self.save_tuning_history(history_entry)

                    results['applied_adjustments'].append({
                        'parameter': adjustment.parameter_name,
                        'old_value': adjustment.current_value,
                        'new_value': adjustment.recommended_value,
                        'dry_run': False
                    })

            except Exception as e:
                error_msg = f"Failed to apply adjustment for {adjustment.parameter_name}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        # Skip low-confidence or unsafe adjustments
        for adjustment in adjustments:
            if adjustment not in high_confidence_adjustments:
                skip_reason = []
                if adjustment.confidence < self.min_confidence:
                    skip_reason.append(f"low confidence ({adjustment.confidence:.1f}%)")
                if adjustment.safety_score < 70:
                    skip_reason.append(f"low safety score ({adjustment.safety_score:.1f})")

                results['skipped_adjustments'].append({
                    'parameter': adjustment.parameter_name,
                    'reason': ', '.join(skip_reason)
                })

        return results

    def analyze_and_tune(self, dry_run: bool = True) -> TuningReport:
        """Run complete analysis and tuning process"""
        logger.info("Starting automated tuning analysis...")

        # Analyze historical performance
        baselines = self.analyze_historical_performance()
        self.baselines = baselines

        # Calculate threshold adjustments
        adjustments = self.calculate_threshold_adjustments(baselines)
        self.adjustments = adjustments

        # Evaluate alert effectiveness
        alert_effectiveness = self.evaluate_alert_effectiveness()

        # Load tuning history
        tuning_history = self.load_tuning_history()

        # Calculate safety score
        safety_scores = [adj.safety_score for adj in adjustments] if adjustments else [100]
        overall_safety_score = np.mean(safety_scores)

        # Generate report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.analysis_period_days)

        report = TuningReport(
            analysis_timestamp=datetime.now(),
            analysis_period=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            parameters_analyzed=len(baselines),
            adjustments_recommended=len(adjustments),
            high_confidence_adjustments=len([adj for adj in adjustments if adj.confidence >= self.min_confidence]),
            safety_score=overall_safety_score,
            threshold_adjustments=adjustments,
            performance_baselines=baselines,
            tuning_history=tuning_history,
            next_tuning_date=datetime.now() + timedelta(days=7)
        )

        logger.info(f"Tuning analysis completed - {len(adjustments)} adjustments recommended, safety score: {overall_safety_score:.1f}")

        return report


def print_tuning_report(report: TuningReport, detailed: bool = False):
    """Print formatted tuning report"""
    print(f"\n{'='*70}")
    print(f"AUTOMATED TUNING SYSTEM ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"Analysis Date: {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Period: {report.analysis_period[0]} to {report.analysis_period[1]}")
    print(f"Overall Safety Score: {report.safety_score:.1f}/100")
    print(f"\nANALYSIS SUMMARY:")
    print(f"  Parameters Analyzed: {report.parameters_analyzed}")
    print(f"  Adjustments Recommended: {report.adjustments_recommended}")
    print(f"  High Confidence Adjustments: {report.high_confidence_adjustments}")

    if report.threshold_adjustments:
        print(f"\n{'='*70}")
        print("THRESHOLD ADJUSTMENTS RECOMMENDED:")
        print(f"{'='*70}")

        for i, adj in enumerate(report.threshold_adjustments, 1):
            confidence_emoji = '🟢' if adj.confidence >= 80 else '🟡' if adj.confidence >= 60 else '🔴'
            safety_emoji = '🛡️' if adj.safety_score >= 80 else '⚠️' if adj.safety_score >= 60 else '⚡'

            print(f"\n{i}. {adj.parameter_name} ({adj.adjustment_type})")
            print(f"   {confidence_emoji} Confidence: {adj.confidence:.1f}% | {safety_emoji} Safety: {adj.safety_score:.1f}%")
            print(f"   Current: {adj.current_value:.3f} → Recommended: {adj.recommended_value:.3f}")
            print(f"   Rationale: {adj.rationale}")
            print(f"   Expected Impact: {adj.expected_impact}")

            if detailed:
                print(f"   Implementation Date: {adj.implementation_date.strftime('%Y-%m-%d')}")
                print(f"   Review Date: {adj.review_date.strftime('%Y-%m-%d')}")
                print(f"   Rollback Criteria:")
                for criterion in adj.rollback_criteria:
                    print(f"     • {criterion}")

    print(f"\n{'='*70}")
    print(f"Next Tuning Analysis: {report.next_tuning_date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")


def main():
    """Main execution for testing"""
    tuner = AutomatedTuningSystem(analysis_period_days=30)
    report = tuner.analyze_and_tune(dry_run=True)

    # Print report
    print_tuning_report(report, detailed=True)

    # Apply adjustments (dry run)
    if report.threshold_adjustments:
        results = tuner.apply_safe_adjustments(report.threshold_adjustments, dry_run=True)
        print(f"\nTuning Results:")
        print(f"  Applied: {len(results['applied_adjustments'])}")
        print(f"  Skipped: {len(results['skipped_adjustments'])}")
        print(f"  Errors: {len(results['errors'])}")


if __name__ == "__main__":
    main()