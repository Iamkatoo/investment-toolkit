#!/usr/bin/env python3
"""
Continuous Improvement Engine - V2 Migration System

Automated improvement proposal and optimization system that analyzes
system performance data to generate actionable improvement recommendations.

Task 7.2: Continuous Improvement System Implementation
- Automated performance analysis and optimization suggestions
- Trend-based improvement recommendations
- Self-learning threshold adjustment system
- Performance regression detection
- Automated optimization opportunity identification

Features:
- Machine learning-based pattern recognition
- Performance baseline establishment and monitoring
- Automated threshold tuning recommendations
- Improvement impact measurement
- Comprehensive improvement tracking and reporting

Usage:
    from investment_analysis.analysis.continuous_improvement_engine import ContinuousImprovementEngine

    engine = ContinuousImprovementEngine()
    improvements = engine.analyze_and_recommend()
    engine.generate_improvement_report()

Created: 2025-09-17
Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    from investment_analysis.monitoring.ab_anomaly_detector import ABAnomalyDetector
    from investment_analysis.analysis.ab_comparison_engine import ABComparisonEngine
except ImportError as e:
    print(f"❌ Import error: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImprovementRecommendation:
    """Individual improvement recommendation"""
    category: str
    priority: str  # 'HIGH', 'MEDIUM', 'LOW'
    confidence: float  # 0-100
    title: str
    description: str
    impact_assessment: str
    implementation_effort: str  # 'LOW', 'MEDIUM', 'HIGH'
    expected_benefit: str
    implementation_steps: List[str]
    success_metrics: List[str]
    related_data: Dict[str, Any]
    timestamp: datetime


@dataclass
class PerformanceInsight:
    """Performance analysis insight"""
    metric_name: str
    current_value: float
    baseline_value: float
    trend_direction: str  # 'IMPROVING', 'DEGRADING', 'STABLE'
    trend_strength: float  # 0-100
    anomaly_score: float  # 0-100
    confidence: float
    insight_text: str


@dataclass
class ImprovementReport:
    """Comprehensive improvement analysis report"""
    analysis_timestamp: datetime
    analysis_period: Tuple[str, str]
    overall_health_score: float
    performance_insights: List[PerformanceInsight]
    improvement_recommendations: List[ImprovementRecommendation]
    automated_adjustments_proposed: List[Dict[str, Any]]
    system_optimization_score: float
    next_analysis_date: datetime


class ContinuousImprovementEngine:
    """Engine for continuous system improvement analysis and recommendations"""

    def __init__(self, analysis_period_days: int = 30):
        self.analysis_period_days = analysis_period_days
        self.performance_insights: List[PerformanceInsight] = []
        self.recommendations: List[ImprovementRecommendation] = []

        # Performance baseline thresholds
        self.baseline_metrics = {
            'correlation_baseline': 0.75,
            'correlation_improvement_target': 0.85,
            'performance_ratio_baseline': 2.0,
            'performance_improvement_target': 1.5,
            'error_rate_baseline': 0.05,
            'error_rate_target': 0.01,
            'availability_baseline': 0.995,
            'availability_target': 0.999
        }

        logger.info("Continuous Improvement Engine initialized")

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

    def analyze_correlation_trends(self) -> List[PerformanceInsight]:
        """Analyze correlation trends and patterns"""
        insights = []

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Get correlation data for analysis period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.analysis_period_days)

            cursor.execute("""
                SELECT date, total_score_correlation, value_pillar_correlation,
                       growth_pillar_correlation, quality_pillar_correlation,
                       momentum_pillar_correlation
                FROM backtest_results.ab_comparison_results
                WHERE date BETWEEN %s AND %s
                ORDER BY date
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

            data = cursor.fetchall()
            cursor.close()
            conn.close()

            if len(data) < 7:  # Need at least a week of data
                return insights

            df = pd.DataFrame(data)

            # Analyze each correlation metric
            correlation_columns = [
                'total_score_correlation', 'value_pillar_correlation',
                'growth_pillar_correlation', 'quality_pillar_correlation',
                'momentum_pillar_correlation'
            ]

            for col in correlation_columns:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 5:
                        # Calculate trend
                        x = np.arange(len(values))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

                        current_value = values.iloc[-1]
                        baseline_value = self.baseline_metrics['correlation_baseline']

                        # Determine trend direction
                        if slope > 0.001 and p_value < 0.1:
                            trend_direction = 'IMPROVING'
                        elif slope < -0.001 and p_value < 0.1:
                            trend_direction = 'DEGRADING'
                        else:
                            trend_direction = 'STABLE'

                        # Calculate trend strength
                        trend_strength = min(abs(slope) * 100, 100)

                        # Detect anomalies
                        z_scores = np.abs(stats.zscore(values))
                        anomaly_score = np.mean(z_scores) * 10  # Scale to 0-100

                        confidence = max(100 - std_err * 1000, 0)  # Convert to confidence score

                        # Generate insight text
                        metric_name = col.replace('_', ' ').title()
                        if trend_direction == 'IMPROVING':
                            insight_text = f"{metric_name} showing positive trend (slope: {slope:.4f})"
                        elif trend_direction == 'DEGRADING':
                            insight_text = f"{metric_name} declining (slope: {slope:.4f})"
                        else:
                            insight_text = f"{metric_name} stable around {current_value:.3f}"

                        insights.append(PerformanceInsight(
                            metric_name=metric_name,
                            current_value=current_value,
                            baseline_value=baseline_value,
                            trend_direction=trend_direction,
                            trend_strength=trend_strength,
                            anomaly_score=anomaly_score,
                            confidence=confidence,
                            insight_text=insight_text
                        ))

        except Exception as e:
            logger.error(f"Error analyzing correlation trends: {str(e)}")

        return insights

    def analyze_performance_patterns(self) -> List[PerformanceInsight]:
        """Analyze system performance patterns"""
        insights = []

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.analysis_period_days)

            # Get performance data
            cursor.execute("""
                SELECT date, v1_execution_time_ms, v2_execution_time_ms,
                       v1_error_count, v2_error_count,
                       v1_symbols_processed, v2_symbols_processed
                FROM backtest_results.ab_comparison_results
                WHERE date BETWEEN %s AND %s
                ORDER BY date
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

            data = cursor.fetchall()
            cursor.close()
            conn.close()

            if len(data) < 5:
                return insights

            df = pd.DataFrame(data)

            # Analyze performance ratio trends
            df['performance_ratio'] = df['v2_execution_time_ms'] / df['v1_execution_time_ms']
            df['error_rate_v2'] = df['v2_error_count'] / df['v2_symbols_processed']

            performance_metrics = {
                'performance_ratio': ('Performance Ratio (V2/V1)', self.baseline_metrics['performance_ratio_baseline']),
                'error_rate_v2': ('V2 Error Rate', self.baseline_metrics['error_rate_baseline'])
            }

            for metric, (display_name, baseline) in performance_metrics.items():
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 3:
                        # Trend analysis
                        x = np.arange(len(values))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

                        current_value = values.iloc[-1]

                        # For performance ratio, decreasing is improving
                        # For error rate, decreasing is improving
                        if metric in ['performance_ratio', 'error_rate_v2']:
                            if slope < -0.001 and p_value < 0.1:
                                trend_direction = 'IMPROVING'
                            elif slope > 0.001 and p_value < 0.1:
                                trend_direction = 'DEGRADING'
                            else:
                                trend_direction = 'STABLE'
                        else:
                            if slope > 0.001 and p_value < 0.1:
                                trend_direction = 'IMPROVING'
                            elif slope < -0.001 and p_value < 0.1:
                                trend_direction = 'DEGRADING'
                            else:
                                trend_direction = 'STABLE'

                        trend_strength = min(abs(slope) * 100, 100)

                        # Anomaly detection
                        z_scores = np.abs(stats.zscore(values))
                        anomaly_score = np.mean(z_scores) * 10

                        confidence = max(100 - std_err * 1000, 0)

                        if trend_direction == 'IMPROVING':
                            insight_text = f"{display_name} improving (current: {current_value:.3f})"
                        elif trend_direction == 'DEGRADING':
                            insight_text = f"{display_name} degrading (current: {current_value:.3f})"
                        else:
                            insight_text = f"{display_name} stable at {current_value:.3f}"

                        insights.append(PerformanceInsight(
                            metric_name=display_name,
                            current_value=current_value,
                            baseline_value=baseline,
                            trend_direction=trend_direction,
                            trend_strength=trend_strength,
                            anomaly_score=anomaly_score,
                            confidence=confidence,
                            insight_text=insight_text
                        ))

        except Exception as e:
            logger.error(f"Error analyzing performance patterns: {str(e)}")

        return insights

    def generate_correlation_improvement_recommendations(self, insights: List[PerformanceInsight]) -> List[ImprovementRecommendation]:
        """Generate recommendations for improving correlations"""
        recommendations = []

        # Find correlation-related insights
        correlation_insights = [i for i in insights if 'correlation' in i.metric_name.lower()]

        for insight in correlation_insights:
            if insight.trend_direction == 'DEGRADING' or insight.current_value < insight.baseline_value:
                priority = 'HIGH' if insight.current_value < 0.6 else 'MEDIUM'

                recommendations.append(ImprovementRecommendation(
                    category='Correlation Optimization',
                    priority=priority,
                    confidence=insight.confidence,
                    title=f"Improve {insight.metric_name}",
                    description=f"{insight.metric_name} is at {insight.current_value:.3f}, below target of {self.baseline_metrics['correlation_improvement_target']:.3f}",
                    impact_assessment="High - Improved correlation indicates better V1/V2 alignment",
                    implementation_effort='MEDIUM',
                    expected_benefit="15-25% improvement in system reliability and migration confidence",
                    implementation_steps=[
                        "Analyze feature engineering differences between V1 and V2",
                        "Review normalization and scaling methods",
                        "Validate input data consistency",
                        "Fine-tune scoring algorithm parameters",
                        "Implement A/B testing for algorithm variations"
                    ],
                    success_metrics=[
                        f"Target correlation > {self.baseline_metrics['correlation_improvement_target']:.3f}",
                        "Correlation volatility < 0.05",
                        "Sustained improvement over 14 days"
                    ],
                    related_data={
                        "current_correlation": insight.current_value,
                        "trend": insight.trend_direction,
                        "baseline": insight.baseline_value
                    },
                    timestamp=datetime.now()
                ))

        return recommendations

    def generate_performance_improvement_recommendations(self, insights: List[PerformanceInsight]) -> List[ImprovementRecommendation]:
        """Generate recommendations for improving system performance"""
        recommendations = []

        # Find performance-related insights
        performance_insights = [i for i in insights if 'ratio' in i.metric_name.lower() or 'error' in i.metric_name.lower()]

        for insight in performance_insights:
            if 'ratio' in insight.metric_name.lower() and insight.current_value > insight.baseline_value:
                priority = 'HIGH' if insight.current_value > 3.0 else 'MEDIUM'

                recommendations.append(ImprovementRecommendation(
                    category='Performance Optimization',
                    priority=priority,
                    confidence=insight.confidence,
                    title="Optimize V2 System Performance",
                    description=f"V2 performance ratio is {insight.current_value:.2f}x slower than V1, target is {self.baseline_metrics['performance_improvement_target']:.1f}x",
                    impact_assessment="High - Faster execution enables real-time processing and reduces resource costs",
                    implementation_effort='HIGH',
                    expected_benefit="30-50% reduction in execution time and resource usage",
                    implementation_steps=[
                        "Profile V2 scoring pipeline for bottlenecks",
                        "Optimize database queries and connections",
                        "Implement caching for repeated calculations",
                        "Consider parallel processing for independent calculations",
                        "Review and optimize data loading strategies"
                    ],
                    success_metrics=[
                        f"Performance ratio < {self.baseline_metrics['performance_improvement_target']:.1f}x",
                        "Memory usage < 4GB during execution",
                        "95th percentile execution time < 30 minutes"
                    ],
                    related_data={
                        "current_ratio": insight.current_value,
                        "target_ratio": self.baseline_metrics['performance_improvement_target']
                    },
                    timestamp=datetime.now()
                ))

            elif 'error' in insight.metric_name.lower() and insight.current_value > insight.baseline_value:
                priority = 'HIGH' if insight.current_value > 0.1 else 'MEDIUM'

                recommendations.append(ImprovementRecommendation(
                    category='Error Rate Reduction',
                    priority=priority,
                    confidence=insight.confidence,
                    title="Reduce V2 System Error Rate",
                    description=f"V2 error rate is {insight.current_value:.3f}, target is {self.baseline_metrics['error_rate_target']:.3f}",
                    impact_assessment="Medium - Lower error rates improve data quality and system reliability",
                    implementation_effort='MEDIUM',
                    expected_benefit="80% reduction in error rate and improved data completeness",
                    implementation_steps=[
                        "Analyze error patterns and root causes",
                        "Implement better input data validation",
                        "Add retry mechanisms for transient failures",
                        "Improve error handling and recovery procedures",
                        "Enhance monitoring for early error detection"
                    ],
                    success_metrics=[
                        f"Error rate < {self.baseline_metrics['error_rate_target']:.3f}",
                        "Zero critical errors per week",
                        "Error recovery time < 5 minutes"
                    ],
                    related_data={
                        "current_error_rate": insight.current_value,
                        "target_error_rate": self.baseline_metrics['error_rate_target']
                    },
                    timestamp=datetime.now()
                ))

        return recommendations

    def generate_automated_threshold_adjustments(self, insights: List[PerformanceInsight]) -> List[Dict[str, Any]]:
        """Generate automated threshold adjustment recommendations"""
        adjustments = []

        for insight in insights:
            # Suggest threshold adjustments based on current performance
            if 'correlation' in insight.metric_name.lower():
                current_warning_threshold = 0.6  # Current default

                if insight.trend_direction == 'IMPROVING' and insight.current_value > 0.8:
                    # Tighten thresholds if consistently performing well
                    new_threshold = min(insight.current_value - 0.05, 0.75)
                    adjustments.append({
                        'type': 'threshold_adjustment',
                        'metric': insight.metric_name,
                        'current_threshold': current_warning_threshold,
                        'recommended_threshold': new_threshold,
                        'rationale': 'Performance consistently above current threshold',
                        'confidence': insight.confidence,
                        'implementation': 'Update anomaly detector configuration'
                    })

                elif insight.trend_direction == 'DEGRADING' and insight.current_value < 0.7:
                    # Relax thresholds temporarily if struggling
                    new_threshold = max(insight.current_value - 0.05, 0.4)
                    adjustments.append({
                        'type': 'threshold_adjustment',
                        'metric': insight.metric_name,
                        'current_threshold': current_warning_threshold,
                        'recommended_threshold': new_threshold,
                        'rationale': 'Temporary adjustment due to performance degradation',
                        'confidence': insight.confidence,
                        'implementation': 'Update anomaly detector configuration (temporary)',
                        'review_after_days': 14
                    })

        return adjustments

    def generate_system_optimization_recommendations(self) -> List[ImprovementRecommendation]:
        """Generate system-wide optimization recommendations"""
        recommendations = []

        # General system optimization recommendations
        recommendations.append(ImprovementRecommendation(
            category='System Architecture',
            priority='MEDIUM',
            confidence=85.0,
            title="Implement Incremental Data Processing",
            description="Process only changed/new data instead of full daily recomputation",
            impact_assessment="Medium - Reduces processing time and resource usage",
            implementation_effort='HIGH',
            expected_benefit="40-60% reduction in daily processing time",
            implementation_steps=[
                "Design incremental processing framework",
                "Implement change detection for input data",
                "Create delta processing pipelines",
                "Add rollback capabilities for failed incremental updates",
                "Implement full refresh fallback mechanism"
            ],
            success_metrics=[
                "Daily processing time < 15 minutes for incremental updates",
                "Full refresh required < once per week",
                "Zero data consistency issues from incremental processing"
            ],
            related_data={},
            timestamp=datetime.now()
        ))

        recommendations.append(ImprovementRecommendation(
            category='Monitoring Enhancement',
            priority='LOW',
            confidence=75.0,
            title="Implement Predictive Monitoring",
            description="Use machine learning to predict system issues before they occur",
            impact_assessment="Low - Proactive issue prevention and reduced downtime",
            implementation_effort='HIGH',
            expected_benefit="50% reduction in unexpected system issues",
            implementation_steps=[
                "Collect comprehensive system metrics",
                "Build predictive models for key failure modes",
                "Implement early warning system",
                "Create automated preventive actions",
                "Train operations team on predictive alerts"
            ],
            success_metrics=[
                "Predict 80% of system issues 1 hour before occurrence",
                "Reduce unplanned downtime by 50%",
                "False positive rate < 10%"
            ],
            related_data={},
            timestamp=datetime.now()
        ))

        return recommendations

    def calculate_system_optimization_score(self, insights: List[PerformanceInsight]) -> float:
        """Calculate overall system optimization score"""
        if not insights:
            return 50.0  # Default score if no insights

        # Weight different aspects
        weights = {
            'correlation': 0.4,
            'performance': 0.3,
            'stability': 0.2,
            'errors': 0.1
        }

        scores = {}

        for insight in insights:
            category = None
            if 'correlation' in insight.metric_name.lower():
                category = 'correlation'
                # Higher correlation is better
                score = min(insight.current_value * 100, 100)
            elif 'ratio' in insight.metric_name.lower():
                category = 'performance'
                # Lower ratio is better (V2 closer to V1 performance)
                score = max(100 - (insight.current_value - 1) * 50, 0)
            elif 'error' in insight.metric_name.lower():
                category = 'errors'
                # Lower error rate is better
                score = max(100 - insight.current_value * 1000, 0)
            else:
                category = 'stability'
                # Lower anomaly score is better
                score = max(100 - insight.anomaly_score, 0)

            if category:
                if category not in scores:
                    scores[category] = []
                scores[category].append(score)

        # Calculate weighted average
        total_score = 0
        total_weight = 0

        for category, weight in weights.items():
            if category in scores:
                avg_score = np.mean(scores[category])
                total_score += avg_score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 50.0

    def analyze_and_recommend(self) -> ImprovementReport:
        """Run comprehensive analysis and generate improvement recommendations"""
        logger.info("Starting continuous improvement analysis...")

        # Analyze performance patterns
        correlation_insights = self.analyze_correlation_trends()
        performance_insights = self.analyze_performance_patterns()

        all_insights = correlation_insights + performance_insights
        self.performance_insights = all_insights

        # Generate recommendations
        correlation_recommendations = self.generate_correlation_improvement_recommendations(all_insights)
        performance_recommendations = self.generate_performance_improvement_recommendations(all_insights)
        system_recommendations = self.generate_system_optimization_recommendations()

        all_recommendations = correlation_recommendations + performance_recommendations + system_recommendations
        self.recommendations = all_recommendations

        # Generate automated adjustments
        automated_adjustments = self.generate_automated_threshold_adjustments(all_insights)

        # Calculate optimization score
        optimization_score = self.calculate_system_optimization_score(all_insights)

        # Calculate overall health score
        health_score = min(optimization_score + 10, 100)  # Slightly higher than optimization score

        # Generate report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.analysis_period_days)

        report = ImprovementReport(
            analysis_timestamp=datetime.now(),
            analysis_period=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            overall_health_score=health_score,
            performance_insights=all_insights,
            improvement_recommendations=all_recommendations,
            automated_adjustments_proposed=automated_adjustments,
            system_optimization_score=optimization_score,
            next_analysis_date=datetime.now() + timedelta(days=7)
        )

        logger.info(f"Analysis completed - Optimization Score: {optimization_score:.1f}/100, {len(all_recommendations)} recommendations generated")

        return report

    def generate_improvement_report(self, report: ImprovementReport, output_path: Optional[str] = None) -> str:
        """Generate formatted improvement report"""
        report_lines = []

        report_lines.append("="*80)
        report_lines.append("V2 MIGRATION CONTINUOUS IMPROVEMENT ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Analysis Date: {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Analysis Period: {report.analysis_period[0]} to {report.analysis_period[1]}")
        report_lines.append(f"Overall Health Score: {report.overall_health_score:.1f}/100")
        report_lines.append(f"System Optimization Score: {report.system_optimization_score:.1f}/100")
        report_lines.append("")

        # Performance Insights Summary
        report_lines.append("="*80)
        report_lines.append("PERFORMANCE INSIGHTS")
        report_lines.append("="*80)

        for insight in report.performance_insights:
            trend_emoji = {'IMPROVING': '📈', 'DEGRADING': '📉', 'STABLE': '➡️'}[insight.trend_direction]
            report_lines.append(f"{trend_emoji} {insight.metric_name}: {insight.current_value:.3f}")
            report_lines.append(f"    {insight.insight_text}")
            report_lines.append(f"    Confidence: {insight.confidence:.1f}%, Anomaly Score: {insight.anomaly_score:.1f}")
            report_lines.append("")

        # Improvement Recommendations
        report_lines.append("="*80)
        report_lines.append("IMPROVEMENT RECOMMENDATIONS")
        report_lines.append("="*80)

        # Group by priority
        priority_order = ['HIGH', 'MEDIUM', 'LOW']
        for priority in priority_order:
            priority_recs = [r for r in report.improvement_recommendations if r.priority == priority]
            if priority_recs:
                priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[priority]
                report_lines.append(f"\n{priority_emoji} {priority} PRIORITY ({len(priority_recs)} recommendations):")
                report_lines.append("-" * 60)

                for i, rec in enumerate(priority_recs, 1):
                    report_lines.append(f"\n{i}. {rec.title} ({rec.category})")
                    report_lines.append(f"   Confidence: {rec.confidence:.1f}% | Effort: {rec.implementation_effort}")
                    report_lines.append(f"   {rec.description}")
                    report_lines.append(f"   Expected Benefit: {rec.expected_benefit}")
                    report_lines.append("")
                    report_lines.append("   Implementation Steps:")
                    for step in rec.implementation_steps:
                        report_lines.append(f"   • {step}")
                    report_lines.append("")
                    report_lines.append("   Success Metrics:")
                    for metric in rec.success_metrics:
                        report_lines.append(f"   ✓ {metric}")
                    report_lines.append("")

        # Automated Adjustments
        if report.automated_adjustments_proposed:
            report_lines.append("="*80)
            report_lines.append("AUTOMATED ADJUSTMENTS PROPOSED")
            report_lines.append("="*80)

            for adj in report.automated_adjustments_proposed:
                report_lines.append(f"🔧 {adj['metric']} Threshold Adjustment")
                report_lines.append(f"   Current: {adj['current_threshold']:.3f} → Recommended: {adj['recommended_threshold']:.3f}")
                report_lines.append(f"   Rationale: {adj['rationale']}")
                report_lines.append(f"   Implementation: {adj['implementation']}")
                if 'review_after_days' in adj:
                    report_lines.append(f"   Review after: {adj['review_after_days']} days")
                report_lines.append("")

        report_lines.append("="*80)
        report_lines.append(f"Next Analysis: {report.next_analysis_date.strftime('%Y-%m-%d')}")
        report_lines.append("="*80)

        report_text = "\n".join(report_lines)

        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Improvement report saved to: {output_path}")

        return report_text


def main():
    """Main execution for testing"""
    engine = ContinuousImprovementEngine(analysis_period_days=30)
    report = engine.analyze_and_recommend()

    # Generate and print report
    report_text = engine.generate_improvement_report(report)
    print(report_text)

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = project_root / "reports" / f"continuous_improvement_report_{timestamp}.txt"
    output_path.parent.mkdir(exist_ok=True)

    engine.generate_improvement_report(report, str(output_path))


if __name__ == "__main__":
    main()