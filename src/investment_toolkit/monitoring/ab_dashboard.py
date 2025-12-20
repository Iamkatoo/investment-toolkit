#!/usr/bin/env python3
"""
AB Monitoring Dashboard - V2 Migration System

Real-time monitoring dashboard for V1/V2 AB testing system.
Provides comprehensive visualization of system health, correlation metrics,
performance indicators, and alert status.

Implementation Task 5.1: Real-time Monitoring Dashboard
- Real-time system health monitoring
- Interactive correlation and performance charts
- Alert status panel with active anomalies
- Performance overview with V1/V2 comparison
- Historical trend analysis and reporting

Key Features:
- Real-time metrics display with auto-refresh
- Interactive Plotly charts for data visualization
- Alert prioritization and status tracking
- Performance degradation monitoring
- System health scoring and trending
- Export capabilities for reporting

Usage:
    from investment_toolkit.monitoring.ab_dashboard import ABMonitoringDashboard
    
    dashboard = ABMonitoringDashboard()
    dashboard.generate_dashboard(output_path="reports/ab_monitoring_dashboard.html")
    dashboard.start_real_time_monitoring()

Created: 2025-09-15
Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from pathlib import Path
import sys
from dataclasses import dataclass
import asyncio
import threading
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from investment_toolkit.monitoring.ab_anomaly_detector import ABAnomalyDetector, AnomalyEvent, SystemHealthScore, AlertLevel
from investment_toolkit.analysis.ab_comparison_engine import ABComparisonEngine, ABComparisonResult
from investment_toolkit.analysis.ab_report_generator import ABReportGenerator
from investment_toolkit.utilities.feature_flags import is_enabled
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from sqlalchemy import create_engine, text

# Setup logging
logger = logging.getLogger(__name__)

# Dashboard color scheme - consistent with existing reports
DASHBOARD_COLORS = {
    'primary': '#007bff',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    
    # Status colors
    'healthy': '#28a745',
    'warning': '#ffc107',
    'critical': '#fd7e14',
    'emergency': '#dc3545',
    
    # Chart colors
    'v1_color': '#007bff',
    'v2_color': '#28a745',
    'correlation_color': '#6f42c1',
    'performance_color': '#fd7e14',
    'alert_color': '#dc3545'
}


@dataclass
class DashboardMetrics:
    """Current dashboard metrics snapshot"""
    timestamp: datetime
    system_health_score: float
    correlation: float
    top_50_overlap: float
    execution_time_ratio: float
    active_alerts: int
    error_rate: float
    data_quality_score: float
    v1_execution_time: float
    v2_execution_time: float
    total_symbols: int
    health_status: str


class ABMonitoringDashboard:
    """
    AB Testing Real-time Monitoring Dashboard
    
    Provides comprehensive real-time monitoring and visualization
    of the V1/V2 AB testing system performance and health.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AB Monitoring Dashboard
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.anomaly_detector = ABAnomalyDetector()
        self.comparison_engine = ABComparisonEngine()
        self.report_generator = ABReportGenerator()
        
        # Database connection
        self.db_engine = self._create_db_engine()
        
        # Dashboard data
        self.current_metrics: Optional[DashboardMetrics] = None
        self.historical_data: List[DashboardMetrics] = []
        self.recent_anomalies: List[AnomalyEvent] = []
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        logger.info("AB Monitoring Dashboard initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load dashboard configuration"""
        default_config = {
            'refresh_interval_seconds': 30,
            'historical_data_days': 30,
            'max_chart_points': 1000,
            'alert_refresh_seconds': 10,
            'health_thresholds': {
                'healthy': 90,
                'warning': 70,
                'critical': 50
            },
            'charts': {
                'correlation_threshold': 0.7,
                'performance_threshold': 2.0,
                'height': 400,
                'show_grid': True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load dashboard config: {e}")
        
        return default_config
    
    def _create_db_engine(self):
        """Create database engine for accessing AB test data"""
        try:
            connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(connection_string, pool_pre_ping=True)
            return engine
        except Exception as e:
            logger.warning(f"Failed to create database engine: {e}")
            return None
    
    def load_current_metrics(self) -> Optional[DashboardMetrics]:
        """Load current system metrics from database and components"""
        try:
            # Get latest health status
            health_status = self.anomaly_detector.get_current_health_status()
            
            # Get recent anomalies
            anomaly_summary = self.anomaly_detector.get_anomaly_summary(hours=24)
            
            # Get latest AB comparison results
            latest_comparison = self._get_latest_ab_comparison()
            
            if not latest_comparison:
                logger.warning("No recent AB comparison data available")
                return None
            
            # Extract metrics
            correlation = latest_comparison.correlation_analysis.pearson_correlation
            top_50_overlap = latest_comparison.ranking_analysis.top_50_overlap
            exec_time_ratio = latest_comparison.performance_metrics.get('execution_time_ratio', 1.0)
            v1_time = latest_comparison.performance_metrics.get('v1_execution_time', 0.0)
            v2_time = latest_comparison.performance_metrics.get('v2_execution_time', 0.0)
            error_rate = latest_comparison.performance_metrics.get('error_rate', 0.0)
            total_symbols = latest_comparison.total_symbols
            
            # Calculate data quality score
            data_quality_score = 100.0
            if latest_comparison.outlier_detections:
                max_outlier_pct = max(d.outlier_percentage for d in latest_comparison.outlier_detections)
                data_quality_score = max(0, 100 - max_outlier_pct * 5)
            
            metrics = DashboardMetrics(
                timestamp=datetime.now(),
                system_health_score=health_status.get('overall_score', 0),
                correlation=correlation,
                top_50_overlap=top_50_overlap,
                execution_time_ratio=exec_time_ratio,
                active_alerts=health_status.get('active_anomalies', 0),
                error_rate=error_rate,
                data_quality_score=data_quality_score,
                v1_execution_time=v1_time,
                v2_execution_time=v2_time,
                total_symbols=total_symbols,
                health_status=health_status.get('status', 'UNKNOWN')
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load current metrics: {e}")
            return None
    
    def _get_latest_ab_comparison(self) -> Optional[ABComparisonResult]:
        """Get latest AB comparison result from database"""
        if not self.db_engine:
            return None
        
        try:
            query = """
            SELECT comparison_data 
            FROM backtest_results.ab_comparison_results 
            ORDER BY analysis_date DESC 
            LIMIT 1
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
                if row:
                    comparison_data = json.loads(row[0])
                    return ABComparisonResult.from_dict(comparison_data)
                    
        except Exception as e:
            logger.error(f"Failed to load latest AB comparison: {e}")
        
        return None
    
    def load_historical_data(self, days: int = 30) -> List[DashboardMetrics]:
        """Load historical metrics data"""
        if not self.db_engine:
            return []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
            SELECT analysis_date, comparison_data 
            FROM backtest_results.ab_comparison_results 
            WHERE analysis_date >= :cutoff_date
            ORDER BY analysis_date DESC
            LIMIT :max_points
            """
            
            historical_metrics = []
            
            with self.db_engine.connect() as conn:
                result = conn.execute(
                    text(query), 
                    {
                        'cutoff_date': cutoff_date,
                        'max_points': self.config['max_chart_points']
                    }
                )
                
                for row in result:
                    try:
                        comparison_data = json.loads(row[1])
                        ab_result = ABComparisonResult.from_dict(comparison_data)
                        
                        # Calculate health score (simplified)
                        correlation = ab_result.correlation_analysis.pearson_correlation
                        exec_time_ratio = ab_result.performance_metrics.get('execution_time_ratio', 1.0)
                        health_score = min(100, max(0, correlation * 100 - (exec_time_ratio - 1.0) * 50))
                        
                        metrics = DashboardMetrics(
                            timestamp=row[0],
                            system_health_score=health_score,
                            correlation=correlation,
                            top_50_overlap=ab_result.ranking_analysis.top_50_overlap,
                            execution_time_ratio=exec_time_ratio,
                            active_alerts=0,  # Would need separate query for historical alerts
                            error_rate=ab_result.performance_metrics.get('error_rate', 0.0),
                            data_quality_score=100.0,  # Simplified
                            v1_execution_time=ab_result.performance_metrics.get('v1_execution_time', 0.0),
                            v2_execution_time=ab_result.performance_metrics.get('v2_execution_time', 0.0),
                            total_symbols=ab_result.total_symbols,
                            health_status="UNKNOWN"  # Would need calculation
                        )
                        
                        historical_metrics.append(metrics)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse historical data row: {e}")
                        continue
            
            return historical_metrics
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return []
    
    def create_health_overview_chart(self) -> go.Figure:
        """Create system health overview chart"""
        if not self.current_metrics:
            return go.Figure()
        
        # Health gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=self.current_metrics.system_health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health Score"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': DASHBOARD_COLORS['primary']},
                'steps': [
                    {'range': [0, 50], 'color': DASHBOARD_COLORS['emergency']},
                    {'range': [50, 70], 'color': DASHBOARD_COLORS['critical']},
                    {'range': [70, 90], 'color': DASHBOARD_COLORS['warning']},
                    {'range': [90, 100], 'color': DASHBOARD_COLORS['healthy']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=self.config['charts']['height'],
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_correlation_trend_chart(self) -> go.Figure:
        """Create correlation trend chart"""
        if not self.historical_data:
            return go.Figure()
        
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'correlation': m.correlation,
                'top_50_overlap': m.top_50_overlap
            }
            for m in self.historical_data
        ])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Correlation Trend', 'Top 50 Ranking Overlap'],
            vertical_spacing=0.1
        )
        
        # Correlation line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['correlation'],
                mode='lines+markers',
                name='Pearson Correlation',
                line=dict(color=DASHBOARD_COLORS['correlation_color'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Correlation threshold line
        fig.add_hline(
            y=self.config['charts']['correlation_threshold'],
            line_dash="dash",
            line_color=DASHBOARD_COLORS['warning'],
            annotation_text="Target Threshold",
            row=1, col=1
        )
        
        # Top 50 overlap
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['top_50_overlap'],
                mode='lines+markers',
                name='Top 50 Overlap',
                line=dict(color=DASHBOARD_COLORS['success'], width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=self.config['charts']['height'] * 1.5,
            showlegend=True,
            title_text="Correlation Analysis Trends"
        )
        
        fig.update_xaxes(showgrid=self.config['charts']['show_grid'])
        fig.update_yaxes(showgrid=self.config['charts']['show_grid'])
        
        return fig
    
    def create_performance_comparison_chart(self) -> go.Figure:
        """Create V1 vs V2 performance comparison chart"""
        if not self.historical_data:
            return go.Figure()
        
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'v1_time': m.v1_execution_time,
                'v2_time': m.v2_execution_time,
                'ratio': m.execution_time_ratio
            }
            for m in self.historical_data
        ])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Execution Time Comparison', 'Performance Ratio (V2/V1)'],
            vertical_spacing=0.1
        )
        
        # Execution times
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['v1_time'],
                mode='lines+markers',
                name='V1 Execution Time',
                line=dict(color=DASHBOARD_COLORS['v1_color'], width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['v2_time'],
                mode='lines+markers',
                name='V2 Execution Time',
                line=dict(color=DASHBOARD_COLORS['v2_color'], width=2)
            ),
            row=1, col=1
        )
        
        # Performance ratio
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['ratio'],
                mode='lines+markers',
                name='V2/V1 Ratio',
                line=dict(color=DASHBOARD_COLORS['performance_color'], width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Performance threshold
        fig.add_hline(
            y=self.config['charts']['performance_threshold'],
            line_dash="dash",
            line_color=DASHBOARD_COLORS['warning'],
            annotation_text="Performance Threshold",
            row=2, col=1
        )
        
        fig.update_layout(
            height=self.config['charts']['height'] * 1.5,
            showlegend=True,
            title_text="Performance Analysis"
        )
        
        return fig
    
    def create_alert_status_panel(self) -> Dict[str, Any]:
        """Create alert status information panel"""
        # Get recent anomalies
        anomaly_summary = self.anomaly_detector.get_anomaly_summary(hours=24)
        
        # Get active anomalies
        active_anomalies = []
        for anomaly in self.anomaly_detector.active_anomalies[-10:]:  # Last 10 active
            active_anomalies.append({
                'timestamp': anomaly.timestamp.strftime('%H:%M:%S'),
                'type': anomaly.anomaly_type.value,
                'level': anomaly.alert_level.value,
                'metric': anomaly.metric_name,
                'message': anomaly.message,
                'current_value': f"{anomaly.current_value:.4f}",
                'expected_value': f"{anomaly.expected_value:.4f}"
            })
        
        return {
            'total_alerts_24h': anomaly_summary['total_anomalies'],
            'active_alerts': len(active_anomalies),
            'alert_breakdown': anomaly_summary['by_level'],
            'active_anomalies': active_anomalies,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def create_metrics_summary_panel(self) -> Dict[str, Any]:
        """Create current metrics summary panel"""
        if not self.current_metrics:
            return {}
        
        # Status color mapping
        status_colors = {
            'HEALTHY': DASHBOARD_COLORS['healthy'],
            'WARNING': DASHBOARD_COLORS['warning'],
            'CRITICAL': DASHBOARD_COLORS['critical'],
            'EMERGENCY': DASHBOARD_COLORS['emergency']
        }
        
        return {
            'system_health': {
                'score': f"{self.current_metrics.system_health_score:.1f}",
                'status': self.current_metrics.health_status,
                'color': status_colors.get(self.current_metrics.health_status, DASHBOARD_COLORS['dark'])
            },
            'correlation': {
                'value': f"{self.current_metrics.correlation:.4f}",
                'percentage': f"{self.current_metrics.correlation * 100:.1f}%",
                'status': 'good' if self.current_metrics.correlation >= 0.7 else 'warning'
            },
            'performance': {
                'ratio': f"{self.current_metrics.execution_time_ratio:.2f}x",
                'v1_time': f"{self.current_metrics.v1_execution_time:.1f}s",
                'v2_time': f"{self.current_metrics.v2_execution_time:.1f}s",
                'status': 'good' if self.current_metrics.execution_time_ratio <= 2.0 else 'warning'
            },
            'data_quality': {
                'score': f"{self.current_metrics.data_quality_score:.1f}",
                'symbols_processed': self.current_metrics.total_symbols,
                'error_rate': f"{self.current_metrics.error_rate * 100:.2f}%"
            },
            'ranking': {
                'top_50_overlap': f"{self.current_metrics.top_50_overlap * 100:.1f}%",
                'status': 'good' if self.current_metrics.top_50_overlap >= 0.8 else 'warning'
            },
            'last_updated': self.current_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def generate_dashboard_html(self, output_path: Optional[str] = None) -> str:
        """Generate complete HTML dashboard"""
        from investment_toolkit.utilities.paths import get_or_create_reports_config

        if output_path is None:
            _reports_config = get_or_create_reports_config()
            output_path = str(_reports_config.graphs_dir / "ab_monitoring_dashboard.html")
        # Load current data
        self.current_metrics = self.load_current_metrics()
        self.historical_data = self.load_historical_data(
            days=self.config['historical_data_days']
        )
        
        # Create charts
        health_chart = self.create_health_overview_chart()
        correlation_chart = self.create_correlation_trend_chart()
        performance_chart = self.create_performance_comparison_chart()
        
        # Create data panels
        alert_panel = self.create_alert_status_panel()
        metrics_panel = self.create_metrics_summary_panel()
        
        # Convert charts to HTML
        health_html = health_chart.to_html(include_plotlyjs=False, div_id="health-chart")
        correlation_html = correlation_chart.to_html(include_plotlyjs=False, div_id="correlation-chart")
        performance_html = performance_chart.to_html(include_plotlyjs=False, div_id="performance-chart")
        
        # Generate complete HTML
        html_content = self._generate_complete_html(
            health_html, correlation_html, performance_html,
            alert_panel, metrics_panel
        )
        
        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard generated: {output_path}")
        return output_path
    
    def _generate_complete_html(self, health_html: str, correlation_html: str, 
                               performance_html: str, alert_panel: Dict[str, Any],
                               metrics_panel: Dict[str, Any]) -> str:
        """Generate complete HTML document"""
        
        # Generate alert rows HTML
        alert_rows = ""
        for alert in alert_panel.get('active_anomalies', []):
            level_color = {
                'WARNING': DASHBOARD_COLORS['warning'],
                'CRITICAL': DASHBOARD_COLORS['critical'],
                'EMERGENCY': DASHBOARD_COLORS['emergency']
            }.get(alert['level'], DASHBOARD_COLORS['info'])
            
            alert_rows += f"""
            <tr>
                <td>{alert['timestamp']}</td>
                <td><span class="badge" style="background-color: {level_color}">{alert['level']}</span></td>
                <td>{alert['type']}</td>
                <td>{alert['metric']}</td>
                <td>{alert['current_value']}</td>
                <td>{alert['message']}</td>
            </tr>
            """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AB Testing Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f8f9fa;
        }}
        
        .dashboard {{ 
            max-width: 1400px; 
            margin: 0 auto; 
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, {DASHBOARD_COLORS['primary']} 0%, {DASHBOARD_COLORS['info']} 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            margin-top: 10px;
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid {DASHBOARD_COLORS['primary']};
        }}
        
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: {DASHBOARD_COLORS['dark']};
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-subtitle {{
            color: #6c757d;
            font-size: 0.9rem;
        }}
        
        .chart-container {{
            background: white;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .chart-header {{
            padding: 15px 20px;
            background-color: {DASHBOARD_COLORS['light']};
            border-bottom: 1px solid #dee2e6;
            font-weight: 600;
            color: {DASHBOARD_COLORS['dark']};
        }}
        
        .alert-panel {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        
        .alert-header {{
            padding: 15px 20px;
            background-color: {DASHBOARD_COLORS['danger']};
            color: white;
            border-radius: 10px 10px 0 0;
            font-weight: 600;
        }}
        
        .alert-content {{
            padding: 20px;
        }}
        
        .alert-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        .alert-table th {{
            background-color: {DASHBOARD_COLORS['light']};
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
            font-weight: 600;
            color: {DASHBOARD_COLORS['dark']};
        }}
        
        .alert-table td {{
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
            font-size: 0.9rem;
        }}
        
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .status-good {{ color: {DASHBOARD_COLORS['success']}; }}
        .status-warning {{ color: {DASHBOARD_COLORS['warning']}; }}
        .status-danger {{ color: {DASHBOARD_COLORS['danger']}; }}
        
        .refresh-info {{
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 20px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {{
            location.reload();
        }}, 30000);
    </script>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>AB Testing Monitoring Dashboard</h1>
            <div class="subtitle">V1/V2 System Performance & Health Monitoring</div>
            <div class="subtitle">Last Updated: {metrics_panel.get('last_updated', 'N/A')}</div>
        </div>
        
        <!-- Metrics Overview -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>System Health</h3>
                <div class="metric-value" style="color: {metrics_panel.get('system_health', {}).get('color', DASHBOARD_COLORS['dark'])}">
                    {metrics_panel.get('system_health', {}).get('score', 'N/A')}
                </div>
                <div class="metric-subtitle">{metrics_panel.get('system_health', {}).get('status', 'Unknown')}</div>
            </div>
            
            <div class="metric-card">
                <h3>Correlation</h3>
                <div class="metric-value status-{'good' if metrics_panel.get('correlation', {}).get('status') == 'good' else 'warning'}">
                    {metrics_panel.get('correlation', {}).get('percentage', 'N/A')}
                </div>
                <div class="metric-subtitle">Pearson Correlation: {metrics_panel.get('correlation', {}).get('value', 'N/A')}</div>
            </div>
            
            <div class="metric-card">
                <h3>Performance Ratio</h3>
                <div class="metric-value status-{'good' if metrics_panel.get('performance', {}).get('status') == 'good' else 'warning'}">
                    {metrics_panel.get('performance', {}).get('ratio', 'N/A')}
                </div>
                <div class="metric-subtitle">V2 vs V1 execution time</div>
            </div>
            
            <div class="metric-card">
                <h3>Ranking Overlap</h3>
                <div class="metric-value status-{'good' if metrics_panel.get('ranking', {}).get('status') == 'good' else 'warning'}">
                    {metrics_panel.get('ranking', {}).get('top_50_overlap', 'N/A')}
                </div>
                <div class="metric-subtitle">Top 50 stocks agreement</div>
            </div>
            
            <div class="metric-card">
                <h3>Active Alerts</h3>
                <div class="metric-value status-{'danger' if alert_panel.get('active_alerts', 0) > 0 else 'good'}">
                    {alert_panel.get('active_alerts', 0)}
                </div>
                <div class="metric-subtitle">{alert_panel.get('total_alerts_24h', 0)} total in 24h</div>
            </div>
            
            <div class="metric-card">
                <h3>Data Quality</h3>
                <div class="metric-value status-good">
                    {metrics_panel.get('data_quality', {}).get('score', 'N/A')}
                </div>
                <div class="metric-subtitle">{metrics_panel.get('data_quality', {}).get('symbols_processed', 'N/A')} symbols processed</div>
            </div>
        </div>
        
        <!-- System Health Gauge -->
        <div class="chart-container">
            <div class="chart-header">System Health Overview</div>
            {health_html}
        </div>
        
        <!-- Correlation Trends -->
        <div class="chart-container">
            <div class="chart-header">Correlation Analysis</div>
            {correlation_html}
        </div>
        
        <!-- Performance Analysis -->
        <div class="chart-container">
            <div class="chart-header">Performance Metrics</div>
            {performance_html}
        </div>
        
        <!-- Active Alerts -->
        <div class="alert-panel">
            <div class="alert-header">
                Active Alerts & Anomalies ({alert_panel.get('active_alerts', 0)})
            </div>
            <div class="alert-content">
                <div style="margin-bottom: 15px;">
                    <strong>Alert Summary (24h):</strong>
                    {' | '.join([f"{level}: {count}" for level, count in alert_panel.get('alert_breakdown', {}).items()])}
                </div>
                
                <table class="alert-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Level</th>
                            <th>Type</th>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Message</th>
                        </tr>
                    </thead>
                    <tbody>
                        {alert_rows if alert_rows else '<tr><td colspan="6" style="text-align: center; color: #6c757d;">No active alerts</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="refresh-info">
            🔄 Dashboard auto-refreshes every 30 seconds | 
            📊 Data from last {self.config['historical_data_days']} days | 
            ⚠️ Monitoring: {'Active' if self.monitoring_active else 'Inactive'}
        </div>
    </div>
</body>
</html>
        """
    
    def start_real_time_monitoring(self, interval_seconds: Optional[int] = None) -> None:
        """Start real-time monitoring in background thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        interval = interval_seconds or self.config['refresh_interval_seconds']
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Generate updated dashboard
                    self.generate_dashboard_html()
                    logger.info("Dashboard updated")
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Real-time monitoring started with {interval}s interval")
    
    def stop_real_time_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Real-time monitoring stopped")
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get current dashboard status"""
        return {
            'monitoring_active': self.monitoring_active,
            'last_update': self.current_metrics.timestamp.isoformat() if self.current_metrics else None,
            'historical_data_points': len(self.historical_data),
            'database_connected': self.db_engine is not None,
            'config': self.config
        }


def main():
    """Example usage and testing"""
    import argparse
    from investment_toolkit.utilities.paths import get_or_create_reports_config

    _reports_config = get_or_create_reports_config()
    default_output = str(_reports_config.graphs_dir / "ab_monitoring_dashboard.html")

    parser = argparse.ArgumentParser(description='AB Monitoring Dashboard')
    parser.add_argument('--output', '-o', default=default_output,
                       help='Output HTML file path')
    parser.add_argument('--monitor', '-m', action='store_true',
                       help='Start real-time monitoring')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = ABMonitoringDashboard()
    
    # Generate initial dashboard
    output_path = dashboard.generate_dashboard_html(args.output)
    print(f"Dashboard generated: {output_path}")
    
    # Start monitoring if requested
    if args.monitor:
        print(f"Starting real-time monitoring (interval: {args.interval}s)")
        dashboard.start_real_time_monitoring(args.interval)
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            dashboard.stop_real_time_monitoring()
    
    print("Dashboard status:", dashboard.get_dashboard_status())


if __name__ == "__main__":
    main()