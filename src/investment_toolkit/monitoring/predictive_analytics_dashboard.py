#!/usr/bin/env python3
"""
Predictive Analytics Dashboard - V2 Migration System

Advanced predictive analytics dashboard for long-term operational planning,
capacity forecasting, and proactive issue prevention.

Task 7.3: Long-term Operations Support Tools
- Predictive system health modeling
- Capacity planning and resource forecasting
- Proactive maintenance scheduling
- Performance trend prediction
- Risk assessment and early warning systems

Features:
- Machine learning-based predictive models
- Interactive forecasting visualizations
- Automated capacity planning recommendations
- Proactive maintenance scheduling
- Long-term trend analysis and insights

Usage:
    from investment_analysis.monitoring.predictive_analytics_dashboard import PredictiveAnalyticsDashboard

    dashboard = PredictiveAnalyticsDashboard()
    dashboard.generate_predictive_dashboard()
    dashboard.forecast_system_health(days_ahead=30)

Created: 2025-09-17
Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    from investment_analysis.monitoring.ab_dashboard import ABMonitoringDashboard
except ImportError as e:
    print(f"❌ Import error: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictiveModel:
    """Predictive model container"""
    model_name: str
    target_metric: str
    model_type: str  # 'linear', 'random_forest', 'polynomial'
    accuracy_score: float  # R² score
    mae_score: float  # Mean Absolute Error
    feature_importance: Dict[str, float]
    model_object: Any
    last_trained: datetime
    forecast_horizon_days: int


@dataclass
class ForecastResult:
    """Forecast result structure"""
    metric_name: str
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    forecast_dates: List[str]
    trend_direction: str  # 'IMPROVING', 'DEGRADING', 'STABLE'
    confidence_score: float
    risk_assessment: str  # 'LOW', 'MEDIUM', 'HIGH'
    recommendations: List[str]


@dataclass
class CapacityForecast:
    """Capacity planning forecast"""
    resource_type: str  # 'storage', 'compute', 'memory'
    current_usage: float
    projected_usage_30d: float
    projected_usage_90d: float
    capacity_limit: float
    time_to_capacity: Optional[int]  # Days until capacity reached
    recommended_actions: List[str]
    growth_rate_per_day: float


@dataclass
class MaintenanceRecommendation:
    """Proactive maintenance recommendation"""
    component: str
    maintenance_type: str  # 'PREVENTIVE', 'OPTIMIZATION', 'UPGRADE'
    priority: str  # 'HIGH', 'MEDIUM', 'LOW'
    recommended_date: datetime
    estimated_downtime: str
    description: str
    benefits: List[str]
    risks_if_delayed: List[str]


class PredictiveAnalyticsDashboard:
    """Advanced predictive analytics dashboard for operations planning"""

    def __init__(self, training_days: int = 90):
        self.training_days = training_days
        self.models: Dict[str, PredictiveModel] = {}
        self.forecasts: List[ForecastResult] = []

        # Model configurations
        self.model_configs = {
            'correlation_forecast': {
                'target': 'total_score_correlation',
                'features': ['performance_ratio', 'error_rate', 'data_quality_score'],
                'model_type': 'random_forest'
            },
            'performance_forecast': {
                'target': 'performance_ratio',
                'features': ['symbols_processed', 'error_rate', 'system_load'],
                'model_type': 'linear'
            },
            'system_health_forecast': {
                'target': 'system_health_score',
                'features': ['correlation', 'performance_ratio', 'error_rate', 'uptime'],
                'model_type': 'random_forest'
            }
        }

        logger.info("Predictive Analytics Dashboard initialized")

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

    def collect_training_data(self) -> pd.DataFrame:
        """Collect historical data for model training"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.training_days)

            # Get comprehensive historical data
            cursor.execute("""
                SELECT
                    date,
                    total_score_correlation,
                    value_pillar_correlation,
                    growth_pillar_correlation,
                    quality_pillar_correlation,
                    momentum_pillar_correlation,
                    top_50_overlap_rate,
                    top_100_overlap_rate,
                    v1_execution_time_ms,
                    v2_execution_time_ms,
                    v1_symbols_processed,
                    v2_symbols_processed,
                    v1_error_count,
                    v2_error_count
                FROM backtest_results.ab_comparison_results
                WHERE date BETWEEN %s AND %s
                ORDER BY date
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

            data = cursor.fetchall()
            cursor.close()
            conn.close()

            if not data:
                logger.warning("No training data available")
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Engineer additional features
            df['performance_ratio'] = df['v2_execution_time_ms'] / df['v1_execution_time_ms']
            df['error_rate'] = (df['v1_error_count'] + df['v2_error_count']) / (df['v1_symbols_processed'] + df['v2_symbols_processed'])
            df['symbols_processed'] = df['v1_symbols_processed'] + df['v2_symbols_processed']

            # Calculate derived metrics
            df['data_quality_score'] = (
                df['total_score_correlation'] * 0.4 +
                df['top_50_overlap_rate'] * 0.3 +
                (1 - df['error_rate']) * 0.3
            ) * 100

            df['system_health_score'] = (
                df['total_score_correlation'] * 30 +
                (1 / df['performance_ratio']) * 20 +  # Inverse performance ratio
                (1 - df['error_rate']) * 25 +
                df['top_50_overlap_rate'] * 25
            )

            # Add temporal features
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['week_of_year'] = pd.to_datetime(df['date']).dt.isocalendar().week

            # Add trend features
            for col in ['total_score_correlation', 'performance_ratio', 'error_rate']:
                if col in df.columns:
                    df[f'{col}_7d_avg'] = df[col].rolling(window=7, min_periods=1).mean()
                    df[f'{col}_trend'] = df[col].diff()

            return df

        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            return pd.DataFrame()

    def train_predictive_models(self, df: pd.DataFrame) -> Dict[str, PredictiveModel]:
        """Train predictive models for various metrics"""
        models = {}

        if df.empty:
            logger.warning("Cannot train models - no data available")
            return models

        for model_name, config in self.model_configs.items():
            try:
                target_col = config['target']
                feature_cols = config['features']

                # Check if target and features exist
                available_features = [f for f in feature_cols if f in df.columns]
                if target_col not in df.columns or len(available_features) < 2:
                    logger.warning(f"Insufficient data for {model_name}")
                    continue

                # Prepare data
                X = df[available_features].dropna()
                y = df.loc[X.index, target_col]

                if len(X) < 10:  # Need minimum data points
                    logger.warning(f"Insufficient samples for {model_name}: {len(X)}")
                    continue

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Split data (use last 20% for testing)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                # Train model
                if config['model_type'] == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:  # default to linear
                    model = LinearRegression()

                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(available_features, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    importance = dict(zip(available_features, np.abs(model.coef_)))
                else:
                    importance = {}

                models[model_name] = PredictiveModel(
                    model_name=model_name,
                    target_metric=target_col,
                    model_type=config['model_type'],
                    accuracy_score=r2,
                    mae_score=mae,
                    feature_importance=importance,
                    model_object=(model, scaler, available_features),
                    last_trained=datetime.now(),
                    forecast_horizon_days=30
                )

                logger.info(f"Trained {model_name}: R²={r2:.3f}, MAE={mae:.3f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")

        return models

    def generate_forecasts(self, models: Dict[str, PredictiveModel], df: pd.DataFrame) -> List[ForecastResult]:
        """Generate forecasts using trained models"""
        forecasts = []

        if df.empty:
            return forecasts

        for model_name, model_info in models.items():
            try:
                model, scaler, feature_cols = model_info.model_object

                # Get latest feature values
                latest_data = df[feature_cols].iloc[-1:].values
                latest_scaled = scaler.transform(latest_data)

                # Generate 30-day forecast
                forecast_dates = []
                forecast_values = []
                confidence_intervals = []

                base_date = datetime.now()

                for i in range(30):
                    forecast_date = base_date + timedelta(days=i+1)
                    forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))

                    # Simple forecast (could be enhanced with time series features)
                    pred = model.predict(latest_scaled)[0]
                    forecast_values.append(pred)

                    # Simple confidence interval (could be improved with proper uncertainty estimation)
                    ci_width = model_info.mae_score * 1.96  # Approximate 95% CI
                    confidence_intervals.append((pred - ci_width, pred + ci_width))

                # Analyze trend
                recent_values = df[model_info.target_metric].iloc[-7:].values
                if len(recent_values) >= 2:
                    trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    if trend_slope > 0.01:
                        trend_direction = 'IMPROVING'
                    elif trend_slope < -0.01:
                        trend_direction = 'DEGRADING'
                    else:
                        trend_direction = 'STABLE'
                else:
                    trend_direction = 'STABLE'

                # Risk assessment
                if model_info.accuracy_score > 0.7 and trend_direction == 'IMPROVING':
                    risk_assessment = 'LOW'
                elif model_info.accuracy_score > 0.5 and trend_direction == 'STABLE':
                    risk_assessment = 'MEDIUM'
                else:
                    risk_assessment = 'HIGH'

                # Generate recommendations
                recommendations = []
                if trend_direction == 'DEGRADING':
                    recommendations.append(f"Monitor {model_info.target_metric} closely - declining trend detected")
                    recommendations.append("Consider proactive intervention to prevent further degradation")

                if model_info.accuracy_score < 0.5:
                    recommendations.append("Model accuracy is low - consider retraining with more data")

                forecasts.append(ForecastResult(
                    metric_name=model_info.target_metric,
                    forecast_values=forecast_values,
                    confidence_intervals=confidence_intervals,
                    forecast_dates=forecast_dates,
                    trend_direction=trend_direction,
                    confidence_score=model_info.accuracy_score * 100,
                    risk_assessment=risk_assessment,
                    recommendations=recommendations
                ))

            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")

        return forecasts

    def analyze_capacity_trends(self, df: pd.DataFrame) -> List[CapacityForecast]:
        """Analyze capacity trends and generate forecasts"""
        capacity_forecasts = []

        if df.empty:
            return capacity_forecasts

        try:
            # Storage capacity analysis (based on data growth)
            if 'symbols_processed' in df.columns:
                symbols_processed = df['symbols_processed'].dropna()
                if len(symbols_processed) >= 7:
                    # Calculate growth rate
                    days = np.arange(len(symbols_processed))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(days, symbols_processed)

                    current_usage = symbols_processed.iloc[-1]
                    growth_per_day = slope

                    # Project usage
                    usage_30d = current_usage + (growth_per_day * 30)
                    usage_90d = current_usage + (growth_per_day * 90)

                    # Assume capacity limit (this would come from system configuration)
                    assumed_capacity = current_usage * 5  # 5x current as limit

                    # Calculate time to capacity
                    if growth_per_day > 0:
                        days_to_capacity = (assumed_capacity - current_usage) / growth_per_day
                        time_to_capacity = int(days_to_capacity) if days_to_capacity > 0 else None
                    else:
                        time_to_capacity = None

                    # Recommendations
                    recommendations = []
                    if time_to_capacity and time_to_capacity < 90:
                        recommendations.append("Consider increasing processing capacity within 90 days")
                        recommendations.append("Monitor data growth trends weekly")

                    if growth_per_day > current_usage * 0.02:  # Growing >2% daily
                        recommendations.append("High growth rate detected - plan capacity expansion")

                    capacity_forecasts.append(CapacityForecast(
                        resource_type='Processing Capacity',
                        current_usage=current_usage,
                        projected_usage_30d=usage_30d,
                        projected_usage_90d=usage_90d,
                        capacity_limit=assumed_capacity,
                        time_to_capacity=time_to_capacity,
                        recommended_actions=recommendations,
                        growth_rate_per_day=growth_per_day
                    ))

            # Performance capacity analysis
            if 'v2_execution_time_ms' in df.columns:
                execution_times = df['v2_execution_time_ms'].dropna()
                if len(execution_times) >= 7:
                    recent_avg = execution_times.tail(7).mean()
                    historical_avg = execution_times.head(30).mean() if len(execution_times) >= 30 else recent_avg

                    performance_degradation = (recent_avg - historical_avg) / historical_avg if historical_avg > 0 else 0

                    recommendations = []
                    if performance_degradation > 0.2:  # 20% slower
                        recommendations.append("Performance degradation detected - consider optimization")
                        recommendations.append("Review system resource allocation")

                    # Simple projection based on trend
                    if len(execution_times) >= 14:
                        days = np.arange(len(execution_times))
                        slope, _, _, _, _ = stats.linregress(days, execution_times)

                        usage_30d = recent_avg + (slope * 30)
                        usage_90d = recent_avg + (slope * 90)

                        # Assume performance limit (e.g., 4x current time)
                        performance_limit = recent_avg * 4

                        capacity_forecasts.append(CapacityForecast(
                            resource_type='Performance Capacity',
                            current_usage=recent_avg,
                            projected_usage_30d=max(usage_30d, 0),
                            projected_usage_90d=max(usage_90d, 0),
                            capacity_limit=performance_limit,
                            time_to_capacity=None,
                            recommended_actions=recommendations,
                            growth_rate_per_day=slope
                        ))

        except Exception as e:
            logger.error(f"Error analyzing capacity trends: {str(e)}")

        return capacity_forecasts

    def generate_maintenance_recommendations(self, df: pd.DataFrame, forecasts: List[ForecastResult]) -> List[MaintenanceRecommendation]:
        """Generate proactive maintenance recommendations"""
        recommendations = []

        try:
            base_date = datetime.now()

            # Database maintenance recommendation
            if not df.empty and len(df) > 30:
                recommendations.append(MaintenanceRecommendation(
                    component='Database',
                    maintenance_type='PREVENTIVE',
                    priority='MEDIUM',
                    recommended_date=base_date + timedelta(days=30),
                    estimated_downtime='2-4 hours',
                    description='Quarterly database optimization and cleanup',
                    benefits=['Improved query performance', 'Reduced storage usage', 'Better data integrity'],
                    risks_if_delayed=['Performance degradation', 'Increased storage costs', 'Potential data corruption']
                ))

            # Model retraining recommendation
            model_age_days = 30  # Assuming models are 30 days old
            if model_age_days >= 60:
                recommendations.append(MaintenanceRecommendation(
                    component='Predictive Models',
                    maintenance_type='OPTIMIZATION',
                    priority='HIGH',
                    recommended_date=base_date + timedelta(days=7),
                    estimated_downtime='1-2 hours',
                    description='Retrain predictive models with recent data',
                    benefits=['Improved forecast accuracy', 'Better anomaly detection', 'Up-to-date insights'],
                    risks_if_delayed=['Degraded prediction quality', 'Missed early warnings', 'Outdated recommendations']
                ))

            # System optimization based on forecasts
            high_risk_forecasts = [f for f in forecasts if f.risk_assessment == 'HIGH']
            if high_risk_forecasts:
                recommendations.append(MaintenanceRecommendation(
                    component='System Performance',
                    maintenance_type='OPTIMIZATION',
                    priority='HIGH',
                    recommended_date=base_date + timedelta(days=14),
                    estimated_downtime='4-6 hours',
                    description='Comprehensive system performance optimization',
                    benefits=['Reduced execution times', 'Better resource utilization', 'Improved reliability'],
                    risks_if_delayed=['Continued performance degradation', 'Increased operational costs', 'System instability']
                ))

            # Configuration review
            recommendations.append(MaintenanceRecommendation(
                component='Configuration',
                maintenance_type='PREVENTIVE',
                priority='LOW',
                recommended_date=base_date + timedelta(days=45),
                estimated_downtime='30 minutes',
                description='Review and update system configuration parameters',
                benefits=['Optimized thresholds', 'Better alert accuracy', 'Improved system behavior'],
                risks_if_delayed=['Suboptimal performance', 'False alerts', 'Missed issues']
            ))

        except Exception as e:
            logger.error(f"Error generating maintenance recommendations: {str(e)}")

        return recommendations

    def create_forecast_visualizations(self, forecasts: List[ForecastResult]) -> go.Figure:
        """Create interactive forecast visualizations"""
        if not forecasts:
            return go.Figure()

        # Create subplots for different metrics
        num_forecasts = len(forecasts)
        fig = make_subplots(
            rows=num_forecasts,
            cols=1,
            subplot_titles=[f"{f.metric_name.replace('_', ' ').title()} Forecast" for f in forecasts],
            vertical_spacing=0.05
        )

        for i, forecast in enumerate(forecasts, 1):
            # Forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast.forecast_dates,
                    y=forecast.forecast_values,
                    mode='lines+markers',
                    name=f'{forecast.metric_name} Forecast',
                    line=dict(color='blue', width=2),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )

            # Confidence interval
            if forecast.confidence_intervals:
                upper_bounds = [ci[1] for ci in forecast.confidence_intervals]
                lower_bounds = [ci[0] for ci in forecast.confidence_intervals]

                fig.add_trace(
                    go.Scatter(
                        x=forecast.forecast_dates + forecast.forecast_dates[::-1],
                        y=upper_bounds + lower_bounds[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        showlegend=(i == 1)
                    ),
                    row=i, col=1
                )

            # Trend indicator
            trend_color = {'IMPROVING': 'green', 'DEGRADING': 'red', 'STABLE': 'orange'}[forecast.trend_direction]
            fig.add_annotation(
                x=forecast.forecast_dates[-1],
                y=forecast.forecast_values[-1],
                text=f"Trend: {forecast.trend_direction}",
                showarrow=True,
                arrowcolor=trend_color,
                arrowhead=2,
                row=i, col=1
            )

        fig.update_layout(
            title="Predictive Analytics - 30-Day Forecasts",
            height=300 * num_forecasts,
            showlegend=True
        )

        return fig

    def create_capacity_planning_charts(self, capacity_forecasts: List[CapacityForecast]) -> go.Figure:
        """Create capacity planning visualizations"""
        if not capacity_forecasts:
            return go.Figure()

        fig = make_subplots(
            rows=len(capacity_forecasts),
            cols=1,
            subplot_titles=[f"{cf.resource_type} Capacity Planning" for cf in capacity_forecasts],
            vertical_spacing=0.1
        )

        for i, cf in enumerate(capacity_forecasts, 1):
            # Current, 30d, 90d projections
            categories = ['Current', '30 Days', '90 Days']
            values = [cf.current_usage, cf.projected_usage_30d, cf.projected_usage_90d]

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    name=cf.resource_type,
                    marker_color=['blue', 'orange', 'red'],
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )

            # Capacity limit line
            fig.add_hline(
                y=cf.capacity_limit,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Capacity Limit: {cf.capacity_limit:.0f}",
                row=i, col=1
            )

        fig.update_layout(
            title="Capacity Planning - Resource Usage Projections",
            height=300 * len(capacity_forecasts),
            showlegend=True
        )

        return fig

    def generate_predictive_dashboard(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive predictive analytics dashboard"""
        logger.info("Generating predictive analytics dashboard...")

        # Collect and prepare data
        df = self.collect_training_data()

        if df.empty:
            logger.warning("No data available for predictive analytics")
            return "<html><body><h1>No data available for predictive analytics</h1></body></html>"

        # Train models and generate forecasts
        models = self.train_predictive_models(df)
        forecasts = self.generate_forecasts(models, df)
        capacity_forecasts = self.analyze_capacity_trends(df)
        maintenance_recommendations = self.generate_maintenance_recommendations(df, forecasts)

        # Create visualizations
        forecast_fig = self.create_forecast_visualizations(forecasts)
        capacity_fig = self.create_capacity_planning_charts(capacity_forecasts)

        # Generate HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Predictive Analytics Dashboard - V2 Migration System</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .forecast-summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .maintenance-item {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
                .high-priority {{ border-left-color: #dc3545; }}
                .medium-priority {{ border-left-color: #ffc107; }}
                .low-priority {{ border-left-color: #28a745; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
                .metric-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔮 Predictive Analytics Dashboard</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Analysis Period:</strong> Last {self.training_days} days</p>
                <p><strong>Forecast Horizon:</strong> 30 days</p>
            </div>

            <div class="section">
                <h2>📈 Predictive Forecasts</h2>
                <div id="forecast-chart"></div>
            </div>

            <div class="section">
                <h2>📊 Forecast Summary</h2>
                <div class="metric-grid">
        """

        # Add forecast summaries
        for forecast in forecasts:
            risk_color = {'LOW': '#d4edda', 'MEDIUM': '#fff3cd', 'HIGH': '#f8d7da'}[forecast.risk_assessment]
            html_content += f"""
                <div class="metric-card" style="background-color: {risk_color};">
                    <h4>{forecast.metric_name.replace('_', ' ').title()}</h4>
                    <p><strong>Trend:</strong> {forecast.trend_direction}</p>
                    <p><strong>Risk Level:</strong> {forecast.risk_assessment}</p>
                    <p><strong>Confidence:</strong> {forecast.confidence_score:.1f}%</p>
                    <p><strong>30-day Forecast:</strong> {forecast.forecast_values[-1]:.3f}</p>
                    <div>
                        <strong>Recommendations:</strong>
                        <ul>
            """
            for rec in forecast.recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += """
                        </ul>
                    </div>
                </div>
            """

        html_content += """
                </div>
            </div>

            <div class="section">
                <h2>💾 Capacity Planning</h2>
                <div id="capacity-chart"></div>
            </div>

            <div class="section">
                <h2>📋 Capacity Forecasts</h2>
        """

        # Add capacity forecasts
        for cf in capacity_forecasts:
            html_content += f"""
                <div class="forecast-summary">
                    <h4>{cf.resource_type}</h4>
                    <p><strong>Current Usage:</strong> {cf.current_usage:.1f}</p>
                    <p><strong>30-day Projection:</strong> {cf.projected_usage_30d:.1f}</p>
                    <p><strong>90-day Projection:</strong> {cf.projected_usage_90d:.1f}</p>
                    <p><strong>Capacity Limit:</strong> {cf.capacity_limit:.1f}</p>
                    <p><strong>Growth Rate:</strong> {cf.growth_rate_per_day:.2f} per day</p>
                    {f'<p><strong>Time to Capacity:</strong> {cf.time_to_capacity} days</p>' if cf.time_to_capacity else ''}
                    <div>
                        <strong>Recommended Actions:</strong>
                        <ul>
            """
            for action in cf.recommended_actions:
                html_content += f"<li>{action}</li>"
            html_content += """
                        </ul>
                    </div>
                </div>
            """

        html_content += """
            </div>

            <div class="section">
                <h2>🔧 Proactive Maintenance Schedule</h2>
        """

        # Add maintenance recommendations
        for maint in maintenance_recommendations:
            priority_class = f"{maint.priority.lower()}-priority"
            html_content += f"""
                <div class="maintenance-item {priority_class}">
                    <h4>{maint.component} - {maint.maintenance_type}</h4>
                    <p><strong>Priority:</strong> {maint.priority}</p>
                    <p><strong>Recommended Date:</strong> {maint.recommended_date.strftime('%Y-%m-%d')}</p>
                    <p><strong>Estimated Downtime:</strong> {maint.estimated_downtime}</p>
                    <p><strong>Description:</strong> {maint.description}</p>
                    <div>
                        <strong>Benefits:</strong>
                        <ul>
            """
            for benefit in maint.benefits:
                html_content += f"<li>{benefit}</li>"
            html_content += """
                        </ul>
                    </div>
                    <div>
                        <strong>Risks if Delayed:</strong>
                        <ul>
            """
            for risk in maint.risks_if_delayed:
                html_content += f"<li>{risk}</li>"
            html_content += """
                        </ul>
                    </div>
                </div>
            """

        # Add JavaScript for charts
        html_content += f"""
            </div>

            <script>
                // Forecast chart
                var forecastData = {forecast_fig.to_json()};
                Plotly.newPlot('forecast-chart', forecastData.data, forecastData.layout);

                // Capacity chart
                var capacityData = {capacity_fig.to_json()};
                Plotly.newPlot('capacity-chart', capacityData.data, capacityData.layout);
            </script>
        </body>
        </html>
        """

        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
            logger.info(f"Predictive analytics dashboard saved to: {output_path}")

        return html_content

    def forecast_system_health(self, days_ahead: int = 30) -> Dict[str, Any]:
        """Generate system health forecast for specified days ahead"""
        df = self.collect_training_data()
        if df.empty:
            return {'error': 'No data available for forecasting'}

        models = self.train_predictive_models(df)
        forecasts = self.generate_forecasts(models, df)

        # Find system health forecast
        health_forecast = next((f for f in forecasts if 'health' in f.metric_name.lower()), None)

        if health_forecast:
            # Get forecast for specific day
            if days_ahead <= len(health_forecast.forecast_values):
                predicted_health = health_forecast.forecast_values[days_ahead - 1]
                confidence_interval = health_forecast.confidence_intervals[days_ahead - 1]

                return {
                    'predicted_health_score': predicted_health,
                    'confidence_interval': confidence_interval,
                    'trend_direction': health_forecast.trend_direction,
                    'risk_assessment': health_forecast.risk_assessment,
                    'forecast_date': health_forecast.forecast_dates[days_ahead - 1],
                    'confidence_score': health_forecast.confidence_score
                }

        return {'error': 'System health forecast not available'}


def main():
    """Main execution for testing"""
    dashboard = PredictiveAnalyticsDashboard(training_days=60)

    # Generate dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = project_root / "reports" / f"predictive_analytics_dashboard_{timestamp}.html"
    output_path.parent.mkdir(exist_ok=True)

    html_content = dashboard.generate_predictive_dashboard(str(output_path))

    # Test health forecast
    health_forecast = dashboard.forecast_system_health(days_ahead=7)
    print(f"7-day health forecast: {health_forecast}")


if __name__ == "__main__":
    main()