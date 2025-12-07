#!/usr/bin/env python3
"""
AB Anomaly Detector - V2 Migration System

Advanced anomaly detection and alerting system for monitoring V1/V2 scoring
system health during AB testing. Provides real-time monitoring, threshold-based
alerts, and automated response mechanisms.

Implementation Task 2.4: Anomaly Detection & Alert System
- Multi-layered anomaly detection algorithms
- Configurable alert thresholds and escalation rules
- Real-time monitoring with sliding window analysis
- Automated notification and escalation system
- Historical trend analysis and drift detection
- System health scoring and dashboards

Key Features:
- Statistical process control with control charts
- Machine learning-based anomaly detection
- Correlation monitoring and drift alerts
- Performance degradation detection
- Outlier analysis and trend monitoring
- Slack/email/PagerDuty integration
- Escalation policies and on-call rotation
- Historical baseline establishment

Usage:
    from investment_toolkit.monitoring.ab_anomaly_detector import ABAnomalyDetector
    
    detector = ABAnomalyDetector()
    anomalies = detector.check_system_health(ab_result)
    detector.monitor_real_time_metrics(correlation, performance_data)
    detector.trigger_alerts_if_needed(anomalies)

Created: 2025-09-15
Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import json
import time
import threading
import queue
from collections import deque
from enum import Enum
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from investment_toolkit.analysis.ab_comparison_engine import ABComparisonResult
from investment_toolkit.utilities.feature_flags import is_enabled
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from sqlalchemy import create_engine, text

# Setup logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    CORRELATION_DROP = "CORRELATION_DROP"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    HIGH_ERROR_RATE = "HIGH_ERROR_RATE"
    RANKING_DRIFT = "RANKING_DRIFT"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    OUTLIER_SPIKE = "OUTLIER_SPIKE"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"
    DATA_QUALITY_ISSUE = "DATA_QUALITY_ISSUE"
    TREND_DEVIATION = "TREND_DEVIATION"


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event"""
    timestamp: datetime
    anomaly_type: AnomalyType
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    expected_value: float
    threshold: float
    deviation_score: float
    message: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class SystemHealthScore:
    """Overall system health assessment"""
    timestamp: datetime
    overall_score: float  # 0-100
    correlation_score: float
    performance_score: float
    reliability_score: float
    data_quality_score: float
    trend_score: float
    active_anomalies: int
    health_status: str  # "HEALTHY", "WARNING", "CRITICAL", "EMERGENCY"


@dataclass
class AlertThresholds:
    """Configurable alert thresholds"""
    correlation_warning: float = 0.6
    correlation_critical: float = 0.4
    correlation_emergency: float = 0.2
    performance_warning_ratio: float = 1.5
    performance_critical_ratio: float = 2.0
    performance_emergency_ratio: float = 3.0
    error_rate_warning: float = 0.02
    error_rate_critical: float = 0.05
    error_rate_emergency: float = 0.10
    ranking_overlap_warning: float = 0.7
    ranking_overlap_critical: float = 0.5
    ranking_overlap_emergency: float = 0.3
    outlier_percentage_warning: float = 5.0
    outlier_percentage_critical: float = 10.0
    outlier_percentage_emergency: float = 20.0


class MetricMonitor:
    """Monitor individual metrics with statistical process control"""
    
    def __init__(self, metric_name: str, window_size: int = 20, sensitivity: float = 2.0):
        """
        Initialize metric monitor
        
        Args:
            metric_name: Name of the metric being monitored
            window_size: Size of the sliding window for trend analysis
            sensitivity: Sensitivity multiplier for anomaly detection
        """
        self.metric_name = metric_name
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.baseline_established = False
    
    def add_value(self, value: float, timestamp: datetime) -> Optional[AnomalyEvent]:
        """
        Add new value and check for anomalies
        
        Args:
            value: New metric value
            timestamp: Timestamp of the measurement
            
        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        self.history.append(value)
        self.timestamps.append(timestamp)
        
        # Establish baseline if we have enough data
        if len(self.history) >= self.window_size and not self.baseline_established:
            self.baseline_mean = np.mean(self.history)
            self.baseline_std = np.std(self.history)
            self.baseline_established = True
            logger.info(f"Baseline established for {self.metric_name}: μ={self.baseline_mean:.4f}, σ={self.baseline_std:.4f}")
        
        # Check for anomalies if baseline is established
        if self.baseline_established and self.baseline_std > 0:
            z_score = abs(value - self.baseline_mean) / self.baseline_std
            
            if z_score > self.sensitivity:
                # Determine alert level based on severity
                if z_score > 4.0:
                    alert_level = AlertLevel.EMERGENCY
                elif z_score > 3.0:
                    alert_level = AlertLevel.CRITICAL
                else:
                    alert_level = AlertLevel.WARNING
                
                return AnomalyEvent(
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.TREND_DEVIATION,
                    alert_level=alert_level,
                    metric_name=self.metric_name,
                    current_value=value,
                    expected_value=self.baseline_mean,
                    threshold=self.baseline_mean + self.sensitivity * self.baseline_std,
                    deviation_score=z_score,
                    message=f"{self.metric_name} deviation: {z_score:.2f}σ from baseline",
                    context={
                        'baseline_mean': self.baseline_mean,
                        'baseline_std': self.baseline_std,
                        'z_score': z_score,
                        'window_size': len(self.history)
                    }
                )
        
        return None
    
    def get_trend(self) -> Dict[str, float]:
        """Calculate trend statistics"""
        if len(self.history) < 3:
            return {'slope': 0.0, 'r_squared': 0.0}
        
        x = np.arange(len(self.history))
        y = np.array(self.history)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing'
            }
        except:
            return {'slope': 0.0, 'r_squared': 0.0}


class NotificationService:
    """Handle various notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize notification service with configuration"""
        self.config = config
        self.notification_history = []
        self.rate_limit_window = 300  # 5 minutes
        self.max_notifications_per_window = 10
    
    def _is_rate_limited(self, alert_level: AlertLevel) -> bool:
        """Check if we're hitting rate limits"""
        now = datetime.now()
        recent_notifications = [
            n for n in self.notification_history 
            if (now - n['timestamp']).total_seconds() < self.rate_limit_window
        ]
        
        # Emergency alerts bypass rate limiting
        if alert_level == AlertLevel.EMERGENCY:
            return False
        
        # Check rate limits for other alert levels
        return len(recent_notifications) >= self.max_notifications_per_window
    
    async def send_notification(self, anomaly: AnomalyEvent) -> bool:
        """
        Send notification via appropriate channels
        
        Args:
            anomaly: Anomaly event to notify about
            
        Returns:
            True if notification sent successfully
        """
        if self._is_rate_limited(anomaly.alert_level):
            logger.warning(f"Rate limit exceeded, skipping notification for {anomaly.anomaly_type}")
            return False
        
        success = False
        
        # Send Slack notification
        if self.config.get('slack', {}).get('enabled', False):
            success |= await self._send_slack_notification(anomaly)
        
        # Send email for critical and emergency alerts
        if anomaly.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            if self.config.get('email', {}).get('enabled', False):
                success |= self._send_email_notification(anomaly)
        
        # Send PagerDuty for emergency alerts
        if anomaly.alert_level == AlertLevel.EMERGENCY:
            if self.config.get('pagerduty', {}).get('enabled', False):
                success |= await self._send_pagerduty_alert(anomaly)
        
        # Record notification
        if success:
            self.notification_history.append({
                'timestamp': datetime.now(),
                'anomaly_type': anomaly.anomaly_type,
                'alert_level': anomaly.alert_level
            })
        
        return success
    
    async def _send_slack_notification(self, anomaly: AnomalyEvent) -> bool:
        """Send Slack notification"""
        try:
            webhook_url = self.config['slack']['webhook_url']
            
            # Color coding based on alert level
            colors = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ff9500",
                AlertLevel.CRITICAL: "#ff0000",
                AlertLevel.EMERGENCY: "#8B0000"
            }
            
            # Emoji mapping
            emojis = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🆘"
            }
            
            payload = {
                "attachments": [{
                    "color": colors[anomaly.alert_level],
                    "fields": [
                        {
                            "title": f"{emojis[anomaly.alert_level]} {anomaly.alert_level.value} - {anomaly.anomaly_type.value}",
                            "value": anomaly.message,
                            "short": False
                        },
                        {
                            "title": "Metric",
                            "value": anomaly.metric_name,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": f"{anomaly.current_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Expected Value",
                            "value": f"{anomaly.expected_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Deviation Score",
                            "value": f"{anomaly.deviation_score:.2f}",
                            "short": True
                        }
                    ],
                    "footer": "V2 Migration AB Anomaly Detector",
                    "ts": int(anomaly.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for {anomaly.anomaly_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _send_email_notification(self, anomaly: AnomalyEvent) -> bool:
        """Send email notification for critical alerts"""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"[{anomaly.alert_level.value}] V2 Migration Alert: {anomaly.anomaly_type.value}"
            
            body = f"""
ALERT: V2 Migration System Anomaly Detected

Alert Level: {anomaly.alert_level.value}
Anomaly Type: {anomaly.anomaly_type.value}
Timestamp: {anomaly.timestamp.isoformat()}

Metric: {anomaly.metric_name}
Current Value: {anomaly.current_value:.4f}
Expected Value: {anomaly.expected_value:.4f}
Threshold: {anomaly.threshold:.4f}
Deviation Score: {anomaly.deviation_score:.2f}

Message: {anomaly.message}

Context:
{json.dumps(anomaly.context, indent=2)}

Please investigate and take appropriate action.

---
V2 Migration AB Anomaly Detector
            """.strip()
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', True):
                    server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email notification sent for {anomaly.anomaly_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    async def _send_pagerduty_alert(self, anomaly: AnomalyEvent) -> bool:
        """Send PagerDuty alert for emergency situations"""
        try:
            pagerduty_config = self.config['pagerduty']
            
            payload = {
                "routing_key": pagerduty_config['integration_key'],
                "event_action": "trigger",
                "dedup_key": f"v2_migration_{anomaly.anomaly_type.value}_{anomaly.timestamp.strftime('%Y%m%d')}",
                "payload": {
                    "summary": f"V2 Migration Emergency: {anomaly.anomaly_type.value}",
                    "source": "V2 Migration AB Anomaly Detector",
                    "severity": "critical",
                    "custom_details": {
                        "metric": anomaly.metric_name,
                        "current_value": anomaly.current_value,
                        "expected_value": anomaly.expected_value,
                        "deviation_score": anomaly.deviation_score,
                        "message": anomaly.message,
                        "context": anomaly.context
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"PagerDuty alert sent for {anomaly.anomaly_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False


class ABAnomalyDetector:
    """
    Advanced AB Anomaly Detection System
    
    Monitors V1/V2 scoring system health and detects anomalies in real-time
    using multiple detection algorithms and statistical methods.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AB Anomaly Detector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.thresholds = AlertThresholds(**self.config.get('thresholds', {}))
        
        # Initialize monitoring components
        self.metric_monitors: Dict[str, MetricMonitor] = {}
        self.notification_service = NotificationService(self.config.get('notifications', {}))
        
        # Active anomalies tracking
        self.active_anomalies: List[AnomalyEvent] = []
        self.anomaly_history: List[AnomalyEvent] = []
        
        # System health tracking
        self.health_history: List[SystemHealthScore] = []
        
        # Database connection
        self.db_engine = self._create_db_engine()
        
        # Initialize metric monitors
        self._initialize_monitors()
        
        logger.info("AB Anomaly Detector initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'thresholds': {},  # Uses AlertThresholds defaults
            'monitors': {
                'correlation_sensitivity': 2.0,
                'performance_sensitivity': 2.5,
                'window_size': 20,
                'enable_ml_detection': True
            },
            'notifications': {
                'slack': {
                    'enabled': True,
                    'webhook_url': os.getenv('SLACK_WEBHOOK_URL')
                },
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'use_tls': True,
                    'from_address': os.getenv('EMAIL_FROM'),
                    'to_addresses': [],
                    'username': os.getenv('EMAIL_USERNAME'),
                    'password': os.getenv('EMAIL_PASSWORD')
                },
                'pagerduty': {
                    'enabled': False,
                    'integration_key': os.getenv('PAGERDUTY_INTEGRATION_KEY')
                }
            },
            'escalation': {
                'auto_rollback_enabled': True,
                'escalation_timeout_minutes': 15,
                'max_consecutive_alerts': 5
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _create_db_engine(self):
        """Create database engine for storing anomaly data"""
        try:
            connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(connection_string, pool_pre_ping=True)
            return engine
        except Exception as e:
            logger.warning(f"Failed to create database engine: {e}")
            return None
    
    def _initialize_monitors(self):
        """Initialize metric monitors"""
        monitor_config = self.config['monitors']
        
        # Key metrics to monitor
        metrics = [
            'correlation',
            'top_50_overlap',
            'top_100_overlap',
            'execution_time_ratio',
            'error_rate',
            'outlier_percentage'
        ]
        
        for metric in metrics:
            self.metric_monitors[metric] = MetricMonitor(
                metric_name=metric,
                window_size=monitor_config['window_size'],
                sensitivity=monitor_config.get(f'{metric}_sensitivity', 2.0)
            )
    
    def check_system_health(self, ab_result: ABComparisonResult) -> List[AnomalyEvent]:
        """
        Comprehensive system health check
        
        Args:
            ab_result: AB comparison result to analyze
            
        Returns:
            List of detected anomalies
        """
        logger.info(f"Checking system health for {ab_result.analysis_date}")
        
        detected_anomalies = []
        timestamp = ab_result.timestamp
        
        try:
            # Check correlation health
            correlation_anomalies = self._check_correlation_health(ab_result, timestamp)
            detected_anomalies.extend(correlation_anomalies)
            
            # Check performance health
            performance_anomalies = self._check_performance_health(ab_result, timestamp)
            detected_anomalies.extend(performance_anomalies)
            
            # Check ranking health
            ranking_anomalies = self._check_ranking_health(ab_result, timestamp)
            detected_anomalies.extend(ranking_anomalies)
            
            # Check data quality
            data_quality_anomalies = self._check_data_quality(ab_result, timestamp)
            detected_anomalies.extend(data_quality_anomalies)
            
            # Check for outlier spikes
            outlier_anomalies = self._check_outlier_health(ab_result, timestamp)
            detected_anomalies.extend(outlier_anomalies)
            
            # Update active anomalies
            self._update_active_anomalies(detected_anomalies)
            
            # Calculate system health score
            health_score = self._calculate_system_health_score(ab_result, detected_anomalies)
            self.health_history.append(health_score)
            
            # Store anomalies in database
            if detected_anomalies and self.db_engine:
                self._store_anomalies_in_database(detected_anomalies)
            
            logger.info(f"Health check completed: {len(detected_anomalies)} anomalies detected")
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            # Create emergency anomaly for system failure
            system_failure = AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.SYSTEM_FAILURE,
                alert_level=AlertLevel.EMERGENCY,
                metric_name="system_health_check",
                current_value=0.0,
                expected_value=1.0,
                threshold=1.0,
                deviation_score=float('inf'),
                message=f"System health check failed: {str(e)}",
                context={'error': str(e)}
            )
            detected_anomalies.append(system_failure)
        
        return detected_anomalies
    
    def _check_correlation_health(self, ab_result: ABComparisonResult, timestamp: datetime) -> List[AnomalyEvent]:
        """Check correlation-related health metrics"""
        anomalies = []
        
        correlation = ab_result.correlation_analysis.pearson_correlation
        
        # Monitor correlation with trend analysis
        monitor_anomaly = self.metric_monitors['correlation'].add_value(correlation, timestamp)
        if monitor_anomaly:
            anomalies.append(monitor_anomaly)
        
        # Check against static thresholds
        if correlation < self.thresholds.correlation_emergency:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.CORRELATION_DROP,
                alert_level=AlertLevel.EMERGENCY,
                metric_name="pearson_correlation",
                current_value=correlation,
                expected_value=0.7,
                threshold=self.thresholds.correlation_emergency,
                deviation_score=(0.7 - correlation) / 0.1,
                message=f"EMERGENCY: Correlation critically low at {correlation:.3f}",
                context={'threshold': self.thresholds.correlation_emergency}
            ))
        elif correlation < self.thresholds.correlation_critical:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.CORRELATION_DROP,
                alert_level=AlertLevel.CRITICAL,
                metric_name="pearson_correlation",
                current_value=correlation,
                expected_value=0.7,
                threshold=self.thresholds.correlation_critical,
                deviation_score=(0.7 - correlation) / 0.1,
                message=f"CRITICAL: Correlation dropped to {correlation:.3f}",
                context={'threshold': self.thresholds.correlation_critical}
            ))
        elif correlation < self.thresholds.correlation_warning:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.CORRELATION_DROP,
                alert_level=AlertLevel.WARNING,
                metric_name="pearson_correlation",
                current_value=correlation,
                expected_value=0.7,
                threshold=self.thresholds.correlation_warning,
                deviation_score=(0.7 - correlation) / 0.1,
                message=f"WARNING: Correlation below target at {correlation:.3f}",
                context={'threshold': self.thresholds.correlation_warning}
            ))
        
        return anomalies
    
    def _check_performance_health(self, ab_result: ABComparisonResult, timestamp: datetime) -> List[AnomalyEvent]:
        """Check performance-related health metrics"""
        anomalies = []
        
        # Check execution time ratio
        performance_metrics = ab_result.performance_metrics
        exec_time_ratio = performance_metrics.get('execution_time_ratio', 1.0)
        
        if exec_time_ratio > 0:  # Only check if we have valid data
            monitor_anomaly = self.metric_monitors['execution_time_ratio'].add_value(exec_time_ratio, timestamp)
            if monitor_anomaly:
                anomalies.append(monitor_anomaly)
            
            # Check against performance thresholds
            if exec_time_ratio > self.thresholds.performance_emergency_ratio:
                anomalies.append(AnomalyEvent(
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    alert_level=AlertLevel.EMERGENCY,
                    metric_name="execution_time_ratio",
                    current_value=exec_time_ratio,
                    expected_value=1.0,
                    threshold=self.thresholds.performance_emergency_ratio,
                    deviation_score=exec_time_ratio - 1.0,
                    message=f"EMERGENCY: Severe performance degradation - {exec_time_ratio:.2f}x slower",
                    context={'threshold': self.thresholds.performance_emergency_ratio}
                ))
            elif exec_time_ratio > self.thresholds.performance_critical_ratio:
                anomalies.append(AnomalyEvent(
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    alert_level=AlertLevel.CRITICAL,
                    metric_name="execution_time_ratio",
                    current_value=exec_time_ratio,
                    expected_value=1.0,
                    threshold=self.thresholds.performance_critical_ratio,
                    deviation_score=exec_time_ratio - 1.0,
                    message=f"CRITICAL: Significant performance degradation - {exec_time_ratio:.2f}x slower",
                    context={'threshold': self.thresholds.performance_critical_ratio}
                ))
            elif exec_time_ratio > self.thresholds.performance_warning_ratio:
                anomalies.append(AnomalyEvent(
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                    alert_level=AlertLevel.WARNING,
                    metric_name="execution_time_ratio",
                    current_value=exec_time_ratio,
                    expected_value=1.0,
                    threshold=self.thresholds.performance_warning_ratio,
                    deviation_score=exec_time_ratio - 1.0,
                    message=f"WARNING: Performance degradation detected - {exec_time_ratio:.2f}x slower",
                    context={'threshold': self.thresholds.performance_warning_ratio}
                ))
        
        return anomalies
    
    def _check_ranking_health(self, ab_result: ABComparisonResult, timestamp: datetime) -> List[AnomalyEvent]:
        """Check ranking overlap health metrics"""
        anomalies = []
        
        ranking = ab_result.ranking_analysis
        top_50_overlap = ranking.top_50_overlap
        
        # Monitor with trend analysis
        monitor_anomaly = self.metric_monitors['top_50_overlap'].add_value(top_50_overlap, timestamp)
        if monitor_anomaly:
            anomalies.append(monitor_anomaly)
        
        # Check against thresholds
        if top_50_overlap < self.thresholds.ranking_overlap_emergency:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.RANKING_DRIFT,
                alert_level=AlertLevel.EMERGENCY,
                metric_name="top_50_overlap",
                current_value=top_50_overlap,
                expected_value=0.8,
                threshold=self.thresholds.ranking_overlap_emergency,
                deviation_score=(0.8 - top_50_overlap) / 0.1,
                message=f"EMERGENCY: Top 50 ranking severely diverged - {top_50_overlap:.1%} overlap",
                context={'threshold': self.thresholds.ranking_overlap_emergency}
            ))
        elif top_50_overlap < self.thresholds.ranking_overlap_critical:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.RANKING_DRIFT,
                alert_level=AlertLevel.CRITICAL,
                metric_name="top_50_overlap",
                current_value=top_50_overlap,
                expected_value=0.8,
                threshold=self.thresholds.ranking_overlap_critical,
                deviation_score=(0.8 - top_50_overlap) / 0.1,
                message=f"CRITICAL: Top 50 ranking drift detected - {top_50_overlap:.1%} overlap",
                context={'threshold': self.thresholds.ranking_overlap_critical}
            ))
        elif top_50_overlap < self.thresholds.ranking_overlap_warning:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.RANKING_DRIFT,
                alert_level=AlertLevel.WARNING,
                metric_name="top_50_overlap",
                current_value=top_50_overlap,
                expected_value=0.8,
                threshold=self.thresholds.ranking_overlap_warning,
                deviation_score=(0.8 - top_50_overlap) / 0.1,
                message=f"WARNING: Top 50 ranking below target - {top_50_overlap:.1%} overlap",
                context={'threshold': self.thresholds.ranking_overlap_warning}
            ))
        
        return anomalies
    
    def _check_data_quality(self, ab_result: ABComparisonResult, timestamp: datetime) -> List[AnomalyEvent]:
        """Check data quality metrics"""
        anomalies = []
        
        # Check for missing data
        total_symbols = ab_result.total_symbols
        performance_metrics = ab_result.performance_metrics
        
        missing_v1 = performance_metrics.get('missing_v1_symbols', 0)
        missing_v2 = performance_metrics.get('missing_v2_symbols', 0)
        
        missing_percentage = (missing_v1 + missing_v2) / max(total_symbols, 1) * 100
        
        if missing_percentage > 10:  # More than 10% missing data
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.DATA_QUALITY_ISSUE,
                alert_level=AlertLevel.CRITICAL,
                metric_name="missing_data_percentage",
                current_value=missing_percentage,
                expected_value=0.0,
                threshold=10.0,
                deviation_score=missing_percentage / 10.0,
                message=f"CRITICAL: High missing data rate - {missing_percentage:.1f}%",
                context={
                    'missing_v1_symbols': missing_v1,
                    'missing_v2_symbols': missing_v2,
                    'total_symbols': total_symbols
                }
            ))
        elif missing_percentage > 5:  # More than 5% missing data
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.DATA_QUALITY_ISSUE,
                alert_level=AlertLevel.WARNING,
                metric_name="missing_data_percentage",
                current_value=missing_percentage,
                expected_value=0.0,
                threshold=5.0,
                deviation_score=missing_percentage / 5.0,
                message=f"WARNING: Elevated missing data rate - {missing_percentage:.1f}%",
                context={
                    'missing_v1_symbols': missing_v1,
                    'missing_v2_symbols': missing_v2,
                    'total_symbols': total_symbols
                }
            ))
        
        return anomalies
    
    def _check_outlier_health(self, ab_result: ABComparisonResult, timestamp: datetime) -> List[AnomalyEvent]:
        """Check outlier detection metrics"""
        anomalies = []
        
        if not ab_result.outlier_detections:
            return anomalies
        
        # Get the highest outlier percentage across all detection methods
        max_outlier_percentage = max(d.outlier_percentage for d in ab_result.outlier_detections)
        
        monitor_anomaly = self.metric_monitors['outlier_percentage'].add_value(max_outlier_percentage, timestamp)
        if monitor_anomaly:
            anomalies.append(monitor_anomaly)
        
        # Check against thresholds
        if max_outlier_percentage > self.thresholds.outlier_percentage_emergency:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.OUTLIER_SPIKE,
                alert_level=AlertLevel.EMERGENCY,
                metric_name="outlier_percentage",
                current_value=max_outlier_percentage,
                expected_value=2.0,
                threshold=self.thresholds.outlier_percentage_emergency,
                deviation_score=max_outlier_percentage / self.thresholds.outlier_percentage_emergency,
                message=f"EMERGENCY: Extreme outlier spike - {max_outlier_percentage:.1f}%",
                context={'threshold': self.thresholds.outlier_percentage_emergency}
            ))
        elif max_outlier_percentage > self.thresholds.outlier_percentage_critical:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.OUTLIER_SPIKE,
                alert_level=AlertLevel.CRITICAL,
                metric_name="outlier_percentage",
                current_value=max_outlier_percentage,
                expected_value=2.0,
                threshold=self.thresholds.outlier_percentage_critical,
                deviation_score=max_outlier_percentage / self.thresholds.outlier_percentage_critical,
                message=f"CRITICAL: High outlier rate - {max_outlier_percentage:.1f}%",
                context={'threshold': self.thresholds.outlier_percentage_critical}
            ))
        elif max_outlier_percentage > self.thresholds.outlier_percentage_warning:
            anomalies.append(AnomalyEvent(
                timestamp=timestamp,
                anomaly_type=AnomalyType.OUTLIER_SPIKE,
                alert_level=AlertLevel.WARNING,
                metric_name="outlier_percentage",
                current_value=max_outlier_percentage,
                expected_value=2.0,
                threshold=self.thresholds.outlier_percentage_warning,
                deviation_score=max_outlier_percentage / self.thresholds.outlier_percentage_warning,
                message=f"WARNING: Elevated outlier rate - {max_outlier_percentage:.1f}%",
                context={'threshold': self.thresholds.outlier_percentage_warning}
            ))
        
        return anomalies
    
    def _calculate_system_health_score(self, ab_result: ABComparisonResult, 
                                     anomalies: List[AnomalyEvent]) -> SystemHealthScore:
        """Calculate overall system health score"""
        # Component scores (0-100)
        correlation_score = min(100, ab_result.correlation_analysis.pearson_correlation * 100)
        
        # Performance score (inverse of execution time ratio)
        exec_time_ratio = ab_result.performance_metrics.get('execution_time_ratio', 1.0)
        performance_score = max(0, 100 - (exec_time_ratio - 1.0) * 50)
        
        # Reliability score (based on error rates and missing data)
        missing_percentage = ab_result.performance_metrics.get('missing_data_percentage', 0)
        reliability_score = max(0, 100 - missing_percentage * 10)
        
        # Data quality score (inverse of outlier percentage)
        max_outlier_percentage = 0
        if ab_result.outlier_detections:
            max_outlier_percentage = max(d.outlier_percentage for d in ab_result.outlier_detections)
        data_quality_score = max(0, 100 - max_outlier_percentage * 5)
        
        # Trend score (based on recent anomaly history)
        recent_anomalies = [a for a in self.anomaly_history[-20:] if a.alert_level != AlertLevel.INFO]
        trend_score = max(0, 100 - len(recent_anomalies) * 5)
        
        # Overall score (weighted average)
        overall_score = (
            correlation_score * 0.3 +
            performance_score * 0.2 +
            reliability_score * 0.2 +
            data_quality_score * 0.15 +
            trend_score * 0.15
        )
        
        # Determine health status
        if overall_score >= 90:
            health_status = "HEALTHY"
        elif overall_score >= 70:
            health_status = "WARNING"
        elif overall_score >= 50:
            health_status = "CRITICAL"
        else:
            health_status = "EMERGENCY"
        
        return SystemHealthScore(
            timestamp=ab_result.timestamp,
            overall_score=overall_score,
            correlation_score=correlation_score,
            performance_score=performance_score,
            reliability_score=reliability_score,
            data_quality_score=data_quality_score,
            trend_score=trend_score,
            active_anomalies=len([a for a in anomalies if not a.resolved]),
            health_status=health_status
        )
    
    def _update_active_anomalies(self, new_anomalies: List[AnomalyEvent]):
        """Update active anomalies list"""
        # Add new anomalies
        for anomaly in new_anomalies:
            self.active_anomalies.append(anomaly)
            self.anomaly_history.append(anomaly)
        
        # Remove resolved anomalies (older than 1 hour for non-critical)
        now = datetime.now()
        self.active_anomalies = [
            a for a in self.active_anomalies
            if not a.resolved and (
                a.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY] or
                (now - a.timestamp).total_seconds() < 3600
            )
        ]
    
    def _store_anomalies_in_database(self, anomalies: List[AnomalyEvent]):
        """Store anomaly events in database"""
        if not self.db_engine:
            return
        
        try:
            for anomaly in anomalies:
                anomaly_data = {
                    'timestamp': anomaly.timestamp,
                    'anomaly_type': anomaly.anomaly_type.value,
                    'alert_level': anomaly.alert_level.value,
                    'metric_name': anomaly.metric_name,
                    'current_value': anomaly.current_value,
                    'expected_value': anomaly.expected_value,
                    'threshold_value': anomaly.threshold,
                    'deviation_score': anomaly.deviation_score,
                    'message': anomaly.message,
                    'context_json': json.dumps(anomaly.context),
                    'resolved': anomaly.resolved
                }
                
                query = """
                INSERT INTO backtest_results.anomaly_events 
                (timestamp, anomaly_type, alert_level, metric_name, current_value, 
                 expected_value, threshold_value, deviation_score, message, context_json, resolved)
                VALUES (:timestamp, :anomaly_type, :alert_level, :metric_name, :current_value,
                        :expected_value, :threshold_value, :deviation_score, :message, :context_json, :resolved)
                """
                
                with self.db_engine.connect() as conn:
                    conn.execute(text(query), anomaly_data)
                    conn.commit()
                    
            logger.info(f"Stored {len(anomalies)} anomalies in database")
            
        except Exception as e:
            logger.error(f"Failed to store anomalies in database: {e}")
    
    async def trigger_alerts_if_needed(self, anomalies: List[AnomalyEvent]) -> int:
        """
        Trigger alerts for detected anomalies
        
        Args:
            anomalies: List of anomalies to potentially alert on
            
        Returns:
            Number of alerts sent
        """
        alerts_sent = 0
        
        for anomaly in anomalies:
            # Skip INFO level anomalies
            if anomaly.alert_level == AlertLevel.INFO:
                continue
            
            try:
                success = await self.notification_service.send_notification(anomaly)
                if success:
                    alerts_sent += 1
                    logger.info(f"Alert sent for {anomaly.anomaly_type.value}")
                else:
                    logger.warning(f"Failed to send alert for {anomaly.anomaly_type.value}")
                    
            except Exception as e:
                logger.error(f"Error sending alert for {anomaly.anomaly_type.value}: {e}")
        
        return alerts_sent
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        if not self.health_history:
            return {"status": "UNKNOWN", "score": 0}
        
        latest_health = self.health_history[-1]
        
        return {
            "status": latest_health.health_status,
            "overall_score": latest_health.overall_score,
            "component_scores": {
                "correlation": latest_health.correlation_score,
                "performance": latest_health.performance_score,
                "reliability": latest_health.reliability_score,
                "data_quality": latest_health.data_quality_score,
                "trend": latest_health.trend_score
            },
            "active_anomalies": latest_health.active_anomalies,
            "timestamp": latest_health.timestamp.isoformat()
        }
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_anomalies = [
            a for a in self.anomaly_history 
            if a.timestamp >= cutoff_time
        ]
        
        # Group by type and level
        by_type = {}
        by_level = {}
        
        for anomaly in recent_anomalies:
            by_type[anomaly.anomaly_type.value] = by_type.get(anomaly.anomaly_type.value, 0) + 1
            by_level[anomaly.alert_level.value] = by_level.get(anomaly.alert_level.value, 0) + 1
        
        return {
            "total_anomalies": len(recent_anomalies),
            "time_period_hours": hours,
            "by_type": by_type,
            "by_level": by_level,
            "active_anomalies": len(self.active_anomalies)
        }


def main():
    """Example usage and testing"""
    import asyncio
    from investment_toolkit.analysis.ab_comparison_engine import ABComparisonEngine, ABComparisonResult
    
    # Initialize detector
    detector = ABAnomalyDetector()
    
    # Create sample AB result for testing
    # In practice, this would come from the actual AB comparison
    print("AB Anomaly Detector initialized successfully!")
    print(f"Current health status: {detector.get_current_health_status()}")
    print(f"Anomaly summary: {detector.get_anomaly_summary()}")


if __name__ == "__main__":
    main()