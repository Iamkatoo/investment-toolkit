"""
Feature Flags Management System for Scoring V2 Migration

This module provides a comprehensive feature flag system to enable safe,
gradual migration from the V1 scoring system to the new V2 5-pillar system.

Key Features:
- Environment-specific flag configurations
- Dependency validation between flags
- Caching for performance
- Rollback condition monitoring
- Integration with existing logging system

Created: 2025-09-14
Author: Claude Code Assistant
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import threading
from dataclasses import dataclass, field
import hashlib
import random

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class FlagConfig:
    """Configuration for a single feature flag"""
    name: str
    default: bool
    description: str
    environments: Dict[str, bool] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    auto_rollback_conditions: Dict[str, Any] = field(default_factory=dict)
    rollout_percentage: Optional[int] = None  # For gradual rollout (0-100)


@dataclass
class RollbackCondition:
    """Configuration for rollback conditions"""
    critical_error_rate: float = 0.05
    performance_degradation: float = 2.0
    correlation_threshold: float = 0.5
    top_overlap_threshold: float = 0.5
    manual_triggers: List[str] = field(default_factory=list)


class FeatureFlagError(Exception):
    """Custom exception for feature flag related errors"""
    pass


class DependencyValidationError(FeatureFlagError):
    """Exception raised when flag dependencies are not satisfied"""
    pass


class FeatureFlags:
    """
    Feature Flag Management System
    
    Manages feature flags for the scoring system migration, including:
    - Loading configuration from YAML
    - Environment-specific flag evaluation
    - Dependency validation
    - Caching for performance
    - Rollback condition monitoring
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize the feature flags system
        
        Args:
            config_path: Path to the feature flags YAML configuration
            environment: Current environment (development, staging, production)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.environment = environment or self._detect_environment()
        self._config: Dict[str, Any] = {}
        self._flags: Dict[str, FlagConfig] = {}
        self._cache: Dict[str, bool] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes cache TTL
        
        self._load_configuration()
        logger.info(f"FeatureFlags initialized for environment: {self.environment}")
    
    @classmethod
    def get_instance(cls, config_path: Optional[str] = None, environment: Optional[str] = None) -> 'FeatureFlags':
        """
        Get singleton instance of FeatureFlags
        
        Args:
            config_path: Path to configuration file (only used on first call)
            environment: Environment name (only used on first call)
            
        Returns:
            FeatureFlags singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path, environment)
        return cls._instance
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        # Get the project root directory (Investment folder)
        current_dir = Path(__file__).resolve()
        project_root = current_dir
        
        # Navigate up to find the Investment directory
        while project_root.name != 'Investment' and project_root.parent != project_root:
            project_root = project_root.parent
        
        if project_root.name != 'Investment':
            raise FeatureFlagError("Could not find Investment project root directory")
        
        config_path = project_root / 'config' / 'feature_flags.yaml'
        return str(config_path)
    
    def _detect_environment(self) -> str:
        """
        Detect the current environment based on various indicators
        
        Returns:
            Environment name (development, staging, production)
        """
        # Check environment variable first
        env = os.getenv('ENVIRONMENT', '').lower()
        if env in ['development', 'staging', 'production']:
            return env
        
        env = os.getenv('ENV', '').lower()
        if env in ['dev', 'development']:
            return 'development'
        elif env in ['stage', 'staging']:
            return 'staging'
        elif env in ['prod', 'production']:
            return 'production'
        
        # Check if we're in a development-like environment
        if os.getenv('USER') in ['HOME', 'developer'] or 'dev' in os.getcwd().lower():
            return 'development'
        
        # Default to development for safety
        logger.warning("Could not detect environment, defaulting to 'development'")
        return 'development'
    
    def _load_configuration(self) -> None:
        """Load feature flags configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            # Parse individual flags
            self._flags = {}
            feature_flags = self._config.get('feature_flags', {})
            
            for flag_name, flag_data in feature_flags.items():
                self._flags[flag_name] = FlagConfig(
                    name=flag_name,
                    default=flag_data.get('default', False),
                    description=flag_data.get('description', ''),
                    environments=flag_data.get('environments', {}),
                    dependencies=flag_data.get('dependencies', []),
                    auto_rollback_conditions=flag_data.get('auto_rollback_conditions', {}),
                    rollout_percentage=flag_data.get('rollout_percentage')
                )
            
            logger.info(f"Loaded {len(self._flags)} feature flags from {self.config_path}")
            
        except FileNotFoundError:
            logger.error(f"Feature flags configuration file not found: {self.config_path}")
            raise FeatureFlagError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in feature flags configuration: {e}")
            raise FeatureFlagError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            logger.error(f"Error loading feature flags configuration: {e}")
            raise FeatureFlagError(f"Configuration loading error: {e}")
    
    def _is_cache_valid(self) -> bool:
        """Check if the current cache is still valid"""
        if self._cache_timestamp is None:
            return False
        
        cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
        return cache_age < self._cache_ttl_seconds
    
    def _clear_cache(self) -> None:
        """Clear the flag evaluation cache"""
        self._cache.clear()
        self._cache_timestamp = None
        logger.debug("Feature flag cache cleared")
    
    def reload_configuration(self) -> None:
        """Reload configuration from file and clear cache"""
        logger.info("Reloading feature flags configuration")
        self._load_configuration()
        self._clear_cache()
    
    def is_enabled(self, flag_name: str, bypass_cache: bool = False) -> bool:
        """
        Check if a feature flag is enabled
        
        Args:
            flag_name: Name of the feature flag
            bypass_cache: If True, skip cache and evaluate fresh
            
        Returns:
            True if flag is enabled, False otherwise
            
        Raises:
            FeatureFlagError: If flag doesn't exist
            DependencyValidationError: If dependencies are not satisfied
        """
        # Check cache first (unless bypassed)
        if not bypass_cache and self._is_cache_valid() and flag_name in self._cache:
            return self._cache[flag_name]
        
        if flag_name not in self._flags:
            logger.error(f"Unknown feature flag: {flag_name}")
            raise FeatureFlagError(f"Unknown feature flag: {flag_name}")
        
        flag_config = self._flags[flag_name]
        
        # Get environment-specific value or default
        if self.environment in flag_config.environments:
            flag_value = flag_config.environments[self.environment]
        else:
            flag_value = flag_config.default
        
        # If flag is enabled, validate dependencies and check rollout percentage
        if flag_value:
            self._validate_dependencies(flag_name)
            
            # Check gradual rollout percentage for applicable flags
            if flag_config.rollout_percentage is not None:
                flag_value = self._should_enable_for_rollout(flag_name, flag_config.rollout_percentage)
        
        # Cache the result
        if self._cache_timestamp is None:
            self._cache_timestamp = datetime.now()
        self._cache[flag_name] = flag_value
        
        logger.debug(f"Flag {flag_name} evaluated to {flag_value} for environment {self.environment}")
        return flag_value
    
    def _validate_dependencies(self, flag_name: str) -> None:
        """
        Validate that all dependencies for a flag are satisfied
        
        Args:
            flag_name: Name of the flag to validate dependencies for
            
        Raises:
            DependencyValidationError: If dependencies are not satisfied
        """
        flag_config = self._flags[flag_name]
        
        for dependency in flag_config.dependencies:
            if dependency not in self._flags:
                raise DependencyValidationError(
                    f"Flag {flag_name} depends on unknown flag: {dependency}"
                )
            
            # Recursively check if dependency is enabled
            if not self.is_enabled(dependency, bypass_cache=True):
                raise DependencyValidationError(
                    f"Flag {flag_name} cannot be enabled because dependency {dependency} is disabled"
                )
    
    def get_dependent_flags(self, flag_name: str) -> List[str]:
        """
        Get list of flags that depend on the specified flag
        
        Args:
            flag_name: Name of the flag to find dependents for
            
        Returns:
            List of flag names that depend on the specified flag
        """
        dependents = []
        
        for name, config in self._flags.items():
            if flag_name in config.dependencies:
                dependents.append(name)
        
        return dependents
    
    def _should_enable_for_rollout(self, flag_name: str, rollout_percentage: int) -> bool:
        """
        Determine if a flag should be enabled based on rollout percentage
        
        Uses a consistent hash-based approach to ensure the same user/system
        always gets the same result for a given flag and percentage.
        
        Args:
            flag_name: Name of the flag
            rollout_percentage: Percentage of users/systems that should see the flag (0-100)
            
        Returns:
            True if the flag should be enabled for this user/system, False otherwise
        """
        if rollout_percentage <= 0:
            return False
        if rollout_percentage >= 100:
            return True
            
        # Create a consistent identifier for this system/user
        # Using environment + hostname + flag_name for deterministic results
        system_id = f"{self.environment}_{os.getenv('HOSTNAME', 'localhost')}_{flag_name}"
        
        # Create hash and convert to percentage (0-99)
        hash_value = hashlib.md5(system_id.encode()).hexdigest()
        hash_int = int(hash_value[:8], 16)  # Use first 8 chars of hash
        percentage_bucket = hash_int % 100
        
        enabled = percentage_bucket < rollout_percentage
        
        logger.debug(f"Rollout check for {flag_name}: {rollout_percentage}% target, "
                    f"bucket={percentage_bucket}, enabled={enabled}")
        
        return enabled
    
    def validate_all_dependencies(self) -> Dict[str, bool]:
        """
        Validate dependencies for all enabled flags
        
        Returns:
            Dictionary mapping flag names to validation status
        """
        validation_results = {}
        
        for flag_name in self._flags:
            try:
                if self.is_enabled(flag_name, bypass_cache=True):
                    self._validate_dependencies(flag_name)
                validation_results[flag_name] = True
            except DependencyValidationError as e:
                logger.warning(f"Dependency validation failed for {flag_name}: {e}")
                validation_results[flag_name] = False
        
        return validation_results
    
    def get_enabled_flags(self) -> List[str]:
        """
        Get list of all currently enabled flags
        
        Returns:
            List of enabled flag names
        """
        enabled = []
        
        for flag_name in self._flags:
            try:
                if self.is_enabled(flag_name):
                    enabled.append(flag_name)
            except (FeatureFlagError, DependencyValidationError):
                # Skip flags with errors
                continue
        
        return enabled
    
    def get_flag_info(self, flag_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific flag
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            Dictionary with flag information
            
        Raises:
            FeatureFlagError: If flag doesn't exist
        """
        if flag_name not in self._flags:
            raise FeatureFlagError(f"Unknown feature flag: {flag_name}")
        
        flag_config = self._flags[flag_name]
        
        return {
            'name': flag_config.name,
            'description': flag_config.description,
            'default': flag_config.default,
            'environments': flag_config.environments,
            'dependencies': flag_config.dependencies,
            'current_value': self.is_enabled(flag_name),
            'dependents': self.get_dependent_flags(flag_name),
            'auto_rollback_conditions': flag_config.auto_rollback_conditions
        }
    
    def get_all_flags_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all flags
        
        Returns:
            Dictionary mapping flag names to their status information
        """
        status = {}
        
        for flag_name in self._flags:
            try:
                status[flag_name] = self.get_flag_info(flag_name)
            except Exception as e:
                status[flag_name] = {
                    'name': flag_name,
                    'error': str(e),
                    'current_value': False
                }
        
        return status
    
    def get_rollback_conditions(self) -> RollbackCondition:
        """
        Get rollback conditions configuration
        
        Returns:
            RollbackCondition object with current settings
        """
        rollback_config = self._config.get('rollback_conditions', {})
        
        return RollbackCondition(
            critical_error_rate=rollback_config.get('critical_error_rate', 0.05),
            performance_degradation=rollback_config.get('performance_degradation', 2.0),
            correlation_threshold=rollback_config.get('correlation_threshold', 0.5),
            top_overlap_threshold=rollback_config.get('top_overlap_threshold', 0.5),
            manual_triggers=rollback_config.get('manual_triggers', [])
        )
    
    def should_auto_rollback(self, error_rate: float = 0.0, 
                           performance_ratio: float = 1.0,
                           correlation: float = 1.0,
                           top_overlap: float = 1.0) -> Dict[str, Any]:
        """
        Check if automatic rollback should be triggered based on metrics
        
        Args:
            error_rate: Current error rate (0.0 to 1.0)
            performance_ratio: Performance ratio vs baseline (1.0 = same performance)
            correlation: Correlation between V1 and V2 scores (0.0 to 1.0)
            top_overlap: Overlap ratio of top rankings (0.0 to 1.0)
            
        Returns:
            Dictionary with rollback decision and reasons
        """
        conditions = self.get_rollback_conditions()
        should_rollback = False
        reasons = []
        
        # Check each condition
        if error_rate is not None and error_rate > conditions.critical_error_rate:
            should_rollback = True
            reasons.append(f"Error rate {error_rate:.3f} exceeds threshold {conditions.critical_error_rate:.3f}")
        
        if performance_ratio is not None and performance_ratio > conditions.performance_degradation:
            should_rollback = True
            reasons.append(f"Performance degradation {performance_ratio:.2f}x exceeds threshold {conditions.performance_degradation:.2f}x")
        
        if correlation is not None and correlation < conditions.correlation_threshold:
            should_rollback = True
            reasons.append(f"Correlation {correlation:.3f} below threshold {conditions.correlation_threshold:.3f}")
        
        if top_overlap is not None and top_overlap < conditions.top_overlap_threshold:
            should_rollback = True
            reasons.append(f"Top overlap {top_overlap:.3f} below threshold {conditions.top_overlap_threshold:.3f}")
        
        return {
            'should_rollback': should_rollback,
            'reasons': reasons,
            'auto_rollback_enabled': self.is_enabled('ENABLE_AUTO_ROLLBACK'),
            'metrics': {
                'error_rate': error_rate,
                'performance_ratio': performance_ratio,
                'correlation': correlation,
                'top_overlap': top_overlap
            }
        }
    
    def is_emergency_disabled(self, flag_name: str = 'EMERGENCY_V2_DISABLE') -> bool:
        """
        Check if emergency disable is active
        
        Args:
            flag_name: Name of the emergency disable flag
            
        Returns:
            True if emergency disable is active
        """
        try:
            return self.is_enabled(flag_name)
        except FeatureFlagError:
            # If emergency flag doesn't exist, assume not disabled
            return False
    
    def trigger_emergency_disable(self, reason: str = "Manual trigger") -> bool:
        """
        Trigger emergency disable and send notifications
        
        Args:
            reason: Reason for emergency disable
            
        Returns:
            True if emergency disable was successful
        """
        logger.critical(f"Emergency disable triggered: {reason}")
        
        try:
            # Send Pushover notification if available
            self._send_emergency_notification(reason)
            
            # In a real implementation, this would update the config file
            # For now, we log the event
            logger.critical(f"EMERGENCY_V2_DISABLE triggered at {datetime.now()}: {reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to trigger emergency disable: {e}")
            return False
    
    def _send_emergency_notification(self, reason: str):
        """
        Send emergency notification via Pushover
        
        Args:
            reason: Reason for the emergency
        """
        try:
            # Import here to avoid circular imports
            from investment_toolkit.utilities.notification import NotificationManager
            
            notification_manager = NotificationManager()
            
            title = "🚨 V2システム緊急停止"
            message = f"""V2スコアリングシステムが緊急停止されました。

【停止理由】
{reason}

【発生時刻】
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【対応が必要です】
システム状態を確認し、問題を解決してください。"""
            
            # Send as high priority notification
            success = notification_manager.send_notification(title, message, priority=2)
            
            if success:
                logger.info("Emergency notification sent via Pushover")
            else:
                logger.error("Failed to send emergency notification")
                
        except Exception as e:
            logger.error(f"Failed to send emergency notification: {e}")
    
    def get_rollout_percentage(self, flag_name: str) -> Optional[int]:
        """
        Get the rollout percentage for a specific flag
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            Rollout percentage (0-100) or None if not configured
        """
        if flag_name not in self._flags:
            return None
            
        return self._flags[flag_name].rollout_percentage
    
    def update_rollout_percentage(self, flag_name: str, percentage: int) -> bool:
        """
        Update rollout percentage for a flag (in-memory only)
        
        Args:
            flag_name: Name of the flag
            percentage: New rollout percentage (0-100)
            
        Returns:
            True if update was successful
        """
        if flag_name not in self._flags:
            raise FeatureFlagError(f"Unknown feature flag: {flag_name}")
        
        if not (0 <= percentage <= 100):
            raise ValueError("Rollout percentage must be between 0 and 100")
        
        self._flags[flag_name].rollout_percentage = percentage
        self._clear_cache()  # Clear cache to force re-evaluation
        
        logger.info(f"Updated rollout percentage for {flag_name} to {percentage}%")
        return True
    
    def get_migration_phase_status(self) -> Dict[str, Any]:
        """
        Get current migration phase based on enabled flags
        
        Returns:
            Dictionary with current phase information
        """
        phases = self._config.get('migration_phases', {})
        current_phase = 'phase_0_preparation'
        
        enabled_flags = set(self.get_enabled_flags())
        
        # Determine current phase based on enabled flags
        for phase_name, phase_config in phases.items():
            required_flags = set(phase_config.get('required_flags', []))
            if required_flags.issubset(enabled_flags):
                current_phase = phase_name
        
        phase_info = phases.get(current_phase, {})
        
        return {
            'current_phase': current_phase,
            'description': phase_info.get('description', ''),
            'required_flags': phase_info.get('required_flags', []),
            'validation_criteria': phase_info.get('validation_criteria', []),
            'enabled_flags': list(enabled_flags)
        }


# Convenience functions for easy access
def is_enabled(flag_name: str, config_path: Optional[str] = None, 
               environment: Optional[str] = None) -> bool:
    """
    Convenience function to check if a flag is enabled
    
    Args:
        flag_name: Name of the feature flag
        config_path: Optional path to config file
        environment: Optional environment override
        
    Returns:
        True if flag is enabled, False otherwise
    """
    flags = FeatureFlags.get_instance(config_path, environment)
    return flags.is_enabled(flag_name)


def get_enabled_flags(config_path: Optional[str] = None, 
                     environment: Optional[str] = None) -> List[str]:
    """
    Convenience function to get all enabled flags
    
    Args:
        config_path: Optional path to config file
        environment: Optional environment override
        
    Returns:
        List of enabled flag names
    """
    flags = FeatureFlags.get_instance(config_path, environment)
    return flags.get_enabled_flags()


def should_use_v2_system() -> bool:
    """
    Convenience function to check if V2 scoring system should be used
    
    Returns:
        True if V2 system should be used, False otherwise
    """
    return is_enabled('SCORING_V2_ENABLED')


def should_use_ab_testing() -> bool:
    """
    Convenience function to check if AB testing should be enabled
    
    Returns:
        True if AB testing should be enabled, False otherwise
    """
    return is_enabled('ENABLE_AB_CONCURRENT_RUN')


def should_use_gradual_rollout() -> bool:
    """
    Convenience function to check if gradual V2 rollout should be used
    
    Returns:
        True if gradual rollout should be used, False otherwise
    """
    return is_enabled('GRADUAL_V2_ROLLOUT')


def is_emergency_disabled() -> bool:
    """
    Convenience function to check if V2 system is emergency disabled
    
    Returns:
        True if V2 system is emergency disabled, False otherwise
    """
    flags = FeatureFlags.get_instance()
    return flags.is_emergency_disabled()


def get_v2_rollout_percentage() -> int:
    """
    Convenience function to get V2 rollout percentage
    
    Returns:
        Rollout percentage (0-100)
    """
    flags = FeatureFlags.get_instance()
    percentage = flags.get_rollout_percentage('GRADUAL_V2_ROLLOUT')
    return percentage if percentage is not None else 0


def trigger_v2_emergency_disable(reason: str = "Manual trigger") -> bool:
    """
    Convenience function to trigger V2 emergency disable
    
    Args:
        reason: Reason for emergency disable
        
    Returns:
        True if emergency disable was successful
    """
    flags = FeatureFlags.get_instance()
    return flags.trigger_emergency_disable(reason)


def get_active_pillars() -> List[str]:
    """
    Get list of V2 pillars that are currently enabled
    
    Returns:
        List of active pillar names
    """
    pillar_flags = [
        'ENABLE_VALUE_PILLAR_V2',
        'ENABLE_GROWTH_PILLAR_V2', 
        'ENABLE_QUALITY_PILLAR_V2',
        'ENABLE_MOMENTUM_PILLAR_V2',
        'ENABLE_RISK_PILLAR_V2'
    ]
    
    active_pillars = []
    for flag in pillar_flags:
        if is_enabled(flag):
            pillar_name = flag.replace('ENABLE_', '').replace('_PILLAR_V2', '').lower()
            active_pillars.append(pillar_name)
    
    return active_pillars


if __name__ == "__main__":
    # Example usage and testing
    try:
        flags = FeatureFlags()
        
        print(f"Environment: {flags.environment}")
        print(f"V2 System Enabled: {flags.is_enabled('SCORING_V2_ENABLED')}")
        print(f"AB Testing Enabled: {flags.is_enabled('ENABLE_AB_CONCURRENT_RUN')}")
        
        print("\nEnabled Flags:")
        for flag in flags.get_enabled_flags():
            print(f"  - {flag}")
        
        print(f"\nMigration Phase: {flags.get_migration_phase_status()}")
        
    except Exception as e:
        print(f"Error: {e}")