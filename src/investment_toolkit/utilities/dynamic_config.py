#!/usr/bin/env python3
"""
Dynamic Configuration Management System - V2 Migration

This module provides dynamic configuration management capabilities for the
V2 migration system, allowing runtime changes to feature flags and settings
without requiring system restarts.

Implementation Task 3.2: Dynamic Configuration Management System
- Runtime flag updates without restart
- Configuration change history tracking  
- Dependency validation for flag changes
- Safe rollback to previous states
- File system monitoring for config changes
- Integration with Pushover notification system

Key Features:
- Hot-reloading of configuration files
- Change validation and dependency checking  
- Rollback to previous configuration states
- Comprehensive audit logging
- Thread-safe operations
- Integration with existing notification system

Usage:
    from investment_toolkit.utilities.dynamic_config import DynamicConfigManager

    config_manager = DynamicConfigManager()
    config_manager.update_flag('GRADUAL_V2_ROLLOUT', True, rollout_percentage=25)
    config_manager.rollback_to_previous_state()
    history = config_manager.get_change_history()

Created: 2025-09-15
Author: Claude Code Assistant
"""

import os
import yaml
import logging
import threading
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import shutil
import fcntl
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    logger.warning("Watchdog library not available. File watching disabled.")
    WATCHDOG_AVAILABLE = False
    
    # Create dummy classes for compatibility
    class Observer:
        def __init__(self): pass
        def schedule(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
        def join(self, timeout=None): pass
        def is_alive(self): return False
    
    class FileSystemEventHandler:
        def __init__(self): pass
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from investment_toolkit.utilities.feature_flags import FeatureFlags, FeatureFlagError, DependencyValidationError
from investment_toolkit.utilities.notification import NotificationManager

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Record of a configuration change"""
    timestamp: datetime
    change_type: str  # 'flag_update', 'rollout_percentage', 'emergency_disable', 'rollback'
    flag_name: Optional[str] = None
    old_value: Any = None
    new_value: Any = None
    user: str = field(default_factory=lambda: os.getenv('USER', 'system'))
    reason: str = ""
    success: bool = True
    error_message: Optional[str] = None


@dataclass 
class ConfigSnapshot:
    """Snapshot of configuration state"""
    timestamp: datetime
    config_data: Dict[str, Any]
    change_id: str
    description: str = ""


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration changes"""
    
    def __init__(self, dynamic_config_manager):
        self.config_manager = dynamic_config_manager
        self.last_modified = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Check if it's a config file we care about
        config_files = [
            'feature_flags.yaml',
            'ab_feature_flags.yaml'
        ]
        
        if any(config_file in event.src_path for config_file in config_files):
            # Debounce rapid file changes
            current_time = time.time()
            if event.src_path in self.last_modified:
                if current_time - self.last_modified[event.src_path] < 1.0:
                    return
                    
            self.last_modified[event.src_path] = current_time
            
            logger.info(f"Configuration file changed: {event.src_path}")
            self.config_manager.reload_from_file(event.src_path)


class DynamicConfigManager:
    """
    Dynamic Configuration Management System
    
    Provides runtime configuration changes, validation, history tracking,
    and safe rollback capabilities for the V2 migration system.
    """
    
    def __init__(self, config_path: Optional[str] = None, watch_files: bool = True):
        """
        Initialize Dynamic Configuration Manager
        
        Args:
            config_path: Path to primary configuration file
            watch_files: Whether to watch configuration files for changes
        """
        self.config_path = config_path or self._get_default_config_path()
        self.feature_flags = FeatureFlags.get_instance()
        self.notification_manager = NotificationManager()
        
        # State management
        self._lock = threading.RLock()
        self._change_history: List[ConfigChange] = []
        self._snapshots: List[ConfigSnapshot] = []
        self._max_history_size = 1000
        self._max_snapshots = 50
        
        # File watching
        self.watch_files = watch_files
        self.observer = None
        if watch_files:
            self._setup_file_watcher()
        
        # Load initial state
        self._create_initial_snapshot()
        
        logger.info("Dynamic Configuration Manager initialized")
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        return str(project_root / 'config' / 'feature_flags.yaml')
    
    def _setup_file_watcher(self):
        """Setup file system watcher for configuration changes"""
        if not WATCHDOG_AVAILABLE:
            logger.info("File watching disabled (watchdog library not available)")
            self.observer = None
            return
            
        try:
            self.observer = Observer()
            config_dir = Path(self.config_path).parent
            
            event_handler = ConfigFileHandler(self)
            self.observer.schedule(event_handler, str(config_dir), recursive=False)
            self.observer.start()
            
            logger.info(f"File watcher setup for directory: {config_dir}")
            
        except Exception as e:
            logger.error(f"Failed to setup file watcher: {e}")
            self.observer = None
    
    def _create_initial_snapshot(self):
        """Create initial configuration snapshot"""
        with self._lock:
            try:
                current_config = self._load_config_file()
                snapshot = ConfigSnapshot(
                    timestamp=datetime.now(),
                    config_data=current_config,
                    change_id="initial",
                    description="Initial configuration snapshot"
                )
                
                self._snapshots.append(snapshot)
                logger.info("Initial configuration snapshot created")
                
            except Exception as e:
                logger.error(f"Failed to create initial snapshot: {e}")
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {}
    
    @contextmanager
    def _file_lock(self, file_path: str):
        """Context manager for file locking"""
        lock_file = f"{file_path}.lock"
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
        except IOError:
            raise Exception(f"Could not acquire lock for {file_path}")
        finally:
            try:
                os.remove(lock_file)
            except:
                pass
    
    def update_flag(self, flag_name: str, enabled: bool, 
                   rollout_percentage: Optional[int] = None,
                   reason: str = "", validate_dependencies: bool = True) -> bool:
        """
        Update a feature flag with validation and history tracking
        
        Args:
            flag_name: Name of the flag to update
            enabled: New enabled state
            rollout_percentage: Optional rollout percentage for gradual flags
            reason: Reason for the change
            validate_dependencies: Whether to validate dependencies
            
        Returns:
            True if update was successful
        """
        with self._lock:
            change = ConfigChange(
                timestamp=datetime.now(),
                change_type='flag_update',
                flag_name=flag_name,
                reason=reason
            )
            
            try:
                # Get current values
                current_config = self._load_config_file()
                flag_config = current_config.get('feature_flags', {}).get(flag_name)
                
                if not flag_config:
                    raise FeatureFlagError(f"Unknown flag: {flag_name}")
                
                # Store old values
                environment = self.feature_flags.environment
                change.old_value = {
                    'enabled': flag_config.get('environments', {}).get(environment, flag_config.get('default', False)),
                    'rollout_percentage': flag_config.get('rollout_percentage')
                }
                
                # Validate dependencies if requested
                if validate_dependencies and enabled:
                    self._validate_flag_dependencies(flag_name, current_config)
                
                # Create backup snapshot before making changes
                self._create_snapshot(f"Before updating {flag_name}")
                
                # Update the configuration
                flag_config['environments'][environment] = enabled
                if rollout_percentage is not None:
                    flag_config['rollout_percentage'] = rollout_percentage
                
                # Save to file
                self._save_config_file(current_config)
                
                # Update in-memory flags
                self.feature_flags.reload_configuration()
                
                # Record successful change
                change.new_value = {
                    'enabled': enabled,
                    'rollout_percentage': rollout_percentage
                }
                change.success = True
                self._add_change_record(change)
                
                # Send notification
                self._send_change_notification(flag_name, enabled, rollout_percentage, reason)
                
                logger.info(f"Successfully updated flag {flag_name}: enabled={enabled}, rollout={rollout_percentage}")
                return True
                
            except Exception as e:
                change.success = False
                change.error_message = str(e)
                self._add_change_record(change)
                
                logger.error(f"Failed to update flag {flag_name}: {e}")
                return False
    
    def _validate_flag_dependencies(self, flag_name: str, config_data: Dict[str, Any]):
        """Validate flag dependencies"""
        flag_config = config_data['feature_flags'][flag_name]
        dependencies = flag_config.get('dependencies', [])
        
        environment = self.feature_flags.environment
        
        for dep_flag in dependencies:
            if dep_flag not in config_data['feature_flags']:
                raise DependencyValidationError(f"Dependency {dep_flag} not found")
            
            dep_config = config_data['feature_flags'][dep_flag]
            dep_enabled = dep_config.get('environments', {}).get(environment, dep_config.get('default', False))
            
            if not dep_enabled:
                raise DependencyValidationError(f"Dependency {dep_flag} is not enabled")
    
    def _save_config_file(self, config_data: Dict[str, Any]):
        """Save configuration data to file"""
        try:
            with self._file_lock(self.config_path):
                # Create backup
                backup_path = f"{self.config_path}.backup.{int(time.time())}"
                shutil.copy2(self.config_path, backup_path)
                
                # Save new configuration
                with open(self.config_path, 'w', encoding='utf-8') as file:
                    yaml.dump(config_data, file, default_flow_style=False, allow_unicode=True)
                    
                logger.debug(f"Configuration saved to {self.config_path}")
                
        except Exception as e:
            raise Exception(f"Failed to save configuration: {e}")
    
    def emergency_disable(self, flag_name: str = 'EMERGENCY_V2_DISABLE', reason: str = "Manual trigger") -> bool:
        """
        Trigger emergency disable with immediate effect
        
        Args:
            flag_name: Name of emergency flag
            reason: Reason for emergency disable
            
        Returns:
            True if emergency disable was successful
        """
        logger.critical(f"Emergency disable triggered: {reason}")
        
        try:
            # Update the emergency flag
            success = self.update_flag(flag_name, True, reason=f"EMERGENCY: {reason}", validate_dependencies=False)
            
            if success:
                # Send emergency notification
                self.notification_manager.send_notification(
                    "🚨 緊急停止実行",
                    f"V2システムが緊急停止されました。\n\n理由: {reason}\n時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    priority=2  # Emergency priority
                )
                
                # Also trigger the feature flags emergency system
                self.feature_flags.trigger_emergency_disable(reason)
                
            return success
            
        except Exception as e:
            logger.error(f"Emergency disable failed: {e}")
            return False
    
    def rollback_to_previous_state(self) -> bool:
        """
        Rollback to the previous configuration state
        
        Returns:
            True if rollback was successful
        """
        with self._lock:
            if len(self._snapshots) < 2:
                logger.warning("No previous state available for rollback")
                return False
            
            change = ConfigChange(
                timestamp=datetime.now(),
                change_type='rollback',
                reason="Rollback to previous state"
            )
            
            try:
                # Get the previous snapshot (second to last)
                previous_snapshot = self._snapshots[-2]
                
                # Save current state as rollback point
                self._create_snapshot("Before rollback")
                
                # Restore previous configuration
                self._save_config_file(previous_snapshot.config_data)
                
                # Reload configuration
                self.feature_flags.reload_configuration()
                
                change.success = True
                change.old_value = "current_state"
                change.new_value = previous_snapshot.change_id
                self._add_change_record(change)
                
                # Send notification
                self.notification_manager.send_notification(
                    "🔄 設定ロールバック実行",
                    f"設定が前の状態にロールバックされました。\n\nロールバック先: {previous_snapshot.description}\n時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    priority=1
                )
                
                logger.info(f"Successfully rolled back to snapshot: {previous_snapshot.change_id}")
                return True
                
            except Exception as e:
                change.success = False
                change.error_message = str(e)
                self._add_change_record(change)
                
                logger.error(f"Rollback failed: {e}")
                return False
    
    def get_change_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get configuration change history
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            List of configuration changes
        """
        with self._lock:
            history = self._change_history.copy()
            if limit:
                history = history[-limit:]
            
            return [asdict(change) for change in history]
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration
        
        Returns:
            Validation results
        """
        try:
            config_data = self._load_config_file()
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'flag_count': len(config_data.get('feature_flags', {})),
                'dependency_validation': {}
            }
            
            # Validate flag dependencies
            dependency_validation = self.feature_flags.validate_all_dependencies()
            validation_results['dependency_validation'] = dependency_validation
            
            # Check for failed dependencies
            failed_deps = [flag for flag, valid in dependency_validation.items() if not valid]
            if failed_deps:
                validation_results['valid'] = False
                validation_results['errors'].extend([f"Dependency validation failed for: {flag}" for flag in failed_deps])
            
            # Validate rollout percentages
            for flag_name, flag_config in config_data.get('feature_flags', {}).items():
                rollout_pct = flag_config.get('rollout_percentage')
                if rollout_pct is not None:
                    if not isinstance(rollout_pct, int) or not (0 <= rollout_pct <= 100):
                        validation_results['warnings'].append(f"Invalid rollout percentage for {flag_name}: {rollout_pct}")
            
            logger.info(f"Configuration validation completed: {'VALID' if validation_results['valid'] else 'INVALID'}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'flag_count': 0,
                'dependency_validation': {}
            }
    
    def reload_from_file(self, file_path: Optional[str] = None):
        """
        Reload configuration from file
        
        Args:
            file_path: Optional specific file path to reload
        """
        try:
            logger.info(f"Reloading configuration from file: {file_path or self.config_path}")
            
            # Create snapshot before reload
            self._create_snapshot("Before file reload")
            
            # Reload feature flags
            self.feature_flags.reload_configuration()
            
            # Record the reload
            change = ConfigChange(
                timestamp=datetime.now(),
                change_type='file_reload',
                reason=f"File change detected: {file_path}",
                success=True
            )
            self._add_change_record(change)
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def _create_snapshot(self, description: str = ""):
        """Create a configuration snapshot"""
        try:
            config_data = self._load_config_file()
            snapshot = ConfigSnapshot(
                timestamp=datetime.now(),
                config_data=config_data,
                change_id=f"snapshot_{int(time.time())}",
                description=description
            )
            
            self._snapshots.append(snapshot)
            
            # Limit snapshot history
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots:]
                
            logger.debug(f"Configuration snapshot created: {description}")
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
    
    def _add_change_record(self, change: ConfigChange):
        """Add a change record to history"""
        self._change_history.append(change)
        
        # Limit history size
        if len(self._change_history) > self._max_history_size:
            self._change_history = self._change_history[-self._max_history_size:]
    
    def _send_change_notification(self, flag_name: str, enabled: bool, 
                                rollout_percentage: Optional[int], reason: str):
        """Send notification about configuration change"""
        try:
            title = "⚙️ 設定変更通知"
            
            message = f"""フィーチャーフラグが更新されました。

【変更内容】
フラグ名: {flag_name}
状態: {'有効' if enabled else '無効'}"""
            
            if rollout_percentage is not None:
                message += f"\nロールアウト割合: {rollout_percentage}%"
                
            if reason:
                message += f"\n理由: {reason}"
                
            message += f"\n\n変更時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.notification_manager.send_notification(title, message, priority=1)
            
        except Exception as e:
            logger.error(f"Failed to send change notification: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current configuration status
        
        Returns:
            Current status information
        """
        try:
            enabled_flags = self.feature_flags.get_enabled_flags()
            migration_phase = self.feature_flags.get_migration_phase_status()
            validation = self.validate_configuration()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'enabled_flags': enabled_flags,
                'migration_phase': migration_phase,
                'validation': validation,
                'recent_changes': len([c for c in self._change_history if c.timestamp > datetime.now() - timedelta(hours=24)]),
                'emergency_disabled': self.feature_flags.is_emergency_disabled(),
                'file_watching': self.observer is not None and self.observer.is_alive()
            }
            
        except Exception as e:
            logger.error(f"Failed to get current status: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
            logger.info("Dynamic Configuration Manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Convenience functions for easy access
_config_manager_instance = None
_config_manager_lock = threading.Lock()


def get_config_manager() -> DynamicConfigManager:
    """Get singleton instance of Dynamic Configuration Manager"""
    global _config_manager_instance
    
    if _config_manager_instance is None:
        with _config_manager_lock:
            if _config_manager_instance is None:
                _config_manager_instance = DynamicConfigManager()
    
    return _config_manager_instance


def update_flag_runtime(flag_name: str, enabled: bool, rollout_percentage: Optional[int] = None, reason: str = "") -> bool:
    """
    Convenience function to update a flag at runtime
    
    Args:
        flag_name: Name of the flag
        enabled: New enabled state
        rollout_percentage: Optional rollout percentage
        reason: Reason for change
        
    Returns:
        True if successful
    """
    manager = get_config_manager()
    return manager.update_flag(flag_name, enabled, rollout_percentage, reason)


def trigger_emergency_disable(reason: str = "Manual trigger") -> bool:
    """
    Convenience function to trigger emergency disable
    
    Args:
        reason: Reason for emergency disable
        
    Returns:
        True if successful
    """
    manager = get_config_manager()
    return manager.emergency_disable(reason=reason)


def rollback_configuration() -> bool:
    """
    Convenience function to rollback configuration
    
    Returns:
        True if successful
    """
    manager = get_config_manager()
    return manager.rollback_to_previous_state()


def get_configuration_status() -> Dict[str, Any]:
    """
    Convenience function to get configuration status
    
    Returns:
        Current configuration status
    """
    manager = get_config_manager()
    return manager.get_current_status()


if __name__ == "__main__":
    # Example usage and testing
    try:
        with DynamicConfigManager() as config_manager:
            print("Dynamic Configuration Manager Test")
            print("=" * 50)
            
            # Get current status
            status = config_manager.get_current_status()
            print(f"Current Status: {status}")
            
            # Validate configuration
            validation = config_manager.validate_configuration()
            print(f"Validation: {validation}")
            
            # Get change history
            history = config_manager.get_change_history(limit=5)
            print(f"Recent Changes: {len(history)}")
            
            print("Dynamic Configuration Manager test completed successfully")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()