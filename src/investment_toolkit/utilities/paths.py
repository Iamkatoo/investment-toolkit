"""
Centralized path configuration module for investment-toolkit.

This module provides a unified interface for managing report output paths,
supporting dual-location output (local + iCloud) and environment-based configuration.
"""

from pathlib import Path
from typing import Optional, Union
import os
import shutil
from dataclasses import dataclass


@dataclass
class ReportsConfig:
    """Configuration for report output paths."""

    base_dir: Path
    graphs_dir: Path
    individual_stocks_dir: Path
    mini_json_dir: Path
    archived_dir: Path
    static_dir: Path
    icloud_enabled: bool = False
    icloud_base_dir: Optional[Path] = None

    def __post_init__(self):
        """Ensure all directories exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.individual_stocks_dir.mkdir(parents=True, exist_ok=True)
        self.mini_json_dir.mkdir(parents=True, exist_ok=True)
        self.archived_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)

        if self.icloud_enabled and self.icloud_base_dir:
            self.icloud_base_dir.mkdir(parents=True, exist_ok=True)
            (self.icloud_base_dir / "graphs").mkdir(parents=True, exist_ok=True)
            (self.icloud_base_dir / "individual_stocks").mkdir(parents=True, exist_ok=True)
            (self.icloud_base_dir / "mini_json").mkdir(parents=True, exist_ok=True)

    def get_icloud_path(self, relative_path: Union[str, Path]) -> Optional[Path]:
        """
        Get the iCloud equivalent of a local path.

        Args:
            relative_path: Path relative to base_dir

        Returns:
            iCloud path if enabled, None otherwise
        """
        if not self.icloud_enabled or not self.icloud_base_dir:
            return None

        rel_path = Path(relative_path)
        # Remove base_dir prefix if present
        try:
            rel_to_base = rel_path.relative_to(self.base_dir)
        except ValueError:
            rel_to_base = rel_path

        return self.icloud_base_dir / rel_to_base

    def save_to_all(
        self,
        filename: Union[str, Path],
        content: Union[str, bytes],
        subdirectory: str = "graphs"
    ) -> tuple[Path, Optional[Path]]:
        """
        Save content to both local and iCloud locations.

        Args:
            filename: Name of the file to save
            content: Content to write (str or bytes)
            subdirectory: Subdirectory within reports (default: "graphs")

        Returns:
            Tuple of (local_path, icloud_path)
        """
        # Determine the local directory
        if subdirectory == "graphs":
            local_dir = self.graphs_dir
        elif subdirectory == "individual_stocks":
            local_dir = self.individual_stocks_dir
        elif subdirectory == "mini_json":
            local_dir = self.mini_json_dir
        elif subdirectory == "archived":
            local_dir = self.archived_dir
        elif subdirectory == "static":
            local_dir = self.static_dir
        else:
            local_dir = self.base_dir / subdirectory
            local_dir.mkdir(parents=True, exist_ok=True)

        local_path = local_dir / filename

        # Write to local location
        if isinstance(content, str):
            local_path.write_text(content, encoding="utf-8")
        else:
            local_path.write_bytes(content)

        # Write to iCloud if enabled
        icloud_path = None
        if self.icloud_enabled and self.icloud_base_dir:
            icloud_dir = self.icloud_base_dir / subdirectory
            icloud_dir.mkdir(parents=True, exist_ok=True)
            icloud_path = icloud_dir / filename

            if isinstance(content, str):
                icloud_path.write_text(content, encoding="utf-8")
            else:
                icloud_path.write_bytes(content)

        return local_path, icloud_path

    def copy_to_all(
        self,
        source_path: Union[str, Path],
        destination_filename: Optional[str] = None,
        subdirectory: str = "graphs"
    ) -> tuple[Path, Optional[Path]]:
        """
        Copy a file to both local and iCloud locations.

        Args:
            source_path: Path to the source file
            destination_filename: Optional destination filename (defaults to source filename)
            subdirectory: Subdirectory within reports (default: "graphs")

        Returns:
            Tuple of (local_path, icloud_path)
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        dest_filename = destination_filename or source.name

        # Determine the local directory
        if subdirectory == "graphs":
            local_dir = self.graphs_dir
        elif subdirectory == "individual_stocks":
            local_dir = self.individual_stocks_dir
        elif subdirectory == "mini_json":
            local_dir = self.mini_json_dir
        elif subdirectory == "archived":
            local_dir = self.archived_dir
        elif subdirectory == "static":
            local_dir = self.static_dir
        else:
            local_dir = self.base_dir / subdirectory
            local_dir.mkdir(parents=True, exist_ok=True)

        local_path = local_dir / dest_filename
        shutil.copy2(source, local_path)

        # Copy to iCloud if enabled
        icloud_path = None
        if self.icloud_enabled and self.icloud_base_dir:
            icloud_dir = self.icloud_base_dir / subdirectory
            icloud_dir.mkdir(parents=True, exist_ok=True)
            icloud_path = icloud_dir / dest_filename
            shutil.copy2(source, icloud_path)

        return local_path, icloud_path


def get_reports_config() -> ReportsConfig:
    """
    Get the current reports configuration.

    Configuration is determined by environment variables:
    - REPORTS_BASE_DIR: Base directory for reports (default: ./reports)
    - ENABLE_ICLOUD_SYNC: Whether to enable iCloud sync (default: false)
    - ICLOUD_REPORTS_DIR: iCloud reports directory (required if ENABLE_ICLOUD_SYNC=true)

    Returns:
        ReportsConfig instance
    """
    # Get base directory from environment or use default (investment-reports repo)
    # Default points to investment-reports repository (sibling of investment-toolkit)
    default_reports_dir = str(Path(__file__).parent.parent.parent.parent.parent / "investment-reports")
    base_dir_str = os.getenv("REPORTS_BASE_DIR", default_reports_dir)
    base_dir = Path(base_dir_str).expanduser().resolve()

    # Check if iCloud sync is enabled (default: false for security)
    icloud_enabled = os.getenv("ENABLE_ICLOUD_SYNC", "false").lower() in ("true", "1", "yes")

    # Get iCloud directory
    icloud_base_dir = None
    if icloud_enabled:
        icloud_path_str = os.getenv("ICLOUD_REPORTS_DIR")
        if icloud_path_str:
            icloud_base_dir = Path(icloud_path_str).expanduser().resolve()
        else:
            # iCloud is enabled but no path provided - disable it
            icloud_enabled = False

    return ReportsConfig(
        base_dir=base_dir,
        graphs_dir=base_dir / "graphs",
        individual_stocks_dir=base_dir / "individual_stocks",
        mini_json_dir=base_dir / "mini_json",
        archived_dir=base_dir / "archived",
        static_dir=base_dir / "static",
        icloud_enabled=icloud_enabled,
        icloud_base_dir=icloud_base_dir
    )


# Singleton instance for convenience
_reports_config: Optional[ReportsConfig] = None


def get_or_create_reports_config() -> ReportsConfig:
    """
    Get or create a singleton ReportsConfig instance.

    This is useful for scripts that need to access the configuration multiple times.

    Returns:
        ReportsConfig instance
    """
    global _reports_config
    if _reports_config is None:
        _reports_config = get_reports_config()
    return _reports_config


def reset_reports_config():
    """Reset the singleton ReportsConfig instance. Useful for testing."""
    global _reports_config
    _reports_config = None


# Backward compatibility: provide direct access to common paths
def get_reports_base_dir() -> Path:
    """Get the base reports directory."""
    return get_or_create_reports_config().base_dir


def get_graphs_dir() -> Path:
    """Get the graphs directory."""
    return get_or_create_reports_config().graphs_dir


def get_individual_stocks_dir() -> Path:
    """Get the individual stocks directory."""
    return get_or_create_reports_config().individual_stocks_dir


def get_mini_json_dir() -> Path:
    """Get the mini JSON directory."""
    return get_or_create_reports_config().mini_json_dir


def get_archived_dir() -> Path:
    """Get the archived reports directory."""
    return get_or_create_reports_config().archived_dir


def get_static_dir() -> Path:
    """Get the static files directory."""
    return get_or_create_reports_config().static_dir
