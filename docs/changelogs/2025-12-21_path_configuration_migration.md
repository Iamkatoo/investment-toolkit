# Path Configuration Migration

**Date:** 2025-12-21
**Type:** Configuration, Security
**Impact:** Breaking change for existing installations

## Summary

Migrated from hardcoded file paths to environment-based configuration to support:
- Public repository distribution (security)
- Multi-repository architecture
- Flexible deployment scenarios
- iCloud sync support

## Changes

### 1. Centralized Path Configuration

**File:** `src/investment_toolkit/utilities/paths.py`

- Added `get_reports_config()` function for centralized path management
- Implemented `ReportsConfig` dataclass for type-safe configuration
- Default paths now point to sibling `investment-reports` repository
- Added iCloud sync support via environment variables

**Environment Variables:**
- `REPORTS_BASE_DIR`: Base directory for reports (default: `../investment-reports`)
- `ENABLE_ICLOUD_SYNC`: Enable dual-location output (default: `false`)
- `ICLOUD_REPORTS_DIR`: iCloud reports directory

### 2. Portfolio Configuration

**File:** `src/investment_toolkit/analysis/portfolio_utils.py`

- Added `load_dotenv()` to ensure environment variables are loaded
- Portfolio path now configurable via `PORTFOLIO_JSON_PATH`
- Default points to `investment-workspace` repository

**Environment Variables:**
- `PORTFOLIO_JSON_PATH`: Path to portfolio.json file (default: `../investment-workspace/config/portfolio.json`)

### 3. Workspace Integration

**File:** `src/investment_toolkit/api/watchlist_api.py`

- Added `load_dotenv()` for environment variable support
- Script execution now uses `INVESTMENT_WORKSPACE_ROOT`
- Report serving uses `get_or_create_reports_config()`

**Environment Variables:**
- `INVESTMENT_WORKSPACE_ROOT`: Path to workspace repository (default: `../investment-workspace`)

### 4. Documentation Updates

**Files:**
- `README.md`: Added comprehensive configuration section
- `CLAUDE.md`: Created guidelines for AI-assisted development
- `.env.example`: Updated with all configuration options

### 5. Database Configuration

**File:** `.env` (gitignored)

- Added database credentials configuration
- Configured for local PostgreSQL setup

## Migration Guide

### For Existing Users

1. **Create `.env` file:**
   ```bash
   cp .env.example .env
   ```

2. **Configure paths in `.env`:**
   ```bash
   # Database
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_NAME=investment

   # Reports
   REPORTS_BASE_DIR=/path/to/investment-reports
   ENABLE_ICLOUD_SYNC=true
   ICLOUD_REPORTS_DIR=~/Library/Mobile Documents/com~apple~CloudDocs/reports

   # Workspace
   INVESTMENT_WORKSPACE_ROOT=/path/to/investment-workspace
   PORTFOLIO_JSON_PATH=/path/to/investment-workspace/config/portfolio.json
   ```

3. **Verify configuration:**
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); from investment_toolkit.utilities.paths import get_or_create_reports_config; config = get_or_create_reports_config(); print(f'Reports: {config.base_dir}'); print(f'iCloud: {config.icloud_base_dir}')"
   ```

### For New Users

1. Clone companion repositories:
   ```bash
   cd /your/investment/directory
   git clone <investment-toolkit-url>
   git clone <investment-reports-url>  # or create empty directory
   git clone <investment-workspace-url>  # private repo
   ```

2. Follow configuration steps above

## Breaking Changes

- **Hardcoded paths removed:** Code previously using `./reports/` now uses environment-configured paths
- **Portfolio location:** Default portfolio path changed from `./config/portfolio.json` to `../investment-workspace/config/portfolio.json`
- **Script execution:** `generate_single_stock_report.py` now executed from workspace repository

## Security Improvements

- ✅ No sensitive information in code
- ✅ All paths configurable via gitignored `.env`
- ✅ `.env.example` contains only placeholders
- ✅ Repository safe for public distribution

## Affected Modules

### Core Changes
- `src/investment_toolkit/utilities/paths.py`
- `src/investment_toolkit/utilities/config.py`

### Updated Integrations
- `src/investment_toolkit/analysis/daily_report.py`
- `src/investment_toolkit/analysis/portfolio_utils.py`
- `src/investment_toolkit/api/watchlist_api.py`

### Configuration
- `.env.example`
- `.gitignore` (already had `.env` excluded)

## Testing

Verified:
- ✅ Reports output to correct directories
- ✅ iCloud sync works when enabled
- ✅ Portfolio data loads from workspace repository
- ✅ API endpoints use correct paths
- ✅ Clean environment works with defaults

## Future Considerations

1. **Configuration validation:** Add startup checks for required paths
2. **Migration script:** Create automated migration helper
3. **Docker support:** Add Docker configuration for reproducible environments
4. **Path discovery:** Auto-detect companion repositories in parent directory

## Related Issues

- Addresses security concerns for public repository release
- Enables multi-user deployments
- Supports cloud storage integration (iCloud, Dropbox, etc.)

## Rollback Instructions

If needed, revert to commit prior to 2025-12-21 and restore hardcoded paths. Note: This will expose sensitive information if repository is public.
