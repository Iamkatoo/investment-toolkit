# Investment Toolkit

Core Python package for investment analysis and equity market research.

## Overview

This package provides a comprehensive toolkit for:
- Market data acquisition and processing
- Technical and fundamental analysis
- Portfolio backtesting and optimization
- Automated report generation
- Real-time monitoring and alerting

## Installation

### Development Installation

```bash
# Clone and install in editable mode
pip install -e ".[dev]"
```

### Production Installation

```bash
pip install investment-toolkit
```

## Package Structure

```
investment_toolkit/
├── analysis/          # Report generation and analysis (22 modules)
├── api/              # REST API integrations (6 modules)
├── backtest/         # Backtesting framework (9 modules)
├── data/             # Data acquisition (4 modules)
├── database/         # Database utilities
├── scoring/          # Stock scoring algorithms (7 modules)
├── utilities/        # Common utilities (11 modules)
└── [8 more subdirectories]
```

## Requirements

- Python >= 3.14.2
- PostgreSQL database (for production use)
- Valid API keys for data providers (FMP, FRED, etc.)

## Configuration

### Environment Variables

This package uses environment variables for configuration. Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

### Key Configuration Options

**Database Configuration**
- `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_HOST`, `DB_PORT`

**Reports Output**
- `REPORTS_BASE_DIR`: Base directory for report outputs (default: `../investment-reports`)
- `ENABLE_ICLOUD_SYNC`: Enable iCloud sync for reports (default: `false`)
- `ICLOUD_REPORTS_DIR`: iCloud reports directory (required if sync enabled)

**External Repository Integration**
- `INVESTMENT_WORKSPACE_ROOT`: Path to investment-workspace repository (default: `../investment-workspace`)
- `PORTFOLIO_JSON_PATH`: Path to portfolio.json file (default: `../investment-workspace/config/portfolio.json`)

**API Keys**
- `FMP_API_KEY_PRIMARY`, `FMP_API_KEY_SECONDARY`: Financial Modeling Prep API keys
- `FRED_API_KEY`: Federal Reserve Economic Data API key
- `JQUANTS_EMAIL`, `JQUANTS_PASSWORD`: J-Quants API credentials

See [.env.example](.env.example) for all available configuration options.

### Multi-Repository Setup

This toolkit is designed to work with companion repositories:

- **investment-toolkit** (this repo): Core Python package
- **investment-reports**: Generated reports and visualizations (gitignored)
- **investment-workspace**: Personal scripts and portfolio configuration (private)

The `.env` file (gitignored) contains paths to these repositories, allowing secure separation of public code and private data.

## Usage

```python
from investment_toolkit.utilities.paths import get_reports_config
from investment_toolkit.analysis import daily_report

# Get report configuration
config = get_reports_config()
print(f"Reports directory: {config.base_dir}")

# Generate daily report
daily_report.generate()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
