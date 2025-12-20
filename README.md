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

Copy `.env.example` to `.env` and configure your environment variables:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for all available configuration options.

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
