# Guidelines for Claude Code Assistant

## Repository Context

**This is a PUBLIC repository.** All modifications must respect the following principles:

### Security and Privacy

1. **Never hardcode sensitive information**
   - No API keys, passwords, or credentials in code
   - No absolute file paths that reveal user/system information
   - No personal data or portfolio information

2. **Use environment variables for configuration**
   - All sensitive or system-specific settings go in `.env` (gitignored)
   - Update `.env.example` with placeholder values and clear documentation
   - Ensure all modules that need env vars call `load_dotenv()`

3. **Path handling**
   - Use relative paths or environment variables for external resources
   - Default paths should work for typical multi-repo setups
   - Document assumptions about directory structure

### Multi-Repository Architecture

This toolkit works with companion repositories:

- **investment-toolkit** (public): Core Python package - THIS REPO
- **investment-reports** (local): Generated reports and visualizations
- **investment-workspace** (private): Personal scripts and portfolio data

Key environment variables for integration:
- `REPORTS_BASE_DIR`: Output directory for reports (default: `../investment-reports`)
- `INVESTMENT_WORKSPACE_ROOT`: Path to workspace repo (default: `../investment-workspace`)
- `PORTFOLIO_JSON_PATH`: Portfolio configuration file path

### Code Modification Guidelines

When making changes:

1. **Check for hardcoded paths**
   - Look for absolute paths like `/Users/...` or `C:\...`
   - Replace with environment variables or relative paths
   - Provide sensible defaults for missing env vars

2. **Verify .gitignore coverage**
   - Ensure `.env` is ignored (not `.env.example`)
   - Check that any new config files are properly ignored
   - Personal data files should never be tracked

3. **Document environment variables**
   - Add new variables to `.env.example` with comments
   - Update README.md if adding significant new config
   - Use descriptive variable names

4. **Test with clean environment**
   - Ensure code works without existing `.env`
   - Verify defaults are reasonable
   - Check that error messages guide users to proper configuration

### Recent Changes (2025-12-21)

**Path Configuration Migration**
- Migrated from hardcoded paths to environment-based configuration
- Reports now output to separate `investment-reports` repository
- Portfolio data sourced from `investment-workspace` repository
- Added iCloud sync support via environment variables

**Modified Files**
- `src/investment_toolkit/utilities/paths.py`: Centralized path configuration
- `src/investment_toolkit/analysis/portfolio_utils.py`: Added dotenv support
- `src/investment_toolkit/api/watchlist_api.py`: Workspace integration
- `.env.example`: Comprehensive configuration template

### Best Practices

1. **Always run `load_dotenv()` early in modules that need configuration**
2. **Use `get_or_create_reports_config()` for report paths** (from `utilities.paths`)
3. **Test changes assuming this is someone else's machine**
4. **Document breaking changes in commit messages**
5. **Prefer configuration over convention when paths are involved**

## Helpful Commands

```bash
# Verify environment setup
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('REPORTS_BASE_DIR:', os.getenv('REPORTS_BASE_DIR'))"

# Check for hardcoded paths (macOS/Linux)
rg -i "/users/|c:\\\\" --type py

# List gitignored files
git status --ignored
```

## Questions to Ask Before Committing

- [ ] Does this change expose any sensitive information?
- [ ] Are all paths configurable via environment variables?
- [ ] Is `.env.example` updated if new variables were added?
- [ ] Would this work on someone else's machine with their own `.env`?
- [ ] Are error messages helpful for missing configuration?
