# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Create environment: `make install`
- Run server: `make run`
- Lint code: `make lint`
- Format code: `make format`
- Clean project: `make clean`

## Code Style Guidelines
- **Imports**: Standard library first, third-party second, project modules third
- **Naming**: snake_case for variables/functions, _prefix for private functions
- **Type Hints**: Use typing annotations (Dict, Any, Optional, Tuple) consistently
- **Docstrings**: Triple-quoted docstrings with brief description and detailed explanation
- **Error Handling**: Catch specific exceptions, log with appropriate level, use fallbacks
- **Formatting**: 4-space indentation, ~100-120 char line limit
- **Logging**: Use module-level logger for all operations, with appropriate severity levels
- **Comments**: Document complex logic and data transformations
- **Null Safety**: Add explicit null checks before accessing potentially missing data

Always run `make lint` and `make format` before committing changes.