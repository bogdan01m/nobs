# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NOBS (Neural NetwOrks Benchmark Stash) is an open-source benchmark suite for evaluating real AI hardware performance on practical workloads. The project focuses on reproducible, honest benchmarks across different hardware (M4 Max, RTX 4060, A100, etc.) for:

- **Text Embeddings** (sentence-transformers) - IMPLEMENTED
- **LLM Inference** (via LM Studio) - IMPLEMENTED
- **VLM/Vision-Language tasks** (via LM Studio) - IMPLEMENTED ✨ NEW
- **Power Metrics Monitoring** - IMPLEMENTED ✨ NEW
- **Diffusion image generation** - PLANNED

### Package Management
This project uses `uv` as the package manager (not pip/poetry).

**Python Version**: 3.12+

## Dependencies

Core dependencies (from `pyproject.toml`):
- `torch>=2.9.0` - Deep learning framework
- `transformers>=4.57.1` - Model architectures
- `sentence-transformers>=5.1.1` - Embedding models
- `openai>=2.5.0` - OpenAI API client (for LM Studio)
- `datasets>=4.2.0` - Dataset loading (HuggingFace)
- `accelerate>=1.10.1` - Distributed training utilities
- `psutil>=7.1.0` - System monitoring
- `python-dotenv>=1.1.1` - Environment variable loading
- `requests>=2.32.5` - HTTP client
- `pydantic>=2.12.3` - Data validation
- `scikit-learn>=1.7.2` - ML utilities
- `numpy>=2.3.4`, `tqdm>=4.67.1` - Utilities

Dev dependencies:
- `ruff>=0.8.0` - Linting and formatting
- `mypy>=1.13.0` - Type checking
- `bandit>=1.7.10` - Security scanning
- `pre-commit>=4.3.0` - Git hooks

## Code Quality

**Pre-commit Hooks (`.pre-commit-config.yaml`)**
- Standard checks: trailing whitespace, file endings, YAML syntax, merge conflicts
- Ruff: Fast Python linting and auto-formatting (only `src/` and `main.py`)
- mypy: Type checking with ignore-missing-imports
- bandit: Security vulnerability detection (severity=high, confidence=high)

**GitHub Actions CI/CD (`.github/workflows/code-quality.yml`)**
- Runs on push and pull requests
- Python 3.12 environment
- Executes pre-commit checks on all files
- Cached pre-commit environments for speed

## Makefile Utilities

Project includes a Makefile for common tasks:

```bash
make          # Generate benchmark results tables (default)
make bench    # Start benchmark with uv run main.py
make generate # Generate benchmark results tables from JSON files
make format   # Run pre-commit hooks on all files
make lint     # Run ruff linter only
make clean    # Clean Python cache files
make help     # Show available commands
```

The `make generate` command processes `results/*.json` files and creates markdown tables for the README.
