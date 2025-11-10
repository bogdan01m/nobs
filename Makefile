.PHONY: all generate format lint clean docs docs-serve docs-build help

# Default target - generate results only
all: generate

# Generate benchmark results tables
generate:
	@echo "📊 Generating benchmark results tables..."
	@uv run python src/generate_results_table.py
	@echo "✨ Done! Run 'make format' to run pre-commit hooks."

bench:
	@echo "🆕 Starting La Perf benchmark"
	@uv run python main.py
	@echo "✨ Done! Run 'make generate' to update results in README.md"

# Run pre-commit hooks on all files
format:
	@echo "🔧 Running pre-commit hooks..."
	@uv run pre-commit run --all-files

# Run linting only (ruff)
lint:
	@echo "🔍 Running ruff linter..."
	@uvx ruff check src/ main.py

# Clean Python cache files
clean:
	@echo "🧹 Cleaning cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Serve documentation locally
docs-serve:
	@echo "📚 Starting documentation server..."
	@echo "📊 Copying plot files..."
	@rm -rf docs/plots && mkdir -p docs/plots
	@cp results/plots/*.png docs/plots/ 2>/dev/null || true
	@uv run --group docs mkdocs serve --watch-theme --livereload

# Build documentation site
docs-build:
	@echo "📚 Building documentation site..."
	@echo "📊 Copying plot files..."
	@rm -rf docs/plots && mkdir -p docs/plots
	@cp results/plots/*.png docs/plots/ 2>/dev/null || true
	@uv run --group docs mkdocs build

# Alias for docs-serve
docs: docs-serve

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make             - Generate benchmark results tables (default)"
	@echo "  make bench       - Run benchmarks"
	@echo "  make generate    - Generate benchmark results tables"
	@echo "  make format      - Run pre-commit hooks on all files"
	@echo "  make lint        - Run ruff linter only"
	@echo "  make clean       - Clean Python cache files"
	@echo "  make docs        - Serve documentation locally"
	@echo "  make docs-serve  - Serve documentation locally"
	@echo "  make docs-build  - Build documentation site"
	@echo "  make help        - Show this help message"
