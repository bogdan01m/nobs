.PHONY: all generate format lint clean help

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
	@echo "✨ Done! Run 'make' to update results in README.md"

# Run pre-commit hooks on all files
format:
	@echo "🔧 Running pre-commit hooks..."
	@uv run pre-commit run --all-files

# Run linting only (ruff)
lint:
	@echo "🔍 Running ruff linter..."
	@uv run ruff check src/ main.py

# Clean Python cache files
clean:
	@echo "🧹 Cleaning cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make          - Generate benchmark results tables (default)"
	@echo "  make generate - Generate benchmark results tables"
	@echo "  make format   - Run pre-commit hooks on all files"
	@echo "  make lint     - Run ruff linter only"
	@echo "  make clean    - Clean Python cache files"
	@echo "  make help     - Show this help message"
