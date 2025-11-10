## 🤝 Contributing

We welcome contributions! Whether it's adding new benchmarks, supporting new models, or submitting your hardware results.

### Development Setup

1. **Fork and clone the repository**
   ```sh
   git clone https://github.com/YOUR_USERNAME/laperf.git
   cd laperf
   ```

2. **Install dependencies including dev tools**
   ```sh
   uv sync --group quality --group dev
   ```

3. **Install pre-commit hooks**
   ```sh
   pre-commit install
   ```

   This sets up automatic code quality checks that run before each commit:

   - **ruff** — Fast Python linter and formatter
   - **mypy** — Static type checking
   - **bandit** — Security vulnerability scanner
   - Standard checks (trailing whitespace, YAML syntax, etc.)

### Making Changes

1. **Create a new branch**
   ```sh
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the existing patterns
   - Add type hints where applicable
   - Update documentation if needed

3. **Test your changes**
   ```sh
   # Run benchmarks to ensure they work
   uv run python main.py

   # Update benchmark results tables (if you modified results)
   make

   # Run code quality checks manually (optional - pre-commit will run them automatically)
   make format
   ```

   **Available Makefile commands:**

   - `make` — Generate benchmark results tables (default)
   - `make generate` — Generate benchmark results tables
   - `make format` — Run pre-commit hooks on all files
   - `make lint` — Run ruff linter only
   - `make clean` — Clean Python cache files
   - `make help` — Show all available commands

4. **Commit your changes**
   ```sh
   git add .
   git commit -m "feat: your descriptive commit message"
   ```

   Pre-commit hooks will automatically:
   - Format your code
   - Check for type errors
   - Scan for security issues
   - Fix common issues (trailing whitespace, etc.)

   If any check fails, fix the issues and commit again.

5. **Push and create a Pull Request**
   ```sh
   git push origin feature/your-feature-name
   ```

### Code Quality Standards

All contributions must pass:

- **Ruff** linting and formatting
- **Mypy** type checking
- **Bandit** security checks

These are enforced automatically via pre-commit hooks.

### Adding New Benchmarks

See [CLAUDE.md](https://github.com/bogdanminko/laperf/blob/main/CLAUDE.md) for detailed instructions on:
- Adding new models to existing benchmarks
- Creating new benchmark categories
- Data loading patterns
- Memory management best practices

Tip: Add CLAUDE.md when working with your AI coding assistant — it helps provide full project context.

---
