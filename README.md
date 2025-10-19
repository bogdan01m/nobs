# NoBS benchmark

[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![MPS](https://img.shields.io/badge/MPS-Optimized-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![AI Performance](https://img.shields.io/badge/AI-Performance-FF6B6B?style=flat&logo=tensorflow&logoColor=white)](https://github.com/bogdanminko/nobs)
[![Open Source](https://img.shields.io/badge/Open%20Source-Benchmark-2ECC71?style=flat&logo=github&logoColor=white)](https://github.com/bogdanminko/nobs)
<div align="center">

### 📚 A collection of AI hardware benchmarks

**No magic scores. No FLOPS. No AI TOPS. No-BullShit.**

*Compare M4 Max, RTX 4060, A100 and other hardware on different AI tasks*

</div>

---

## 🚀 Overview
**NoBS (Neural NetwOrks Benchmark Stash)** is an open-source benchmark suite
for evaluating *real AI hardware performance* — not synthetic FLOPS or polished demos.

It's a collection of reproducible tests and community-submitted results for:
- 🧩 **Embeddings** — ✅ Ready (sentence-transformers, IMDB dataset)
- 💬 **LLM inference** — 🚧 In Progress (LM Studio support only)
- 👁️ **VLM inference** — 📋 Planned
- 🎨 **Diffusion image generation** — 📋 Planned

---

## Philosophy

> *"We don’t measure synthetic FLOPS. We measure how your GPU cries in real life."*

NOBS was built by engineers tired of meaningless benchmark charts.
No synthetic kernels, no fake workloads — just **real models, real data, and honest numbers**.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation
```sh
# Clone the repository
git clone https://github.com/bogdanminko/nobs.git
cd nobs

# Install dependencies
uv sync
```

### Running Benchmarks

#### Run all benchmarks
```sh
uv run python main.py
```

This will:
1. Auto-detect your hardware (CUDA/MPS/CPU)
2. Run all available benchmarks (currently: embeddings)
3. Save results to `results/report_{your_device}.json`

#### Run specific benchmarks
```sh
# Embeddings only
uv run python -m src.tasks.text_embeddings.runner

# LLM inference (requires LM Studio running on localhost:1234)
uv run python -m src.tasks.llms.runner
```

### LLM Benchmarks Setup

**Note:** LLM benchmarks currently require [LM Studio](https://lmstudio.ai/) running locally.

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model in LM Studio
3. Start the local server (default: `http://localhost:1234`)
4. Run the LLM benchmark:
   ```sh
   uv run python -m src.tasks.llms.runner
   ```

---

## 🤝 Contributing

We welcome contributions! Whether it's adding new benchmarks, supporting new models, or submitting your hardware results.

### Development Setup

1. **Fork and clone the repository**
   ```sh
   git clone https://github.com/YOUR_USERNAME/nobs.git
   cd nobs
   ```

2. **Install dependencies including dev tools**
   ```sh
   uv sync --group quality --group dev
   ```

3. **Install pre-commit hooks**
   ```sh
   uv run pre-commit install
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

   # Run code quality checks manually (optional - pre-commit will run them automatically)
   uv run pre-commit run --all-files
   ```

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
- ✅ **Ruff** linting and formatting
- ✅ **Mypy** type checking
- ✅ **Bandit** security checks

These are enforced automatically via pre-commit hooks.

### Adding New Benchmarks

See [CLAUDE.md](CLAUDE.md) for detailed instructions on:
- Adding new models to existing benchmarks
- Creating new benchmark categories
- Data loading patterns
- Memory management best practices

---
