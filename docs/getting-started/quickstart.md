# Quick Start

Get started with La Perf in just a few minutes!

---

## Prerequisites

Before running La Perf, ensure you have:

- **[uv](https://docs.astral.sh/uv/)** package manager
- **Python 3.12+** - uv will automatically install it
- **Ollama** - for LLM, VLM inference (Optional)
- **LM Studio** - for LLM, VLM inference (Optional)

!!! info "Why uv?"
    La Perf uses `uv` for fast, reliable dependency management. It's significantly faster than pip and handles environment isolation automatically.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bogdanminko/laperf.git
cd laperf
```

### 2. (Optional) Configure environment variables

La Perf works out of the box with default settings, but you can customize it:

```bash
cp .env.example .env
# Edit .env to customize settings
```

Common customizations:

- **Change provider URLs** - Use different OpenAI-compatible providers (vLLM, TGI, LocalAI)
- **Adjust dataset sizes** - Change `LLM_DATA_SIZE`, `VLM_DATA_SIZE`, `EMBEDDING_DATA_SIZE`
- **Select backends** - Use `LM_STUDIO`, `OLLAMA`, or `BOTH` for benchmarking
- **Customize models** - Set different model names for your provider

!!! example "Using a custom provider"
    To use vLLM or another OpenAI-compatible provider:

    ```bash
    # In your .env file:
    LLM_BACKEND=LM_STUDIO
    LMS_LLM_BASE_URL=http://localhost:8000/v1
    LMS_LLM_MODEL_NAME=Qwen/Qwen3-30B-Instruct
    LLM_API_KEY=your-api-key-if-needed
    ```

### 3. Install dependencies (optional)

```bash
uv sync
```

This will:

- Create a virtual environment
- Install all required dependencies
- Set up the project for immediate use

---

## Running Your First Benchmark

### Run all benchmarks
**Using make**
```bash
make bench
```

**Using uv**
```bash
uv run python main.py
```

This will:

1. **Auto-detect** your hardware (CUDA / MPS / CPU)
2. **Run** all available benchmarks
   (all are pre-selected — you can toggle individual ones in the TUI using `Space`)
3. **Save** the results to `results/report_{your_device}.json`

!!! success "Hardware Detection"
    La Perf automatically detects your GPU and optimizes accordingly. No manual configuration needed!

## Understanding Results

After running benchmarks, you'll find:

- **JSON results** in `results/report_{device}.json`
- **Plots** in `results/plots/`
- **Summary tables** in the terminal

### Generate Markdown Tables
Run
```bash
make
```
or

```bash
make generate
```

This processes JSON results and generates markdown tables for the README.

---

## Next Steps

- [View Results](../results.md) - Compare your results with other devices
- [Understand Metrics](../metrics.md) - Learn how we measure performance
- [View Results](../results.md) - See benchmark results across devices
- [Contribute](../contributing.md) - Submit your results or add new benchmarks

---

## Troubleshooting

### Out of memory

If you encounter out-of-memory errors, create a `.env` file and adjust these settings:

```bash
cp .env.example .env
```

Then edit `.env` to reduce resource usage:

- **Reduce batch size**: `EMBEDDING_BATCH_SIZE=16` (default: 32)
- **Reduce dataset size**: `EMBEDDING_DATA_SIZE=1000` (default: 3000)
- **Reduce LLM/VLM samples**: `LLM_DATA_SIZE=5` or `VLM_DATA_SIZE=5` (default: 10)
- **Close** other GPU-intensive applications
- **Use CPU** mode for testing (slower but works)

---

## Get Help

Need help? Check out:

- [GitHub Issues](https://github.com/bogdanminko/laperf/issues)
- [Contributing Guide](../contributing.md)
- [Metrics Guide](../metrics.md)
