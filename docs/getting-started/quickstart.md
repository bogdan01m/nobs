# Quick Start

Get started with La Perf in just a few minutes!

---

## Prerequisites

Before running La Perf, ensure you have:

- **[uv](https://docs.astral.sh/uv/)** package manager
- **Python 3.12+** - uv will automatically install it

!!! info "Why uv?"
    La Perf uses `uv` for fast, reliable dependency management. It's significantly faster than pip and handles environment isolation automatically.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bogdanminko/laperf.git
cd laperf
```

### 2. Install dependencies (optional)

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

---

## Run Specific Benchmarks

### Embeddings Only

```bash
uv run python -m src.tasks.text_embeddings.runner
```

**Requirements:**
- 8 GB RAM minimum
- GPU recommended

### LLM Inference

```bash
uv run python -m src.tasks.llms.runner
```

**Requirements:**
- 16+ GB RAM
- LM Studio or Ollama running locally
- GPU highly recommended

### VLM Inference

```bash
uv run python -m src.tasks.vlms.runner
```

**Requirements:**
- 12+ GB RAM
- LM Studio or Ollama running locally
- GPU highly recommended

---

## LLM/VLM Setup

La Perf supports both **LM Studio** and **Ollama** for LLM/VLM benchmarks.

### Option 1: LM Studio (Recommended)

1. **Download** [LM Studio](https://lmstudio.ai/)
2. **Load a model** (e.g., gpt-oss-20b)
3. **Start the local server** (default: `http://localhost:1234`)
4. **Run benchmarks**

!!! tip "macOS Users"
    For best performance on Apple Silicon, use MLX models in LM Studio. They're 10-20% faster than GGUF.

### Option 2: Ollama

1. **Install** [Ollama](https://ollama.com/)
2. **Pull a model**:
   ```bash
   ollama pull gpt-oss:20b
   ```
3. **Run benchmarks** (Ollama starts automatically)

---

## Understanding Results

After running benchmarks, you'll find:

- **JSON results** in `results/report_{device}.json`
- **Plots** in `results/plots/`
- **Summary tables** in the terminal

### Generate Markdown Tables

```bash
make
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

### GPU not detected

=== "NVIDIA"
    Ensure CUDA drivers are installed:
    ```bash
    nvidia-smi
    ```

=== "Apple Silicon"
    MPS should be available by default on macOS 12.3+

=== "AMD"
    Ensure ROCm is installed and configured

### Out of memory

- **Reduce batch size** in config files
- **Close** other GPU-intensive applications
- **Use CPU** mode for testing (slower but works)

### LM Studio connection failed

- Ensure LM Studio server is **running**
- Check it's on **localhost:1234**
- Verify the model is **loaded**

---

## Get Help

Need help? Check out:

- [GitHub Issues](https://github.com/bogdanminko/laperf/issues)
- [Contributing Guide](../contributing.md)
- [Metrics Guide](../metrics.md)
