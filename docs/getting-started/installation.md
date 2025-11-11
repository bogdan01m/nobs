# Installation

Detailed installation instructions for La Perf across different platforms.

---

## System Requirements

### Minimum Requirements

- **Python**: 3.12 or higher
- **RAM**: 8 GB (embeddings), 16 GB (LLM), 18 GB (VLM)
- **Disk Space**: ~100 GB free for models and datasets
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **GPU**: NVIDIA (CUDA), AMD (ROCm), or Apple Silicon (MPS)
- **RAM**: 24 GB+ for comfortable multitasking
- **SSD**: Fast storage for dataset loading

---

## Installing uv

La Perf uses `uv` as its package manager.

=== "macOS/Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "With pip"
    ```bash
    pip install uv
    ```

Verify installation:

```bash
uv --version
```
!!! info "Why uv?"
    La Perf uses `uv` for fast, reliable dependency management. It's significantly faster than pip and handles environment isolation automatically.

---

## Installing La Perf

### 1. Clone the repository

```bash
git clone https://github.com/bogdanminko/laperf.git
cd laperf
```

### 2. Install dependencies

#### For benchmarking only

```bash
uv sync
```

#### For development

```bash
uv sync --group quality --group dev
```

This installs additional tools:

- `ruff` - Fast Python linter
- `mypy` - Type checker
- `bandit` - Security scanner
- `pre-commit` - Git hooks

### 3. Verify installation

```bash
uv run python -c "import torch; print(torch.__version__)"
```

---

## LM Studio Setup

For LLM/VLM benchmarks, install LM Studio:

### 1. Download LM Studio

Visit [lmstudio.ai](https://lmstudio.ai/) and download for your platform.

### 2. Load a model
Best way to find it is using LM Studio UI

**Load LLM**

Search for `gpt-oss-20b` in available models

=== "macOS (MLX)"
    `mlx-community/gpt-oss-20b-MXFP4-Q8`

=== "Windows/Linux (GGUF)"
    `lmstudio-community/gpt-oss-20b-GGUF`

**Load VLM**

Search for `Qwen3-VL-8B-Instruct` in available models

=== "macOS (MLX)"
    `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit`

=== "Windows/Linux (GGUF)"
    `lmstudio-community/Qwen3-VL-8B-Instruct-GGUF-Q4_K_M`


### 3. Start the server

1. Click **"Developer"** tab
2. Click **"Start Server"**
3. Verify it's running on `http://localhost:1234`

---

## Ollama Setup
For LLM/VLM benchmarks, install Ollama:

### 1. Install Ollama

=== "macOS"
    ```bash
    brew install ollama
    ```

=== "Linux"
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

=== "Windows"
    Download from [ollama.com](https://ollama.com/)

### 2. Pull a model
**Pull LLM**
```bash
ollama pull gpt-oss:20b
```
**Pull VLM**
```bash
ollama pull qwen3-vl:8b
```

---

## Verifying Your Setup

Run a quick test to ensure everything works:

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
## Troubleshooting

### uv command not found

After installing uv, restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc on macOS
```

### Python version mismatch

Ensure you're using Python 3.12+:

```bash
uv run python --version
```

### CUDA not detected

- Install [NVIDIA drivers](https://www.nvidia.com/download/index.aspx)
- Install [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
- Restart your system

---

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first benchmark
- [Requirements](requirements.md) - Detailed hardware requirements
- [Benchmark Results](../results.md) - View benchmark results and metrics
