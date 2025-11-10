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

## GPU Setup

### NVIDIA (CUDA)

#### Check CUDA availability

```bash
nvidia-smi
```

#### Install CUDA toolkit

Follow [NVIDIA's guide](https://developer.nvidia.com/cuda-downloads) for your platform.

#### Verify PyTorch CUDA support

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon (MPS)

MPS is available by default on macOS 12.3+ with Apple Silicon.

#### Verify MPS support

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

### AMD (ROCm)

#### Install ROCm

Follow [AMD's guide](https://rocm.docs.amd.com/) for your platform.

#### Verify ROCm support

```bash
rocm-smi
```

---

## LM Studio Setup

For LLM/VLM benchmarks, install LM Studio:

### 1. Download LM Studio

Visit [lmstudio.ai](https://lmstudio.ai/) and download for your platform.

### 2. Load a model

=== "macOS (MLX)"
    Search for: `mlx-community/gpt-oss-20b-MXFP4-Q8`

=== "Windows/Linux (GGUF)"
    Search for: `lmstudio-community/gpt-oss-20b-GGUF`

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

```bash
ollama pull gpt-oss:20b
```

---

## Verifying Your Setup

Run a quick test to ensure everything works:

```bash
uv run python main.py
```

You should see:

- Hardware detection output
- Benchmark progress bars
- Results saved to `results/` directory

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

### Out of disk space

Models and datasets require ~5 GB. Free up space or use a different directory:

```bash
export HF_HOME=/path/to/large/disk
```

---

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first benchmark
- [Requirements](requirements.md) - Detailed hardware requirements
- [Benchmark Results](../results.md) - View benchmark results and metrics
