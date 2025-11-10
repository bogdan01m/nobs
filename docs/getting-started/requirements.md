# Requirements

Detailed hardware and software requirements for running La Perf benchmarks.

---

## Software Requirements

### Core Requirements

| Software | Version | Required |
|----------|---------|----------|
| **Python** | 3.12+ | ✅ Yes |
| **uv** | Latest | ✅ Yes |
| **Git** | Any | ✅ Yes |

### Benchmark-Specific Requirements

| Benchmark | Additional Software | Required |
|-----------|---------------------|----------|
| **Embeddings** | None | - |
| **LLM Inference** | LM Studio or Ollama | ✅ Yes |
| **VLM Inference** | LM Studio or Ollama | ✅ Yes |

---

## Hardware Requirements

### Embeddings Benchmark

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | Optional | NVIDIA/AMD/Apple |
| **VRAM** | - | 4 GB+ |
| **Disk** | 2 GB | SSD |

!!! tip "Performance Note"
    GPU acceleration provides 3-10x speedup over CPU for embeddings.

### LLM Inference Benchmark

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 18 GB | 32 GB+ |
| **GPU** | Optional | NVIDIA/AMD/Apple |
| **VRAM** | 6 GB | 8 GB+ |
| **Disk** | 10 GB | SSD |

!!! warning "Memory Requirements"
    20B parameter models require at least 18 GB RAM. Smaller models (7B-8B) work with 12 GB.

### VLM Inference Benchmark

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 18 GB | 32 GB+ |
| **GPU** | Optional | NVIDIA/AMD/Apple |
| **VRAM** | 8 GB | 12 GB+ |
| **Disk** | 10 GB | SSD |

!!! info "VLM Requirements"
    Vision-Language Models require more VRAM due to image processing.

---

## GPU Support

### NVIDIA GPUs (CUDA)

**Supported:**

- GeForce RTX 20/30/40 series
- RTX A series (workstation)
- Tesla/A100/H100 (datacenter)

**Requirements:**

- CUDA 11.8+
- NVIDIA drivers 520+

**Recommended:**

- RTX 4060+ (8 GB VRAM)
- RTX A4000+ (workstation)

### AMD GPUs (ROCm)

**Supported:**

- RX 6000/7000 series
- Instinct MI series

**Requirements:**

- ROCm 5.7+
- Compatible GPU from [AMD's list](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)

**Note:** ROCm support is experimental and may have compatibility issues.

### Apple Silicon (MPS/MLX)

**Supported:**

- M1/M2/M3/M4 series
- M1/M2/M3/M4 Pro/Max/Ultra

**Requirements:**

- macOS 12.3+ (for MPS)
- macOS 13.0+ (for MLX)

**Recommended:**

- M4 Max (48 GB unified memory)
- M3 Max (36 GB unified memory)

!!! success "Apple Silicon Performance"
    MLX models on Apple Silicon often outperform GGUF alternatives by 10-20%.

### Intel GPUs

**Supported:**

- Intel Arc A series
- Intel Iris Xe (limited)

**Requirements:**

- Intel Extension for PyTorch
- Latest Intel GPU drivers

**Note:** Performance may vary. CPU fallback is used if GPU support fails.

---

## Disk Space Requirements

### By Benchmark Type

| Benchmark | Models | Datasets | Total |
|-----------|--------|----------|-------|
| **Embeddings** | ~1 GB | ~500 MB | ~1.5 GB |
| **LLM Inference** | ~12 GB | ~10 MB | ~12 GB |
| **VLM Inference** | ~5 GB | ~50 MB | ~5 GB |
| **All Benchmarks** | ~18 GB | ~550 MB | ~19 GB |

### Disk Performance

- **SSD recommended** for faster model/dataset loading
- **HDD acceptable** but expect slower startup times
- **NVMe SSD** ideal for multiple benchmarks

---

## Network Requirements

### Download Sizes

| Component | Size | Required For |
|-----------|------|--------------|
| **Python dependencies** | ~2 GB | All benchmarks |
| **HuggingFace datasets** | ~500 MB | Embeddings |
| **Embedding models** | ~1 GB | Embeddings |
| **LLM models** | ~12 GB | LLM/VLM inference |

!!! tip "Offline Usage"
    After initial setup, benchmarks can run offline. Models and datasets are cached locally.

### Bandwidth Recommendations

- **Minimum**: 10 Mbps (for patient users)
- **Recommended**: 50 Mbps+
- **Initial setup**: 1-2 hours on slow connections

---

## Operating System Support

### Linux

**Supported:**

- Ubuntu 20.04+
- Debian 11+
- Fedora 36+
- Arch Linux
- CentOS/RHEL 8+

**Best for:**

- NVIDIA/AMD GPU users
- Server deployments
- Advanced configurations

### macOS

**Supported:**

- macOS 12.3+ (Intel)
- macOS 12.3+ (Apple Silicon)

**Best for:**

- Apple Silicon users
- MLX acceleration
- Local development

### Windows

**Supported:**

- Windows 10 (21H2+)
- Windows 11

**Best for:**

- NVIDIA GPU users
- Gaming PCs
- Workstations

!!! warning "Windows Limitations"
    Some features may require WSL2. Native Windows support is improving.

---

## Performance Expectations

### Embeddings (3000 samples)

| Hardware | Time | RPS |
|----------|------|-----|
| **RTX 4060** | ~20s | ~150 |
| **M4 Max** | ~55s | ~55 |
| **CPU (Ryzen 9)** | ~5min | ~10 |

### LLM Inference (10 prompts)

| Hardware | Time | TPS |
|----------|------|-----|
| **RTX 4060** | ~1min | ~15 |
| **M4 Max** | ~30s | ~55 |
| **CPU (Ryzen 9)** | ~10min | ~2 |

!!! info "Performance Varies"
    Actual performance depends on model size, precision, and system configuration.

---

## Next Steps

- [Installation Guide](installation.md) - Install La Perf
- [Quick Start](quickstart.md) - Run your first benchmark
- [Benchmark Results](../results.md) - View benchmark results and metrics
