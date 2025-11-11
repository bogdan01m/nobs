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

!!! tip "Performance Note"
    GPU acceleration provides 3-10x speedup over CPU for AI Tasks.

### Embeddings Benchmark

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 12 GB+ |
| **GPU** | Optional | NVIDIA/AMD/Apple |
| **Disk** | 2 GB | SSD |

### LLM Inference Benchmark

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM/VRAM** | 12 GB | 16 GB+ |
| **GPU** | Optional | NVIDIA/AMD/Apple |
| **Disk** | 14 GB | SSD |

!!! warning "Memory Requirements"
    20B parameter models require at least 16 GB RAM.

### VLM Inference Benchmark

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM/VRAM** | 12 GB | 16 GB+ |
| **GPU** | Optional | NVIDIA/AMD/Apple |
| **Disk** | 10 GB | SSD |

!!! info "VLM Requirements"
    Vision-Language Models require more VRAM due to image processing.

## Next Steps

- [Installation Guide](installation.md) - Install La Perf
- [Quick Start](quickstart.md) - Run your first benchmark
- [Benchmark Results](../results.md) - View benchmark results and metrics
