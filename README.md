<div align="center">

# La Perf
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![MPS](https://img.shields.io/badge/MPS-Optimized-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![MLX](https://img.shields.io/badge/MLX-Accelerated-FF6B35?style=flat&logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![AI Performance](https://img.shields.io/badge/AI-Performance-FF6B6B?style=flat&logo=tensorflow&logoColor=white)](https://github.com/bogdanminko/nobs)
[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-2ECC71?style=flat&logo=github&logoColor=white)](https://bogdanminko.github.io/laperf/)

### La Perf — a local AI performance benchmark
for comparing AI performance across different devices.

</div>

---
The goal of this project is to create an all-in-one source of information you need **before buying your next laptop or PC for local AI tasks**.

It’s designed for **AI/ML engineers** who prefer to run workloads locally — and for **AI enthusiasts** who want to understand real-world device performance.

> **See full benchmark results here:**
> [Laperf Results](https://bogdanminko.github.io/laperf/results.html)

## Table of Contents

- [Overview](#overview)
- [Philosophy](#philosophy)
- [Benchmark Results](#benchmark-results)
  - [Power metrics](#⚡-power-metrics)
  - [Embeddings](#embeddings)
  - [LLMs](#llms)
  - [VLMs](#vlms)
- [Quick Start](#-quick-start)
- [Contributing](#-contributing)

---

## Overview
### Tasks
La Perf is a collection of reproducible tests and community-submitted results for :
- #### **Embeddings** — ✅ Ready (sentence-transformers, [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb))
   sts models:
   - [modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)
- #### **LLM inference** — ✅ Ready (LM Studio and Ollama, [Awesome Prompts dataset](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts))
   llm models:
   - **LM Studio**: [gpt-oss-20b](https://lmstudio.ai/models/openai/gpt-oss-20b)
     - *macOS*: `mlx-community/gpt-oss-20b-MXFP4-Q8` (MLX MXFP4-Q8)
     - *Other platforms*: `lmstudio-community/gpt-oss-20b-GGUF` (GGUF)
   - **Ollama**: [gpt-oss-20b](https://ollama.com/library/gpt-oss:20b)


- #### **VLM inference** — ✅ Ready (LM Studio and Ollama, [Hallucination_COCO dataset](https://huggingface.co/datasets/DogNeverSleep/Hallucination_COCO))
   vlm models:
   - **LM Studio**: [Qwen3-VL-8B-Instruct](https://lmstudio.ai/models/qwen/qwen3-vl-8b)
     - *macOS*: `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit` (MLX 4-bit)
     - *Other platforms*: `lmstudio-community/Qwen3-VL-8B-Instruct-GGUF-Q4_K_M` (Q4_K_M)
   - **Ollama**: [qwen3-vl:8b](https://ollama.com/library/qwen3-vl:8b)
      - **all platforms**: `qwen3-vl:8b` (Q4_K_M)
- #### **Diffusion image generation** — 📋 Planned
- #### **Speach to Text** - 📋 Planned (whisper)
- #### **Classic ML** — 📋 Planned (scikit-learn, XGBoost, LightGBM, Catboost)

**Note For mac-users**: If it's possible prefer to use lmstudio with `mlx` backend, which gives 10-20% more performance then `gguf`. If you run ollama (by default benchmarks runs both lmstudio and ollama) then you'll see a difference between `mlx` and `gguf` formats.

The `MLX` backend makes the benchmark harder to maintain, but it provides a more realistic performance view, since it’s easy to convert a `safetensors` model into an `mlx` x-bit model.

### Requirements

La Perf is compatible with **Linux**, **macOS**, and **Windows**.
For embedding tasks, **8 GB of RAM** is usually sufficient.
However for all tasks, it is **recommended to have at least 16 GB**, **18 GB** is better, and **24 GB or more** provides the best performance and reduces swap usage.

It’s designed to run anywhere the **`uv` package manager** is installed.

It’s recommended to use a GPU from **NVIDIA**, **AMD**, **Intel**, or **Apple**, since AI workloads run significantly faster on GPUs.
Make sure to enable **full GPU offload** in tools like **LM Studio** or **Ollama** for optimal performance.

For embedding tasks, La Perf **automatically detects your available device** and runs computations accordingly.

---

## Benchmark Results

> **Last Updated**: 2025-11-14

| Device | Platform | GPU | VRAM | Emb RPS P50 | LLM TPS P50 (lms) | LLM TPS P50 (ollama) | VLM TPS P50 (lms) | VLM TPS P50 (ollama) | GPU Power P50 | CPU Power P50 | Emb Efficiency (RPS/W) | LLM Efficiency (TPS/W) lms | LLM Efficiency (TPS/W) ollama | VLM Efficiency (TPS/W) lms | VLM Efficiency (TPS/W) ollama |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | 🐧 Linux | NVIDIA GeForce RTX 4060 Laptop GPU | 8 GB | 162.2 | 15.4 | 16.0 | 22.4 | 13.6 | 18.3 W | - | 8.88 | 0.84 | 0.88 | 1.23 | 0.74 |
| Mac16,6 | 🍏 macOS | Apple M4 Max (32 cores) | shared with system RAM | 55.8 | 56.5 | 61.0 | 51.5 | 47.8 | 11.7 W | 1.1 W | 4.77 | 4.84 | 5.22 | 4.40 | 4.09 |
| Mac16,6 (on battery) | 🍏 macOS | Apple M4 Max (32 cores) (on battery) | shared with system RAM | 53.9 | 55.3 | 62.2 | 49.0 | 46.5 | 11.3 W | 1.1 W | 4.79 | 4.91 | 5.52 | 4.35 | 4.13 |
| OpenStack Nova 26.0.7-1 A100 40GB | 🐧 Linux | NVIDIA A100-PCIE-40GB | 39 GB | 453.6 | - | 113.5 | - | 108.0 | 218.2 W | - | 2.08 | - | 0.52 | - | 0.50 |
| OpenStack Nova A100 80GB | 🐧 Linux | NVIDIA A100 80GB PCIe | 79 GB | 623.8 | - | 135.5 | - | 121.2 | 230.5 W | - | 2.71 | - | 0.59 | - | 0.53 |
| OpenStack Nova RTX3090 | 🐧 Linux | NVIDIA GeForce RTX 3090 | 24 GB | 349.5 | - | 114.8 | - | 105.3 | 345.6 W | - | 1.01 | - | 0.33 | - | 0.30 |
| OpenStack Nova RTX4090 | 🐧 Linux | NVIDIA GeForce RTX 4090 | 24 GB | 643.6 | - | 148.7 | - | 130.4 | 282.5 W | - | 2.28 | - | 0.53 | - | 0.46 |
| OpenStack Nova Tesla T4 | 🐧 Linux | Tesla T4 | 15 GB | 133.7 | - | 41.5 | - | 32.6 | 68.9 W | - | 1.94 | - | 0.60 | - | 0.47 |

*RPS - Requests Per Second (embeddings throughput)*

*TPS - Tokens Per Second (generation speed)*

*W - Watts (power consumption)*

*Efficiency metrics (RPS/W, TPS/W) are calculated using GPU power consumption*


## ⚡ Quick Start

For a full quickstart and setup instructions, please visit the La Perf documentation: [Quickstart](https://bogdanminko.github.io/laperf/getting-started/quickstart.html).

### 1. Clone the repository

```bash
git clone https://github.com/bogdanminko/laperf.git
cd laperf
```

### 2. (Optional) Configure environment variables

La Perf works out of the box with default settings, but you can customize it for different providers:

```bash
cp .env.example .env
# Edit .env to change URLs, models, dataset sizes, etc.
```

See [`.env.example`](.env.example) for all available options, including how to use custom OpenAI-compatible providers like vLLM, TGI, or LocalAI.

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



---

## Citation

If you use **LaPerf** in your research or reports, please cite it as follows:

> Minko B. (2025). *LaPerf: Local AI Performance Benchmark Suite.*
> GitHub repository. Available at: https://github.com/bogdan01m/laperf
> Licensed under the Apache License, Version 2.0.

**BibTeX:**

```bibtex
@software{laperf,
  author       = {Bogdan Minko},
  title        = {LaPerf: Local AI Performance Benchmark Suite},
  year         = {2025},
  url          = {https://github.com/bogdan01m/laperf},
  license      = {Apache-2.0},
  note         = {GitHub repository}
}
```
