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
   - [thenlper/gte-large](https://huggingface.co/thenlper/gte-large)
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

> **Last Updated**: 2025-11-10


### ⚡ Power Metrics

| Device | CPU Usage (p50/p95) | RAM Used (p50/p95) | GPU Usage (p50/p95) | GPU Temp (p50/p95) | Battery (start/end/Δ) | GPU Power (p50/p95) | CPU Power (p50/p95) |
|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | 24.2% / 25.7% | 10.8GB / 13.2GB | 16.0% / 41.0% | 64.0°C / 66.0°C | 99.0% / 100.0% / -1.0% | 18.3W / 44.8W | N/A |
| Mac16,6 | 4.0% / 12.0% | 22.3GB / 23.9GB | 97.0% / 100.0% | N/A | 85% / 85% / +0.0% | 11.7W / 32.3W | 1.1W / 2.2W |

*p50 = median, p95 = 95th percentile*



### Embeddings

#### Text Embeddings (3000 IMDB samples)

_RPS = Rows Per Second — number of text samples encoded per second._

| Device | Model | RPS (mean ± std) | Time (s) (mean ± std) | Embedding Dim | Batch Size |
|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | nomic-ai/modernbert-embed-base | 162.17 ± 0.61 | 18.50 ± 0.07 | 768 | 32 |
| Mac16,6 | nomic-ai/modernbert-embed-base | 55.81 ± 0.75 | 53.76 ± 0.72 | 768 | 32 |


### LLMs

#### LLM Inference (10 prompts from awesome-chatgpt-prompts)


**LM STUDIO**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | openai/gpt-oss-20b | 15.36 ± 0.10 | 16.81 ± 0.17 | 3.12 ± 0.07 | 6.36 ± 0.07 | 0.93 ± 0.13 | 65.72 ± 0.98 | 6.15 ± 0.15 | 69.19 ± 0.87 | 1728 | 4004 |
| Mac16,6 | openai/gpt-oss-20b | 56.53 ± 1.65 | 77.21 ± 1.99 | 0.92 ± 0.02 | 1.23 ± 0.03 | 0.24 ± 0.00 | 17.09 ± 0.57 | 1.28 ± 0.04 | 18.28 ± 0.60 | 1728 | 3906 |

**OLLAMA**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | gpt-oss:20b | 16.03 ± 0.04 | 16.43 ± 0.02 | 35.68 ± 13.48 | 158.11 ± 0.38 | 4.53 ± 0.05 | 74.99 ± 1.27 | 59.90 ± 0.02 | 199.34 ± 0.39 | 1728 | 13563 |
| Mac16,6 | gpt-oss:20b | 61.03 ± 4.29 | 63.50 ± 6.07 | 4.18 ± 0.31 | 56.83 ± 0.82 | 0.46 ± 0.04 | 25.17 ± 0.33 | 4.64 ± 0.35 | 79.54 ± 0.91 | 1728 | 12939 |


### VLMs

#### VLM Inference (10 questions from Hallucination_COCO)


**LM STUDIO**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | qwen/qwen3-vl-8b | 22.43 ± 0.08 | 23.20 ± 0.55 | 0.75 ± 0.05 | 0.84 ± 0.05 | 22.24 ± 0.03 | 31.98 ± 0.10 | 23.03 ± 0.06 | 32.65 ± 0.10 | 290 | 5128 |
| Mac16,6 | qwen/qwen3-vl-8b | 51.47 ± 1.30 | 53.62 ± 1.82 | 1.58 ± 0.01 | 1.77 ± 0.07 | 9.62 ± 0.48 | 13.42 ± 0.37 | 11.24 ± 0.48 | 15.06 ± 0.30 | 310 | 5966 |

**OLLAMA**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | qwen3-vl:8b | 13.60 ± 0.08 | 14.12 ± 0.06 | 54.82 ± 5.26 | 72.83 ± 0.45 | 58.42 ± 1.03 | 83.23 ± 0.56 | 109.44 ± 6.02 | 152.33 ± 1.20 | 1814 | 14636 |
| Mac16,6 | qwen3-vl:8b | 47.78 ± 4.93 | 49.61 ± 6.79 | 15.29 ± 1.24 | 27.64 ± 0.60 | 16.28 ± 0.91 | 19.59 ± 1.52 | 33.09 ± 3.44 | 44.33 ± 0.41 | 1814 | 15490 |


---
_All metrics are shown as mean ± standard deviation across 3 runs. 
## ⚡ Quick Start

For a full quickstart and setup instructions, please visit the La Perf documentation: [Quickstart](https://bogdanminko.github.io/laperf/getting-started/quickstart.html).

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
