<div align="center">

# La Perf
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![MPS](https://img.shields.io/badge/MPS-Optimized-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![MLX](https://img.shields.io/badge/MLX-Accelerated-FF6B35?style=flat&logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![AI Performance](https://img.shields.io/badge/AI-Performance-FF6B6B?style=flat&logo=tensorflow&logoColor=white)](https://github.com/bogdanminko/nobs)
[![Open Source](https://img.shields.io/badge/Open%20Source-Benchmark-2ECC71?style=flat&logo=github&logoColor=white)](https://github.com/bogdanminko/nobs)
### La Perf — a local AI performance benchmark
for comparing AI performance across different devices.

</div>

---
The goal of this project is to create an all-in-one source of information you need **before buying your next laptop or PC for local AI tasks**.

It’s designed for **AI/ML engineers** who prefer to run workloads locally — and for **AI enthusiasts** who want to understand real-world device performance.

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
- #### 🧩 **Embeddings** — ✅ Ready (sentence-transformers, [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb))
   sts models:
   - [thenlper/gte-large](https://huggingface.co/thenlper/gte-large)
   - [modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)
- #### 💬 **LLM inference** — ✅ Ready (LM Studio and Ollama, [Awesome Prompts dataset](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts))
   llm models:
   - **LM Studio**: [gpt-oss-20b](https://lmstudio.ai/models/openai/gpt-oss-20b)
     - *macOS*: `mlx-community/gpt-oss-20b-MXFP4-Q8` (MLX MXFP4-Q8)
     - *Other platforms*: `lmstudio-community/gpt-oss-20b-GGUF` (GGUF)
   - **Ollama**: [gpt-oss-20b](https://ollama.com/library/gpt-oss:20b)
- #### 👁️ **VLM inference** — ✅ Ready (LM Studio and Ollama, [Hallucination_COCO dataset](https://huggingface.co/datasets/DogNeverSleep/Hallucination_COCO))
   vlm models:
   - **LM Studio**: [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
     - *macOS*: `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit` (MLX 8-bit)
     - *Other platforms*: `lmstudio-community/Qwen3-VL-8B-Instruct-GGUF-Q8_0` (GGUF Q8_0)
   - **Ollama**: `qwen3-vl:8b`
- #### 🎨 **Diffusion image generation** — 📋 Planned
- #### 🗣️ **Speach to Text** - 📋 Planned (whisper)
- #### 🔬 **Classic ML** — 📋 Planned (scikit-learn, XGBoost, LightGBM, Catboost)

**Note For mac-users**: If it's possible prefer to use lmstudio with `mlx` backend, which gives 10-20% more performance then `gguf`. If you run ollama (by default benchmarks runs both lmstudio and ollama) then you'll see a difference between `mlx` and `gguf` formats.

The `MLX` backend makes the benchmark harder to maintain, but it provides a more realistic performance view, since it’s easy to convert a `safetensors` model into an `mlx` x-bit model.

### Requirements

Laperf is compatible with **Linux**, **macOS**, and **Windows**.
For embedding tasks, **8 GB of RAM** is usually enough — sometimes even **4 GB** will work.
It’s designed to run anywhere the **`uv` package manager** is installed.

For LLM or VLM benchmarks, make sure you have **at least 16 GB of RAM** available.

Please note that this project is still in its early stages — some features like **power metrics** and **GPU power tracking** may not yet work on all devices.

It’s recommended to use a GPU from **NVIDIA**, **AMD**, **Intel**, or **Apple**, since AI workloads run significantly faster on GPUs.
Make sure to enable **full GPU offload** in tools like **LM Studio** or **Ollama** for optimal performance.

For embedding tasks, Laperf **automatically detects your available device** and runs computations accordingly.

---

## Philosophy

> *"We don’t measure synthetic FLOPS. We measure how your GPU cries in real life."*

NoBS was built to understand how different devices — from everyday laptops and PCs to large inference giants — actually perform on real AI tasks.

---

## Benchmark Results

> **Last Updated**: 2025-11-03

### 🏆 Overall Ranking

| Rank | Device | Platform | CPU | RAM | GPU | VRAM | Embeddings, sts (s) | LLM, lms (s) | LLM, ollama (s) | VLM, lms (s) | VLM, ollama (s) | Total Time (s) |
|------|--------|----------|-----|-----|-----|------|----------------|-----------------|--------------|-----------------|--------------|----------------|
| 🥇 1 | Mac16,10 | 🍏 macOS | Apple M4 Max (14 cores) | 36 GB | Apple M4 Max (40 cores) | shared with system RAM | 8.56 | 115.88 | - | 85.06 | - | **209.50** |
| 🥈 2 | RTX4060Ti-PC | 🐧 Linux | Intel Core i7-13700K | 32 GB | NVIDIA RTX 4060 Ti | 16 GB | 10.50 | - | 127.28 | - | 84.87 | **222.65** |

*sts - sentence transformers*

*lms - lm stuido*

*ollama - ollama*




### ⚡ Power Metrics

| Device | CPU Usage (p50/p95) | RAM Used (p50/p95) | GPU Usage (p50/p95) | GPU Temp (p50/p95) | Battery Drain (p50/p95) | GPU Power (p50/p95) | CPU Power (p50/p95) |
|--------|---------------------|--------------------|--------------------|--------------------|-----------------------|--------------------|--------------------|
| Mac16,10 | 36.3% / 61.2% | 25.1GB / 25.7GB | 60.0% / 71.6% | 51.3°C / 58.5°C | 29.2W / 30.4W | 23.1W / 48.2W | 9.7W / 17.8W |
| RTX4060Ti-PC | 51.4% / 77.5% | 21.8GB / 24.3GB | 87.2% / 94.0% | 69.7°C / 78.6°C | N/A | 127.7W / 204.8W | N/A |

*p50 = median, p95 = 95th percentile*



### Embeddings

#### Text Embeddings (100 IMDB samples)

| Device | Model | Rows/sec | Time (s) | Embedding Dim | Batch Size |
|--------|-------|----------|----------|---------------|------------|
| Mac16,10 | nomic-ai/nomic-embed-text-v1.5 | 220.49 ± 15.86 | 4.02 ± 0.38 | 768 | 24 |
| Mac16,10 | text-embedding-3-large | 281.46 ± 31.40 | 2.69 ± 0.26 | 3072 | 16 |
| Mac16,10 | text-embedding-3-small | 425.60 ± 50.04 | 1.85 ± 0.27 | 1536 | 32 |
| RTX4060Ti-PC | nomic-ai/nomic-embed-text-v1.5 | 197.21 ± 16.52 | 4.59 ± 0.41 | 768 | 24 |
| RTX4060Ti-PC | text-embedding-3-large | 272.31 ± 31.78 | 3.86 ± 0.50 | 3072 | 16 |
| RTX4060Ti-PC | text-embedding-3-small | 434.32 ± 32.45 | 2.06 ± 0.29 | 1536 | 32 |

![Embeddings Performance Profile](results/plots/embeddings_performance.png)

*Throughput comparison for different embedding models across hardware. Higher values indicate better performance.*


### LLMs

#### LLM Inference (3 prompts from awesome-chatgpt-prompts)


**LM STUDIO**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| Mac16,10 | granite-coder-34b | 160.72 ± 11.77 | 1.62 ± 0.58 | 6.03 ± 2.53 | 9769 | 13303 |

**OLLAMA**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| RTX4060Ti-PC | llama3:13b-instruct | 127.45 ± 9.33 | 1.98 ± 0.74 | 7.95 ± 3.16 | 6551 | 13000 |

![LLM TTFT vs Input Tokens](results/plots/llm_ttft_vs_input_tokens.png)

*Time To First Token across prompt lengths. Lower values mean faster first responses.*


![LLM Generation Time vs Output Tokens](results/plots/llm_tg_vs_output_tokens.png)

*Generation time growth relative to output length. Lower values reflect faster completions.*

![LLM TTFT Performance](results/plots/llm_ttft.png)

*Time To First Token (TTFT) - Lower is better. Measures response latency.*


![LLM Throughput Performance](results/plots/llm_tps.png)

*Token Generation per second (TG) - Higher is better. Measures token generation.*


### VLMs

#### VLM Inference (3 questions from Hallucination_COCO)


**LM STUDIO**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| Mac16,10 | minicpm-v:8b | 105.73 ± 8.11 | 2.38 ± 0.49 | 6.98 ± 3.00 | 5202 | 6566 |

**OLLAMA**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| RTX4060Ti-PC | minicpm-v:8b | 80.82 ± 5.12 | 3.43 ± 1.19 | 10.40 ± 3.62 | 5873 | 4726 |

![VLM TTFT vs Input Tokens](results/plots/vlm_ttft_vs_input_tokens.png)

*TTFT behaviour for multimodal prompts. Lower values mean faster first visual-token outputs.*


![VLM Generation Time vs Output Tokens](results/plots/vlm_tg_vs_output_tokens.png)

*Generation time vs output token count for multimodal responses. Lower values are faster.*

![VLM TTFT Performance](results/plots/vlm_ttft.png)

*Time To First Token (TTFT) - Lower is better. Measures response latency.*


![VLM Throughput Performance](results/plots/vlm_tps.png)

*Token Generation per second (TG) - Higher is better. Measures token generation.*


---

_All metrics are shown as median �� standard deviation across 3 runs.
Lower times are better (faster performance)._

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
2. Run all available benchmarks
3. Save results to `results/report_{your_device}.json`

#### Run specific benchmarks
```sh
# Embeddings only
uv run python -m src.tasks.text_embeddings.runner

# LLM inference (requires LM Studio running on localhost:1234)
uv run python -m src.tasks.llms.runner
```

#### LLM Benchmarks Setup

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

Tip: Add CLAUDE.md when working with your AI coding assistant — it helps provide full project context.

---
