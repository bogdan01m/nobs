<div align="center">

# La Perf
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![MPS](https://img.shields.io/badge/MPS-Optimized-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
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
  - [Embeddings](#embeddings)
  - [LLMs](#llms)
  - [VLMs](#vlms)
- [Quick Start](#-quick-start)
- [Contributing](#-contributing)

---

## Overview
**laperf** is an open-source benchmark suite
for evaluating *real AI hardware performance* — not synthetic FLOPS or polished demos.

### Tasks
Nobs is a collection of reproducible tests and community-submitted results for :
- #### 🧩 **Embeddings** — ✅ Ready (sentence-transformers, [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb))
   sts models:
   - [thenlper/gte-large](https://huggingface.co/thenlper/gte-large)
   - [modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)
- #### 💬 **LLM inference** — ✅ Ready (LM Studio and Ollama, [Awesome Prompts dataset](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts))
   llm models:
   - [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) quantization: mxfp4
- #### 👁️ **VLM inference** — ✅ Ready (LM Studio and Ollama, [Hallucination_COCO dataset](https://huggingface.co/datasets/DogNeverSleep/Hallucination_COCO))
   vlm models:
   - []
- #### 🎨 **Diffusion image generation** — 📋 Planned
- #### 🗣️ **Speach to Text** - 📋 Planned (whisper)
- #### 🔬 **Classic ML** — 📋 Planned (scikit-learn, XGBoost, LightGBM, Catboost)

**NOTE for mac-users**: If it's possible prefer to use lmstudio with `mlx` backend, which gives 10-20% more performance then `gguf`. If you run ollama (by default benchmarks runs both lmstudio and ollama) then you'll see a difference between `mlx` and `gguf` formats.

---

## Philosophy

> *"We don’t measure synthetic FLOPS. We measure how your GPU cries in real life."*

NoBS was built to understand how different devices — from everyday laptops and PCs to large inference giants — actually perform on real AI tasks.

---

## Benchmark Results

> **Last Updated**: 2025-11-02

### 🏆 Overall Ranking

| Rank | Device | Platform | CPU | RAM | GPU | VRAM | Embeddings (s) | LLM (s) | VLM (s) | Total Time (s) |
|------|--------|----------|-----|-----|-----|------|----------------|---------|---------|----------------|
| 🥇 1 | Mac16,10 | 🍏 macOS | Apple M4 Max (14 cores) | 36 GB | Apple M4 Max (40 cores) | shared with system RAM | 8.56 | 115.88 | 85.06 | **209.50** |
| 🥈 2 | RTX4060Ti-PC | 🐧 Linux | Intel Core i7-13700K | 32 GB | NVIDIA RTX 4060 Ti | 16 GB | 9.99 | 128.56 | 81.74 | **220.29** |
| 🥉 3 | Arc-A770-Lab | 🐧 Linux | Intel Core Ultra 7 165H | 48 GB | Intel Arc A770 | 16 GB | 12.22 | 235.28 | - | **247.50** |
| 4 | Radeon-Workstation | 🪟 Windows | AMD Ryzen 9 7950X | 64 GB | AMD Radeon RX 7900 XTX | 24 GB | 9.64 | 250.99 | 91.75 | **352.38** |
| 5 | Mac14,7 | 🍏 macOS | Apple M2 (8 cores) | 24 GB | Apple M2 (10 cores) | shared with system RAM | 12.13 | 211.87 | 132.94 | **356.94** |


#### Embeddings Performance Visualization

![Embeddings Performance Profile](results/plots/embeddings_performance.png)

*Throughput comparison for different embedding models across hardware. Higher values indicate better performance.*


### 📈 Embeddings

#### Text Embeddings (100 IMDB samples)

| Device | Model | Rows/sec | Time (s) | Embedding Dim | Batch Size |
|--------|-------|----------|----------|---------------|------------|
| Arc-A770-Lab | nomic-ai/nomic-embed-text-v1.5 | 142.27 ± 12.22 | 5.16 ± 0.39 | 768 | 24 |
| Arc-A770-Lab | text-embedding-3-large | 180.92 ± 21.01 | 4.15 ± 0.27 | 3072 | 16 |
| Arc-A770-Lab | text-embedding-3-small | 349.04 ± 17.43 | 2.90 ± 0.25 | 1536 | 32 |
| Mac14,7 | nomic-ai/nomic-embed-text-v1.5 | 151.85 ± 14.73 | 4.77 ± 0.34 | 768 | 24 |
| Mac14,7 | text-embedding-3-large | 236.12 ± 19.00 | 4.60 ± 0.47 | 3072 | 16 |
| Mac14,7 | text-embedding-3-small | 325.82 ± 14.32 | 2.76 ± 0.18 | 1536 | 32 |
| Mac16,10 | nomic-ai/nomic-embed-text-v1.5 | 220.49 ± 15.86 | 4.02 ± 0.38 | 768 | 24 |
| Mac16,10 | text-embedding-3-large | 281.46 ± 31.40 | 2.69 ± 0.26 | 3072 | 16 |
| Mac16,10 | text-embedding-3-small | 425.60 ± 50.04 | 1.85 ± 0.27 | 1536 | 32 |
| RTX4060Ti-PC | nomic-ai/nomic-embed-text-v1.5 | 171.67 ± 7.23 | 5.34 ± 0.48 | 768 | 24 |
| RTX4060Ti-PC | text-embedding-3-large | 269.83 ± 27.46 | 2.84 ± 0.38 | 3072 | 16 |
| RTX4060Ti-PC | text-embedding-3-small | 388.34 ± 44.31 | 1.80 ± 0.20 | 1536 | 32 |
| Radeon-Workstation | nomic-ai/nomic-embed-text-v1.5 | 196.82 ± 18.14 | 4.26 ± 0.31 | 768 | 24 |
| Radeon-Workstation | text-embedding-3-large | 231.25 ± 16.41 | 3.13 ± 0.41 | 3072 | 16 |
| Radeon-Workstation | text-embedding-3-small | 406.83 ± 46.62 | 2.24 ± 0.30 | 1536 | 32 |

![Embeddings Performance Profile](results/plots/embeddings_performance.png)

*Throughput comparison for different embedding models across hardware. Higher values indicate better performance.*


### 🧠 LLMs

#### LLM Inference (3 prompts from awesome-chatgpt-prompts)


**LM STUDIO**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| Mac14,7 | nous-hermes-llama2-13b | 75.03 ± 3.55 | 3.43 ± 1.56 | 19.46 ± 6.16 | 5915 | 12286 |
| Mac16,10 | granite-coder-34b | 160.72 ± 11.77 | 1.62 ± 0.58 | 6.03 ± 2.53 | 9769 | 13303 |
| Radeon-Workstation | mixtral-8x7b-instruct | 119.73 ± 8.88 | 1.49 ± 1.06 | 9.04 ± 3.00 | 6043 | 12851 |

**OLLAMA**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| Arc-A770-Lab | phi3:mini-4k | 68.32 ± 5.63 | 4.54 ± 1.69 | 17.65 ± 6.56 | 8345 | 12702 |
| RTX4060Ti-PC | llama3:13b-instruct | 126.76 ± 6.38 | 2.17 ± 0.73 | 8.85 ± 3.25 | 8081 | 12895 |
| Radeon-Workstation | mistral:7b-instruct | 106.79 ± 7.38 | 3.15 ± 1.04 | 10.54 ± 4.73 | 6566 | 8673 |

![LLM TTFT vs Input Tokens](results/plots/llm_ttft_vs_input_tokens.png)

*Time To First Token across prompt lengths. Lower values mean faster first responses.*


![LLM Generation Time vs Output Tokens](results/plots/llm_tg_vs_output_tokens.png)

*Generation time growth relative to output length. Lower values reflect faster completions.*

![LLM TTFT Performance](results/plots/llm_ttft.png)

*Time To First Token (TTFT) - Lower is better. Measures response latency.*


![LLM Throughput Performance](results/plots/llm_tps.png)

*Token Generation per second (TG) - Higher is better. Measures token generation.*


### 👁️ VLMs

#### VLM Inference (3 questions from Hallucination_COCO)


**LM STUDIO**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| Mac14,7 | llava:7b | 55.16 ± 3.65 | 4.70 ± 1.53 | 17.23 ± 3.51 | 4053 | 5936 |
| Mac16,10 | minicpm-v:8b | 105.73 ± 8.11 | 2.38 ± 0.49 | 6.98 ± 3.00 | 5202 | 6566 |
| Radeon-Workstation | llava:13b | 72.16 ± 4.65 | 3.45 ± 0.80 | 8.70 ± 3.26 | 5317 | 4482 |

**OLLAMA**

| Device | Model | Tokens/sec | TTFT (s) | Latency (s) | Input Tokens | Output Tokens |
|--------|-------|------------|----------|-------------|--------------|---------------|
| RTX4060Ti-PC | minicpm-v:8b | 84.37 ± 5.38 | 3.06 ± 1.15 | 10.32 ± 3.19 | 5198 | 4365 |

![VLM TTFT vs Input Tokens](results/plots/vlm_ttft_vs_input_tokens.png)

*TTFT behaviour for multimodal prompts. Lower values mean faster first visual-token outputs.*


![VLM Generation Time vs Output Tokens](results/plots/vlm_tg_vs_output_tokens.png)

*Generation time vs output token count for multimodal responses. Lower values are faster.*

![VLM TTFT Performance](results/plots/vlm_ttft.png)

*Time To First Token (TTFT) - Lower is better. Measures response latency.*


![VLM Throughput Performance](results/plots/vlm_tps.png)

*Token Generation per second (TG) - Higher is better. Measures token generation.*


---

_All metrics are shown as median ± standard deviation across 3 runs.
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
2. Run all available benchmarks (currently: embeddings)
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
