# La Perf

<div align="center" markdown>

[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![MPS](https://img.shields.io/badge/MPS-Optimized-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![MLX](https://img.shields.io/badge/MLX-Accelerated-FF6B35?style=flat&logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![AI Performance](https://img.shields.io/badge/AI-Performance-FF6B6B?style=flat&logo=tensorflow&logoColor=white)](https://github.com/bogdanminko/laperf)
[![Open Source](https://img.shields.io/badge/Open%20Source-Benchmark-2ECC71?style=flat&logo=github&logoColor=white)](https://github.com/bogdanminko/laperf)

### La Perf — a local AI performance benchmark
**Compare AI performance across different devices.**

</div>

---

## What is La Perf?

La Perf is an open-source benchmark suite designed to help you make **informed hardware decisions** for local AI workloads.

Whether you're an **AI/ML engineer** running workloads locally, or an **AI enthusiast** looking to understand real-world device performance, La Perf provides:

- **Reproducible benchmarks** across different hardware (M4 Max, RTX 4060, A100, etc.)
- **Real-world workloads** (embeddings, LLM inference, VLM tasks, power monitoring)
- **Transparent metrics** with detailed methodology documentation
- **Community-driven results** to help you compare before you buy

---

## Why La Perf?

The goal of this project is to create an **all-in-one source of information** you need **before buying your next laptop or PC for local AI tasks**.

!!! info "Philosophy"
    We believe in **honest, reproducible benchmarks** that reflect real-world performance, not synthetic marketing numbers.

---

## Features

### Supported Benchmarks

=== "Embeddings"
    **Text embeddings** via `sentence-transformers`

    - Models: [modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)
    - Dataset: [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) (3000 samples)
    - Metrics: RPS (Rows Per Second), E2E Latency

=== "LLMs"
    **LLM inference** via LM Studio and Ollama

    - Models: [gpt-oss-20b](https://lmstudio.ai/models/openai/gpt-oss-20b)
    - Dataset: [Awesome ChatGPT Prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)
    - Metrics: TPS, TTFT, Token Generation Time, E2E Latency

=== "VLMs"
    **Vision-Language Model inference** via LM Studio and Ollama

    - Models: [Qwen3-VL-8B](https://lmstudio.ai/models/qwen/qwen3-vl-8b)
    - Dataset: [Hallucination_COCO](https://huggingface.co/datasets/DogNeverSleep/Hallucination_COCO)
    - Metrics: TPS, TTFT, Token Generation Time, E2E Latency

### On-device Metrics

=== "Power Metrics"
    **Real-time power and resource monitoring**

    - CPU/GPU usage
    - Memory consumption (RAM, VRAM)
    - GPU power draw
    - Battery drain (laptops)

---

## Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get up and running in minutes

    [:octicons-arrow-right-24: Getting Started](getting-started/quickstart.md)

-   :material-chart-line:{ .lg .middle } __View Results__

    ---

    Compare benchmark results across devices

    [:octicons-arrow-right-24: Results](results.md)

-   :material-book-open-variant:{ .lg .middle } __Metrics__

    ---

    Understand how we measure performance

    [:octicons-arrow-right-24: Metrics Reference](metrics.md)

-   :material-github:{ .lg .middle } __Contribute__

    ---

    Help improve La Perf or submit your results

    [:octicons-arrow-right-24: Contributing Guide](contributing.md)

</div>

---

## Supported Hardware

La Perf automatically detects and optimizes for:

- **NVIDIA GPUs** (CUDA)
- **AMD GPUs** (ROCm)
- **Apple Silicon** (MPS/MLX)
- **Intel GPUs**
- **CPU fallback** (all platforms)

---

## Platform Support

Compatible with **Linux**, **macOS**, and **Windows**.

!!! tip "Recommended Setup"
    - **RAM**: 8 GB for embeddings, 18+ GB for LLM/VLM benchmarks
    - **GPU**: Highly recommended for optimal performance
    - **Tools**: Enable full GPU offload in LM Studio/Ollama

---

## Community

Join the discussion, share your results, and help improve La Perf:

- [GitHub Repository](https://github.com/bogdanminko/laperf)
- [Issue Tracker](https://github.com/bogdanminko/laperf/issues)
- [Contributing Guide](contributing.md)

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
