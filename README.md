# NOBS — Neural NetwOrks Benchmark Stash

[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![MPS](https://img.shields.io/badge/MPS-Optimized-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![AI Performance](https://img.shields.io/badge/AI-Performance-FF6B6B?style=flat&logo=tensorflow&logoColor=white)](https://github.com/bogdanminko/nobs)
[![Open Source](https://img.shields.io/badge/Open%20Source-Benchmark-2ECC71?style=flat&logo=github&logoColor=white)](https://github.com/bogdanminko/nobs)
<div align="center">

### 🎯 A collection of honest, reproducible AI hardware benchmarks

**Real workloads. Real results. No-Bullshit.**

*Compare M4 Max, RTX 4060, A100 and other hardware on LLM, diffusion, and embedding tasks*

</div>

---

## 🚀 Overview
**NOBS (Neural NetwOrks Benchmark Stash)** is an open-source benchmark suite  
for evaluating *real AI hardware performance* — not synthetic FLOPS or polished demos.

It’s a stash of reproducible tests and community-submitted results for:
- 🧩 Embeddings  
- 💬 LLM inference
- 👁️ VLM / Vision-Language tasks  
- 🎨 Diffusion image generation

---

## Philosophy

> *"We don’t measure synthetic FLOPS. We measure how your GPU cries in real life."*

NOBS was built by engineers tired of meaningless benchmark charts.  
No synthetic kernels, no fake workloads — just **real models, real data, and honest numbers**.

---

## ⚡ Quick Start

### 1. Install
```bash
uv add nobs

## Run Localy
```sh
uv run python -m src.tasks.embeddings.runner
```