## Tasks
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



### Power Metrics

| Device | CPU Usage (p50/p95) | RAM Used GB (p50/p95) | VRAM Used GB (p50/p95) | GPU Usage (p50/p95) | GPU Temp (p50/p95) | Battery (start/end/Δ) | Duration | GPU Power (p50/p95) | CPU Power (p50/p95) |
|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | 24.2% / 25.7% | 10.8GB / 13.2GB | 7.0GB / 7.2GB | 16.0% / 41.0% | 64.0°C / 66.0°C | 99.0% / 100.0% / -1.0% | 2h 8m | 18.3W / 44.8W | N/A |
| Mac16,6 | 4.0% / 12.0% | 22.3GB / 23.9GB | 10.7GB / 14.5GB | 97.0% / 100.0% | N/A | 85% / 85% / +0.0% | 42m 56s | 11.7W / 32.3W | 1.1W / 2.2W |
| Mac16,6 (on battery) | 4.1% / 10.8% | 21.4GB / 24.5GB | 11.5GB / 14.6GB | 96.0% / 100.0% | N/A | 85% / 29% / +56.0% | 44m 32s | 11.3W / 30.5W | 1.1W / 2.3W |
| OpenStack Nova 26.0.7-1 A100 40GB | 23.4% / 32.0% | 5.4GB / 6.2GB | 12.0GB / 13.6GB | 77.0% / 85.0% | 59.0°C / 66.0°C | N/A | 16m 44s | 218.2W / 256.2W | N/A |
| OpenStack Nova A100 80GB | 8.7% / 11.3% | 5.6GB / 6.3GB | 12.0GB / 13.6GB | 86.0% / 90.0% | 52.0°C / 55.0°C | N/A | 14m 38s | 230.5W / 274.4W | N/A |
| OpenStack Nova RTX3090 | 17.9% / 22.2% | 4.9GB / 5.6GB | 11.7GB / 13.2GB | 82.0% / 86.0% | 62.0°C / 62.0°C | N/A | 15m 10s | 345.6W / 348.7W | N/A |
| OpenStack Nova RTX4090 | 17.5% / 20.9% | 4.8GB / 5.6GB | 11.8GB / 13.5GB | 84.0% / 89.0% | 57.0°C / 60.0°C | N/A | 13m 12s | 282.5W / 331.8W | N/A |
| OpenStack Nova Tesla T4 | 14.7% / 16.7% | 3.8GB / 4.4GB | 10.7GB / 12.4GB | 95.0% / 96.0% | 49.0°C / 49.0°C | N/A | 44m 32s | 68.9W / 71.5W | N/A |

!!! Note

    For devices with unified memory (e.g. Apple Silicon), **VRAM usage** represents the portion of shared RAM allocated to the GPU — it does not indicate a separate dedicated memory pool as on discrete GPUs.


    **Duration** shows the total monitoring time during benchmark execution.



### Embeddings

#### Text Embeddings (3000 IMDB samples)

_RPS = Rows Per Second — number of text samples encoded per second._

| Device | Model | RPS (mean ± std) | Time (s) (mean ± std) | Embedding Dim | Batch Size |
|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | nomic-ai/modernbert-embed-base | 162.17 ± 0.61 | 18.50 ± 0.07 | 768 | 32 |
| Mac16,6 | nomic-ai/modernbert-embed-base | 55.81 ± 0.75 | 53.76 ± 0.72 | 768 | 32 |
| Mac16,6 (on battery) | nomic-ai/modernbert-embed-base | 53.93 ± 3.78 | 55.82 ± 4.07 | 768 | 32 |
| OpenStack Nova 26.0.7-1 A100 40GB | nomic-ai/modernbert-embed-base | 453.58 ± 2.09 | 6.61 ± 0.03 | 768 | 32 |
| OpenStack Nova A100 80GB | nomic-ai/modernbert-embed-base | 623.81 ± 1.30 | 4.81 ± 0.01 | 768 | 32 |
| OpenStack Nova RTX3090 | nomic-ai/modernbert-embed-base | 349.50 ± 0.97 | 8.58 ± 0.02 | 768 | 32 |
| OpenStack Nova RTX4090 | nomic-ai/modernbert-embed-base | 643.55 ± 2.16 | 4.66 ± 0.02 | 768 | 32 |
| OpenStack Nova Tesla T4 | nomic-ai/modernbert-embed-base | 133.71 ± 1.22 | 22.44 ± 0.20 | 768 | 32 |

![Embeddings Performance Profile](plots/embeddings_performance.png)

*Throughput comparison for different embedding models across hardware. Higher values indicate better performance.*


![Embeddings Efficiency](plots/efficiency_embeddings.png)

*Embeddings efficiency (RPS/W) across devices. Higher values indicate better performance per watt.*


### LLMs

#### LLM Inference (10 prompts from awesome-chatgpt-prompts)


**LM STUDIO**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | openai/gpt-oss-20b | 15.36 ± 0.10 | 16.81 ± 0.17 | 3.12 ± 0.07 | 6.36 ± 0.07 | 0.93 ± 0.13 | 65.72 ± 0.98 | 6.15 ± 0.15 | 69.19 ± 0.87 | 1728 | 4004 |
| Mac16,6 | openai/gpt-oss-20b | 56.53 ± 1.65 | 77.21 ± 1.99 | 0.92 ± 0.02 | 1.23 ± 0.03 | 0.24 ± 0.00 | 17.09 ± 0.57 | 1.28 ± 0.04 | 18.28 ± 0.60 | 1728 | 3906 |
| Mac16,6 (on battery) | openai/gpt-oss-20b | 55.34 ± 0.91 | 78.55 ± 0.97 | 0.90 ± 0.01 | 1.18 ± 0.02 | 0.24 ± 0.00 | 17.56 ± 0.19 | 1.22 ± 0.02 | 18.67 ± 0.20 | 1728 | 3982 |

**OLLAMA**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | gpt-oss:20b | 16.03 ± 0.04 | 16.43 ± 0.02 | 35.68 ± 13.48 | 158.11 ± 0.38 | 4.53 ± 0.05 | 74.99 ± 1.27 | 59.90 ± 0.02 | 199.34 ± 0.39 | 1728 | 13563 |
| Mac16,6 | gpt-oss:20b | 61.03 ± 4.29 | 63.50 ± 6.07 | 4.18 ± 0.31 | 56.83 ± 0.82 | 0.46 ± 0.04 | 25.17 ± 0.33 | 4.64 ± 0.35 | 79.54 ± 0.91 | 1728 | 12939 |
| Mac16,6 (on battery) | gpt-oss:20b | 62.19 ± 3.33 | 66.18 ± 5.45 | 10.95 ± 1.08 | 48.79 ± 1.11 | 1.74 ± 0.11 | 29.83 ± 2.93 | 22.61 ± 0.77 | 55.19 ± 1.84 | 1728 | 14932 |
| OpenStack Nova 26.0.7-1 A100 40GB | gpt-oss:20b | 113.51 ± 1.74 | 119.83 ± 0.78 | 1.92 ± 0.01 | 31.23 ± 15.21 | 0.56 ± 0.00 | 11.08 ± 0.85 | 5.24 ± 0.09 | 35.87 ± 15.85 | 1728 | 16927 |
| OpenStack Nova A100 80GB | gpt-oss:20b | 135.49 ± 0.36 | 141.08 ± 0.38 | 1.58 ± 0.01 | 26.31 ± 12.50 | 0.48 ± 0.01 | 9.41 ± 0.67 | 4.40 ± 0.01 | 30.23 ± 12.96 | 1728 | 16927 |
| OpenStack Nova RTX3090 | gpt-oss:20b | 114.83 ± 0.13 | 119.78 ± 0.46 | 3.24 ± 0.03 | 9.86 ± 0.04 | 0.24 ± 0.00 | 10.64 ± 0.07 | 5.30 ± 0.01 | 19.43 ± 0.09 | 1728 | 8855 |
| OpenStack Nova RTX4090 | gpt-oss:20b | 148.69 ± 0.54 | 153.80 ± 0.24 | 2.69 ± 0.02 | 13.65 ± 0.04 | 0.26 ± 0.00 | 8.68 ± 0.03 | 6.25 ± 0.12 | 18.51 ± 0.08 | 1728 | 11939 |
| OpenStack Nova Tesla T4 | gpt-oss:20b | 41.49 ± 0.23 | 42.17 ± 0.07 | 13.07 ± 2.85 | 52.33 ± 15.42 | 0.85 ± 0.11 | 35.54 ± 4.21 | 15.88 ± 1.51 | 84.51 ± 10.32 | 1728 | 12812 |

![LLM E2E Latency Performance](plots/llm_latency.png)

*End-to-End Latency P50 - Lower is better. Measures full request-to-response time.*


![LLM Throughput Performance](plots/llm_tps.png)

*Token Generation per second (TPS) - Higher is better. Measures token generation speed.*


![LLM Efficiency](plots/efficiency_llm.png)

*LLM inference efficiency (TPS/W) by backend. Higher values indicate better performance per watt.*


### VLMs

#### VLM Inference (10 questions from Hallucination_COCO)


**LM STUDIO**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | qwen/qwen3-vl-8b | 22.43 ± 0.08 | 23.20 ± 0.55 | 0.75 ± 0.05 | 0.84 ± 0.05 | 22.24 ± 0.03 | 31.98 ± 0.10 | 23.03 ± 0.06 | 32.65 ± 0.10 | 290 | 5128 |
| Mac16,6 | qwen/qwen3-vl-8b | 51.47 ± 1.30 | 53.62 ± 1.82 | 1.58 ± 0.01 | 1.77 ± 0.07 | 9.62 ± 0.48 | 13.42 ± 0.37 | 11.24 ± 0.48 | 15.06 ± 0.30 | 310 | 5966 |
| Mac16,6 (on battery) | qwen/qwen3-vl-8b | 48.95 ± 2.10 | 53.44 ± 5.07 | 1.63 ± 0.02 | 1.82 ± 0.07 | 10.66 ± 1.12 | 13.86 ± 1.22 | 12.36 ± 1.18 | 15.52 ± 1.26 | 310 | 5907 |

**OLLAMA**

| Device | Model | TPS P50 | TPS P95 | TTFT P50 (s) | TTFT P95 (s) | TG P50 (s) | TG P95 (s) | Latency P50 (s) | Latency P95 (s) | Input Tokens | Output Tokens |
|------|------|------|------|------|------|------|------|------|------|------|------|
| ASUSTeK COMPUTER ASUS Vivobook Pro N6506MV | qwen3-vl:8b | 13.60 ± 0.08 | 14.12 ± 0.06 | 54.82 ± 5.26 | 72.83 ± 0.45 | 58.42 ± 1.03 | 83.23 ± 0.56 | 109.44 ± 6.02 | 152.33 ± 1.20 | 1814 | 14636 |
| Mac16,6 | qwen3-vl:8b | 47.78 ± 4.93 | 49.61 ± 6.79 | 15.29 ± 1.24 | 27.64 ± 0.60 | 16.28 ± 0.91 | 19.59 ± 1.52 | 33.09 ± 3.44 | 44.33 ± 0.41 | 1814 | 15490 |
| Mac16,6 (on battery) | qwen3-vl:8b | 46.46 ± 0.24 | 51.22 ± 8.10 | 16.29 ± 0.84 | 18.49 ± 0.62 | 17.14 ± 0.16 | 21.68 ± 0.57 | 32.78 ± 1.95 | 38.83 ± 1.10 | 1814 | 15516 |
| OpenStack Nova 26.0.7-1 A100 40GB | qwen3-vl:8b | 108.03 ± 0.17 | 108.57 ± 0.57 | 7.09 ± 0.01 | 11.59 ± 0.70 | 6.97 ± 0.03 | 9.42 ± 0.01 | 14.03 ± 0.02 | 19.40 ± 0.46 | 1814 | 16497 |
| OpenStack Nova A100 80GB | qwen3-vl:8b | 121.16 ± 0.23 | 121.55 ± 0.26 | 6.26 ± 0.01 | 10.34 ± 0.70 | 6.25 ± 0.03 | 8.43 ± 0.02 | 12.53 ± 0.03 | 17.34 ± 0.52 | 1814 | 16497 |
| OpenStack Nova RTX3090 | qwen3-vl:8b | 105.30 ± 0.42 | 105.65 ± 0.28 | 7.44 ± 0.09 | 11.97 ± 0.06 | 7.59 ± 0.28 | 9.05 ± 0.02 | 14.17 ± 0.11 | 19.59 ± 0.06 | 1814 | 15915 |
| OpenStack Nova RTX4090 | qwen3-vl:8b | 130.42 ± 0.25 | 130.94 ± 0.22 | 5.53 ± 0.01 | 8.99 ± 0.01 | 5.85 ± 0.07 | 7.29 ± 0.01 | 10.97 ± 0.21 | 15.24 ± 0.03 | 1814 | 14971 |
| OpenStack Nova Tesla T4 | qwen3-vl:8b | 32.63 ± 0.01 | 32.74 ± 0.06 | 23.38 ± 0.04 | 32.20 ± 0.02 | 23.17 ± 0.10 | 32.79 ± 0.04 | 46.78 ± 0.51 | 63.59 ± 0.05 | 1814 | 15706 |

![VLM E2E Latency Performance](plots/vlm_latency.png)

*End-to-End Latency P50 - Lower is better. Measures full request-to-response time.*


![VLM Throughput Performance](plots/vlm_tps.png)

*Token Generation per second (TPS) - Higher is better. Measures token generation speed.*


![VLM Efficiency](plots/efficiency_vlm.png)

*VLM inference efficiency (TPS/W) by backend. Higher values indicate better performance per watt.*


---
_All metrics are shown as mean ± standard deviation across 3 runs.
