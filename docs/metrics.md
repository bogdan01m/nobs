# Metrics

This section describes how La Perf calculates and evaluates metrics across different benchmark tasks.

## Embeddings

### Overview
Embedding benchmarks use the `sentence-transformers` library for encoding operations.

| Metric | Description | Unit |
|--------|-------------|------|
| **E2E Latency** | Total time to encode full dataset | seconds |
| **RPS** | Rows Per Second (throughput) | rows/s |

### Measurement Methodology
The total encoding latency is measured around the `.encode()` call, which internally handles batching.
Each run includes device synchronization before and after encoding to ensure accurate timing.

**Implementation details:**
- Uses `torch.cuda.synchronize()` for NVIDIA GPUs
- Uses `torch.mps.synchronize()` for Apple Silicon GPUs
- Ensures complete device-side execution before measurement

### Cross-Run Statistics
For multiple benchmark runs, simple mean and standard deviation are calculated:
```python
mean(run1, run2, run3) ± std(run1, run2, run3)

# Example with RPS:
# final_mean_rps = mean([run1_rps, run2_rps, run3_rps])
# final_std_rps = std([run1_rps, run2_rps, run3_rps])
# On table you see: final_mean_rps ± final_std_rps
```

**Note:** Embeddings use direct mean/std across runs, not percentile-based statistics.

---


## LLMs & VLMs

### Overview

| Metric | Description | Unit |
|--------|-------------|------|
| **TTFT** | Time To First Token — prompt processing latency | seconds |
| **TG** | Token Generation — time spent generating output | seconds |
| **TPS** | Tokens Per Second — generation throughput | tokens/s |
| **E2E Latency** | End-to-end request latency | seconds |

### Measurement Methodology

#### Streaming & Token Counting

La Perf uses streaming APIs (Ollama, LM Studio via OpenAI SDK) to measure both latency and throughput.

**Critical distinction:** API chunks ≠ tokens

The server sends responses in chunks, but each chunk may contain multiple tokens. Token counts are obtained from server-side usage statistics.

#### Per-Request Measurements

For each prompt in the benchmark:

| Timestamp | Description |
|-----------|-------------|
| `t0_stream` | Request start time |
| `first_token_ts` | First chunk received (≈ first token) |
| `t1_stream` | Response complete |

| Token Count | Source |
|-------------|--------|
| `input_tokens` | From server usage stats |
| `output_tokens` | From server usage stats |

#### Metric Calculations

| Metric | Formula | Notes |
|--------|---------|-------|
| **E2E Latency** | `t1_stream - t0_stream` | Total request time |
| **TTFT** | `first_token_ts - t0_stream` | Prompt processing time |
| **TG** | `t1_stream - first_token_ts` | Generation phase time |
| **TPS** | `output_tokens / E2E Latency` | Client-side throughput metric |

#### Why TPS = output_tokens / E2E Latency?

**Incorrect approach:**
```python
TPS = output_tokens / TG  # ❌ WRONG
# Example: 38 tokens / 0.0007s = 52 285.714 tokens/sec
```
> Fifty-two thousand tokens per second? Goodbye H100, my local PC just destroyed you!

Yeah, no. This calculation is hilariously wrong.

This vastly overestimates performance because `TG` measures only the time between first and last chunk, not the actual token generation time.

**Correct approach:**
```python
TPS = output_tokens / E2E Latency  # ✅ CORRECT
# Example: 38 tokens / 0.6668s = 56.988 tokens/sec
```

This reflects real-world throughput from the client perspective.

**Limitation:** For very short outputs (1-2 chunks), `TG` may not accurately represent generation time. Server-side metrics would be more precise but are not currently collected.

---

### Per-Metric Percentiles
For each metric across all requests, La Perf computes:

| Percentile | Description |
|------------|-------------|
| **P25** | 25th percentile |
| **P50** | Median |
| **P75** | 75th percentile |
| **P95** | 95th percentile |

### Cross-Run Statistics
For multiple benchmark runs, statistics are calculated from percentile values across runs:
```python
mean(run1_percentile, run2_percentile, run3_percentile) ± std(run1_percentile, run2_percentile, run3_percentile)

# Example with P50 TPS:
# final_p50_tps = mean([run1_p50_tps, run2_p50_tps, run3_p50_tps])
# final_p50_tps_std = std([run1_p50_tps, run2_p50_tps, run3_p50_tps])
# On table you see: final_p50_tps ± final_p50_tps_std
```

**Note:** LLM/VLM compute percentiles per run first, then aggregate across runs. This differs from Embeddings which use direct mean/std.

These aggregated values appear in the results tables.

---
### Notes
- All timing values are wall-clock times measured via `time.perf_counter()`.
- Benchmarks are repeated at least 3 times to compute mean and standard deviation.
- All metrics are device-synchronized and exclude warmup runs.
