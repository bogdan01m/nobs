# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NOBS (Neural NetwOrks Benchmark Stash) is an open-source benchmark suite for evaluating real AI hardware performance on practical workloads. The project focuses on reproducible, honest benchmarks across different hardware (M4 Max, RTX 4060, A100, etc.) for:

- Embeddings (sentence-transformers)
- LLM inference
- VLM/Vision-Language tasks
- Diffusion image generation

The philosophy is "no-bullshit" — measuring real models with real data, not synthetic benchmarks.

## Development Environment

### Package Management
This project uses `uv` as the package manager (not pip/poetry).

### Running Benchmarks

Run all benchmarks:
```bash
uv run python -m src.main
```

This will:
1. Detect device (CUDA/MPS/CPU) and get host identifier
2. Run all enabled benchmark tasks (embeddings, llms, etc.)
3. Clear memory between models
4. Save report to `results/report_{host}.json`

Reports are saved as:
```
results/
├── report_Mac16,6.json
├── report_custom_pc.json
```

Each report contains `timestamp`, `device_info`, `task`, and `models` results.

## Code Architecture

### Main Orchestrator (`src/main.py`)

Entry point that:
1. Calls `get_device_info()` to detect hardware
2. Runs benchmark tasks (currently: embeddings)
3. Saves results to `results/report_{host}.json` using `save_report()`

### Core Utilities (`src/`)

**Device Detection (`device_info.py`)**
- `get_device_info()` returns dict with: `platform`, `device`, `ram_gb`, `processor`, `gpu_name`, `gpu_memory_gb`, `host`
- Detects: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- Uses `fastfetch` for detailed hardware info
- `host` field used for report filename (e.g., "Mac16,6")

**Memory Management (`memory_cleaner.py`)**
- `clear_memory()` - Cross-platform memory clearing
- Handles CUDA (`torch.cuda.empty_cache()`) and MPS (`torch.mps.empty_cache()`)
- Called between model runs to prevent OOM

### Task Structure (`src/tasks/<task_name>/`)

Each task has:
- `runner.py` - Orchestrates models, returns `{"task": "...", "dataset_size": N, "models": {...}}`
- `executor.py` - Runs individual model, handles lifecycle

**Embeddings (`src/tasks/embeddings/`)**

`runner.py`:
- `run_embeddings_benchmark()` - Main entry point
- Loads IMDB dataset via `from src.data.imdb_data import dataset`
- Defines models as variables (e.g., `gte_model = "thenlper/gte-large"`)
- Uses model name as dict key: `results["models"][gte_model] = run_single_model(...)`

`executor.py`:
- `run_single_model(model_name, model_key, texts, batch_size)` - Single model execution
- Loads via `SentenceTransformer(model_name, trust_remote_code=True)`
- Times `model.encode(texts, show_progress_bar=True, batch_size=batch_size)`
- Returns: `{"model_name": "...", "encoding_time_seconds": X, "texts_per_second": Y}`
- Cleans memory before returning

### Data Loading (`src/data/`)

**IMDB Dataset (`imdb_data.py`)**
- Loads IMDB train split (25,000 reviews) via HuggingFace `datasets`
- Exports as `dataset` for import by task runners

## Adding New Benchmarks

**Add model to embeddings:**
1. In `src/tasks/embeddings/runner.py`, add variable: `new_model = "org/model-name"`
2. Add call: `results["models"][new_model] = run_single_model(model_name=new_model, model_key="short_name", texts=texts, batch_size=16)`

**Add new task type (e.g., llms):**
1. Create `src/tasks/llms/runner.py` with `run_llms_benchmark()` returning dict with `"task": "llms"`
2. Create `src/tasks/llms/executor.py` with model execution logic
3. Import in `src/main.py` and call in `main()` function
4. Results auto-saved to same report file

## Output Format

```json
{
  "timestamp": "2025-10-19T00:03:52.634101",
  "device_info": {
    "platform": "Darwin",
    "device": "mps",
    "ram_gb": 36.0,
    "processor": "Apple M4 Max (14)",
    "gpu_name": "Apple M4 Max (32 cores)",
    "gpu_memory_gb": "shared with system RAM",
    "host": "Mac16,6"
  },
  "task": "embeddings",
  "dataset_size": 25000,
  "models": {
    "thenlper/gte-large": {
      "model_name": "thenlper/gte-large",
      "encoding_time_seconds": 931.3,
      "texts_per_second": 26.84
    }
  }
}
```

## Important Patterns

1. **Memory Management**: Always call `clear_memory()` after model execution
2. **Device Agnostic**: Code works on CUDA, MPS, CPU without modification
3. **Model Names as Keys**: Use full model name (e.g., `"thenlper/gte-large"`) as dict key
4. **Trust Remote Code**: Use `trust_remote_code=True` when loading models
5. **Progress Bars**: Use `show_progress_bar=True` for user feedback
