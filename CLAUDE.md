# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NOBS (Neural NetwOrks Benchmark Stash) is an open-source benchmark suite for evaluating real AI hardware performance on practical workloads. The project focuses on reproducible, honest benchmarks across different hardware (M4 Max, RTX 4060, A100, etc.) for:

- **Text Embeddings** (sentence-transformers) - IMPLEMENTED
- **LLM Inference** (via LM Studio) - IMPLEMENTED
- **VLM/Vision-Language tasks** (via LM Studio) - IMPLEMENTED ✨ NEW
- **Power Metrics Monitoring** - IMPLEMENTED ✨ NEW
- **Diffusion image generation** - PLANNED

The philosophy is "no-bullshit" — measuring real models with real data, not synthetic benchmarks.

### Recent Updates

**VLM (Vision-Language Models) Benchmark** - Now fully implemented with:
- Support for multimodal inference (text + images)
- Integration with LM Studio VLM models
- Hallucination_COCO dataset for evaluation
- Same robust testing methodology (warmup + 3 runs with median metrics)

**Power Metrics Monitoring** - Comprehensive resource tracking:
- System-wide metrics: CPU utilization, RAM usage, battery drain
- GPU metrics: Utilization, VRAM usage, temperature (platform-specific)
- Power consumption: Detailed wattage tracking (NVIDIA via nvidia-smi, macOS via ioreg/powermetrics)
- Optional sudo powermetrics (macOS): Enhanced GPU/CPU power measurements with user consent
- Statistical analysis: p50 and p95 percentiles for all metrics
- Detailed logging: All samples logged to `logs/power_metrics.log`

**Device Timing Synchronization** - Improved accuracy:
- Added `torch.cuda.synchronize()` for CUDA devices
- Added `torch.mps.synchronize()` for Apple Silicon
- Ensures accurate timing measurements across all platforms

## Development Environment

### Package Management
This project uses `uv` as the package manager (not pip/poetry).

**Python Version**: 3.12+

### Running Benchmarks

Run interactive benchmarks:
```bash
uv run python main.py
```

This will:
1. Detect device (CUDA/MPS/CPU) and display hardware info
2. Prompt which benchmark tasks to run (embeddings, llms, vlms)
3. For LLMs/VLMs: optionally auto-setup LM Studio model
4. Start power monitoring (asks about sudo powermetrics on macOS)
5. Run selected benchmarks with 3 repeats + warmup
6. Calculate task scores using median-based metrics
7. Clear memory between models
8. Stop power monitoring and collect metrics
9. Save aggregated report with power metrics to `results/report_{host}.json`

Reports are saved as:
```
results/
├── report_Mac16,6.json
├── report_custom_pc.json
```

Each report contains `timestamp`, `device_info`, `tasks` array, and `power_metrics` object with results.

## Code Architecture

### Main Orchestrator (`main.py`)

Interactive entry point (265 lines) that:
1. Displays device info via `get_device_info()`
2. Prompts user to select benchmarks (embeddings, llms, vlms)
3. For LLMs/VLMs: asks about automated LM Studio setup
4. Starts PowerMonitor context manager for entire benchmark session
5. Runs selected benchmark tasks sequentially within PowerMonitor
6. Stops PowerMonitor and collects power metrics
7. Aggregates results from all tasks + power metrics
8. Saves combined report with power metrics to `results/report_{host}.json`

**Key improvements:**
- Power monitoring runs continuously throughout all benchmarks
- Cleaner code structure: all user prompts first, then execution
- Power metrics automatically included in final report
- Setup/cleanup handled within PowerMonitor context

### Core Utilities (`src/`)

**Device Detection (`device_info.py`)** - 291 lines
- `get_device_info()` returns dict with: `platform`, `device`, `ram_gb`, `processor`, `gpu_name`, `gpu_memory_gb`, `host`
- Platform-specific detection: Darwin (macOS), Linux, Windows
- Detects: CUDA (NVIDIA), MPS (Apple Silicon), ROCm (AMD), or CPU
- GPU core count detection for Apple Silicon (e.g., "32 cores")
- Uses `fastfetch` for detailed hardware info (if available)
- Comprehensive fallback chains for robust detection
- `host` field used for report filename (e.g., "Mac16,6")

**Memory Management (`memory_cleaner.py`)** - 14 lines
- `clear_memory()` - Cross-platform memory clearing
- Handles CUDA (`torch.cuda.empty_cache()` + synchronization)
- Handles MPS (`torch.mps.empty_cache()` + synchronization)
- Python garbage collection (`gc.collect()`)
- Called between model runs to prevent OOM

**Scoring System (`score.py`)** - 15 lines
- `calculate_score(num_tasks: int, total_time: float, C: int = 3600)`
- Formula: `score = num_tasks * C / total_time`
- Normalizes by 1 hour (3600 seconds)
- Higher scores = faster execution (better performance)
- Used for both embeddings and LLM tasks

**LM Studio Setup (`lm_studio_setup.py`)** - 265 lines
- `setup_lm_studio()` - Full automated pipeline:
  - Check for `lms` CLI availability
  - Download model via `lms get <model>` (if not present)
  - Start server via `lms server start`
  - Load model into memory via `lms load`
  - Verify server readiness with ping + test inference
- `cleanup_lm_studio()` - Post-benchmark cleanup:
  - Unload model via `lms unload`
  - Stop server via `lms server stop`
- Helper functions: `check_lms_cli()`, `is_model_downloaded()`, `is_server_running()`, `is_model_loaded()`
- Default model: `bartowski/QwQ-32B-Preview-GGUF/QwQ-32B-Preview-Q4_K_M.gguf`

**Settings (`settings.py`)** - 17 lines
- Loads environment variables from `.env` file
- **LLM Settings:**
  - `LLM_API_KEY` - API key for LM Studio (default: "api-key")
  - `LLM_BASE_URL` - API endpoint (default: "http://127.0.0.1:1234/v1")
  - `LLM_MODEL_NAME` - Model name (default: "gpt-oss-20b")
- **VLM Settings (falls back to LLM settings if not specified):**
  - `VLM_API_KEY` - API key for VLM (default: same as LLM_API_KEY)
  - `VLM_BASE_URL` - API endpoint (default: same as LLM_BASE_URL)
  - `VLM_MODEL_NAME` - Model name (default: "qwen/qwen3-vl-8b")

**Power Metrics (`power_metrics.py`)** - 664 lines ✨ NEW
- `PowerMetrics` class - Core monitoring functionality:
  - `__init__(interval=1.0, use_sudo_powermetrics=False)` - Initialize with sampling interval
  - `start()` - Begin background monitoring thread
  - `stop()` - Stop monitoring and return aggregated results (p50, p95 statistics)
  - Platform detection: Automatically detects CUDA, MPS, or CPU
  - Capability checking: Verifies power/battery monitoring availability
- **Sampling methods:**
  - `_sample_nvidia_power()` - NVIDIA GPU metrics via nvidia-smi (power, VRAM, utilization, temperature)
  - `_sample_macos_power()` - macOS system power via ioreg (battery discharge rate, charging status)
  - `_sample_macos_gpu()` - Apple Silicon GPU via ioreg (utilization, VRAM, temperature)
  - `_sample_system()` - Cross-platform CPU/RAM via psutil
- **Optional sudo powermetrics (macOS):**
  - `_start_powermetrics()` - Launches background sudo powermetrics process
  - `_stop_powermetrics()` - Terminates process gracefully
  - `_parse_powermetrics_log()` - Extracts GPU/CPU power from logs (in watts)
  - Requires user consent and sudo password
  - Provides detailed GPU/CPU power consumption (p50, p95)
- `PowerMonitor` context manager - Easy integration:
  - `__init__(interval=1.0, use_sudo_powermetrics=None, ask_user=True)` - Optional interactive prompt
  - `__enter__()` - Starts monitoring
  - `__exit__()` - Stops and stores results in `.results` attribute
  - Example: `with PowerMonitor() as pm: ... # pm.results contains metrics`
- **Logging:**
  - All samples logged to `logs/power_metrics.log` in JSON format
  - Includes timestamp and all metric values
  - Logged initialization, start, stop, and final statistics

### Task Structure (`src/tasks/<task_name>/`)

Each task follows this pattern:
- `runner.py` - Orchestrates models, returns dict with `"task"`, `"dataset_size"`, `"models"`, `"task_score"`
- `executor.py` - Runs individual model with repeats, handles lifecycle
- Results use median values from multiple runs (not mean)

**Text Embeddings (`src/tasks/text_embeddings/`)** - IMPLEMENTED

`runner.py` (49 lines):
- `run_embeddings_benchmark()` - Main entry point
- Loads first 100 rows from IMDB dataset via `from src.data.imdb_data import dataset`
- Tests 2 models:
  - `thenlper/gte-large`
  - `nomic-ai/modernbert-embed-base`
- Runs each model with `run_model_with_repeats(num_repeats=3)`
- Calculates `task_score` using `calculate_score(num_tasks=num_models, total_time=median_sum)`

`executor.py` (172 lines):
- `run_single_model(model_name, texts, batch_size=16, max_seq_length=512)` - Single execution
  - Loads via `SentenceTransformer(model_name, trust_remote_code=True)`
  - Times `model.encode(texts, show_progress_bar=True, batch_size=batch_size)`
  - Returns: `model_name`, `device`, `encoding_time_seconds`, `rows_per_second`, `embedding_dimension`, `dtype`
- `run_model_with_repeats(model_name, texts, num_repeats=3, batch_size=16)` - Multiple runs
  - First run (50 samples): warmup, excluded from results
  - Subsequent runs (100 samples): actual measurements
  - Calculates median values across all repeats
  - Returns: `model_name`, `device`, median metrics, `all_runs` array
- Cleans memory after each run

**LLMs (`src/tasks/llms/`)** - IMPLEMENTED

`runner.py` (44 lines):
- `run_llms_benchmark()` - Main entry point
- Loads first 3 prompts from awesome-chatgpt-prompts dataset
- Runs benchmark with `run_model_with_repeats(num_repeats=3)`
- Calculates `task_score` using `calculate_score(num_tasks=num_prompts, total_time=final_median_latency)`

`executor.py` (149 lines):
- `run_single_model(prompts)` - Single execution
  - Streams prompts via `chat_stream.chat_stream()`
  - Collects latency and token metrics per prompt
  - Returns: aggregate stats including median latency, TTFT, tokens/sec
- `run_model_with_repeats(prompts, num_repeats=3)` - Multiple runs
  - First run (1 prompt): warmup, excluded from results
  - Subsequent runs (full prompt set): actual measurements
  - Calculates final median values across all repeats
  - Returns: `model_name`, final medians, `all_runs` array

`chat_stream.py` (82 lines):
- `chat_stream(user_message, system_prompt="You are a helpful assistant.")` - Streaming interface
- Uses OpenAI Python client to communicate with LM Studio
- Streams tokens and measures:
  - `total_latency_s` - Total request time
  - `ttft_s` - Time To First Token
  - `generation_time_s` - Time after first token
  - `input_tokens`, `output_tokens` - Token counts
  - `chars_per_sec`, `tokens_per_sec` - Throughput metrics
- Default configuration from `settings.py`

**VLMs (`src/tasks/vlms/`)** - IMPLEMENTED ✨ NEW

`runner.py` (41 lines):
- `run_vlms_benchmark()` - Main entry point
- Loads first 3 questions + images from Hallucination_COCO dataset via `from src.data.hallucination_coco import dataset`
- Runs benchmark with `run_model_with_repeats(model_name=VLM_MODEL_NAME, prompts=questions, images=images, num_runs=3)`
- Uses median latency as total time metric
- Returns dict with: `"task": "vlms"`, `"dataset": "Hallucination_COCO"`, `"num_questions"`, `"model"`, `"total_time_seconds"`

`executor.py` (181 lines):
- `run_single_model(prompts, images=None)` - Single execution with optional images
  - Streams prompts via `chat_stream.stream_with_results()`
  - Supports text-only or multimodal (text + image) inference
  - Collects latency and token metrics per prompt with tqdm progress bar
  - Returns: aggregate stats including median latency, TTFT, tokens/sec
- `run_model_with_repeats(model_name, prompts, images=None, num_runs=3)` - Multiple runs
  - First run (1 prompt + image): warmup, excluded from results
  - Subsequent runs (full prompt set): actual measurements
  - Calculates final median values and standard deviations across all repeats
  - Returns: `model_name`, final medians (p50), standard deviations, `all_runs` array
- Memory cleared after each run

`chat_stream.py` (91 lines):
- `stream_with_results(prompt, image=None)` - Streaming interface with optional image
- Converts PIL Image to base64 for LM Studio multimodal API
- Supports both file paths and PIL Image objects
- Uses OpenAI Python client with vision format:
  - Text-only: `content = prompt`
  - Multimodal: `content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]`
- Measures same metrics as LLM chat_stream: latency, TTFT, tokens/sec
- Configuration from `VLM_*` settings in `settings.py`

### Data Loading (`src/data/`)

**IMDB Dataset (`imdb_data.py`)**
- Loads IMDB train split (25,000 reviews) via HuggingFace `datasets`
- Source: `load_dataset("imdb")`
- Exports as `dataset` for import by task runners
- Used for text embeddings benchmarks (first 100 rows)

**Awesome ChatGPT Prompts (`awesome_prompts.py`)**
- Loads creative prompts via HuggingFace `datasets`
- Source: `load_dataset("fka/awesome-chatgpt-prompts")`
- Exports as `dataset` for import by task runners
- Used for LLM inference benchmarks (first 3 prompts)

**Hallucination_COCO Dataset (`hallucination_coco.py`)** ✨ NEW
- Loads vision-language hallucination detection dataset via HuggingFace `datasets`
- Source: `load_dataset("DogNeverSleep/Hallucination_COCO")`
- Contains questions + PIL Image objects for multimodal evaluation
- Exports as `dataset` for import by task runners
- Used for VLM benchmarks (first 3 questions + images)

## Dependencies

Core dependencies (from `pyproject.toml`):
- `torch>=2.9.0` - Deep learning framework
- `transformers>=4.57.1` - Model architectures
- `sentence-transformers>=5.1.1` - Embedding models
- `openai>=2.5.0` - OpenAI API client (for LM Studio)
- `datasets>=4.2.0` - Dataset loading (HuggingFace)
- `accelerate>=1.10.1` - Distributed training utilities
- `psutil>=7.1.0` - System monitoring
- `python-dotenv>=1.1.1` - Environment variable loading
- `requests>=2.32.5` - HTTP client
- `pydantic>=2.12.3` - Data validation
- `scikit-learn>=1.7.2` - ML utilities
- `numpy>=2.3.4`, `tqdm>=4.67.1` - Utilities

Dev dependencies:
- `ruff>=0.8.0` - Linting and formatting
- `mypy>=1.13.0` - Type checking
- `bandit>=1.7.10` - Security scanning
- `pre-commit>=4.3.0` - Git hooks

## Code Quality

**Pre-commit Hooks (`.pre-commit-config.yaml`)**
- Standard checks: trailing whitespace, file endings, YAML syntax, merge conflicts
- Ruff: Fast Python linting and auto-formatting (only `src/` and `main.py`)
- mypy: Type checking with ignore-missing-imports
- bandit: Security vulnerability detection (severity=high, confidence=high)

**GitHub Actions CI/CD (`.github/workflows/code-quality.yml`)**
- Runs on push and pull requests
- Python 3.12 environment
- Executes pre-commit checks on all files
- Cached pre-commit environments for speed

## Adding New Benchmarks

**Add model to embeddings:**
1. In `src/tasks/text_embeddings/runner.py`, add variable: `new_model = "org/model-name"`
2. Add to models list and run: `results = run_model_with_repeats(model_name=new_model, texts=texts, num_repeats=3, batch_size=16)`
3. Memory cleaning is handled automatically in executor

**Add new task type (e.g., vlm):**
1. Create `src/tasks/vlm/runner.py` with `run_vlm_benchmark()` returning dict with:
   - `"task": "vlm"`
   - `"dataset_size": N`
   - `"models": {...}`
   - `"task_score": X.XX`
2. Create `src/tasks/vlm/executor.py` with:
   - `run_single_model()` - Single execution logic
   - `run_model_with_repeats()` - Multiple runs with warmup + median calculation
3. Import in `main.py` and add prompt for user selection
4. Results auto-aggregated into same report file

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
  "tasks": [
    {
      "task": "embeddings",
      "dataset_size": 100,
      "models": {
        "thenlper/gte-large": {
          "model_name": "thenlper/gte-large",
          "device": "mps",
          "final_median_encoding_time_seconds": 2.89,
          "final_median_rows_per_second": 34.55,
          "embedding_dimension": 1024,
          "dtype": "float32",
          "all_runs": [...]
        },
        "nomic-ai/modernbert-embed-base": {
          "model_name": "nomic-ai/modernbert-embed-base",
          "device": "mps",
          "final_median_encoding_time_seconds": 2.76,
          "final_median_rows_per_second": 36.30,
          "embedding_dimension": 768,
          "dtype": "float32",
          "all_runs": [...]
        }
      },
      "task_score": 637.17
    },
    {
      "task": "llms",
      "dataset_size": 3,
      "models": {
        "gpt-oss-20b": {
          "model_name": "gpt-oss-20b",
          "final_median_latency_s": 22.81,
          "final_median_ttft_s": 6.50,
          "final_median_tokens_per_sec": 168.14,
          "all_runs": [...]
        }
      },
      "task_score": 157.84
    }
  ],
  "power_metrics": {
    "duration_seconds": 180.5,
    "num_samples": 181,
    "cpu_utilization_percent_p50": 45.2,
    "cpu_utilization_percent_p95": 78.3,
    "ram_used_gb_p50": 24.5,
    "ram_used_gb_p95": 28.1,
    "gpu_utilization_percent_p50": 82.4,
    "gpu_utilization_percent_p95": 95.7,
    "gpu_vram_used_gb_p50": 12.3,
    "gpu_vram_used_gb_p95": 14.8,
    "battery_drain_rate_w_p50": 18.5,
    "battery_drain_rate_w_p95": 32.1
  }
}
```

## Important Patterns

1. **Memory Management**: Always call `clear_memory()` after model execution (handled automatically in executors)
2. **Device Agnostic**: Code works on CUDA, MPS, CPU, ROCm without modification
3. **Device Synchronization**: Always call `sync_device()` before/after timing measurements for accurate results
4. **Model Names as Keys**: Use full model name (e.g., `"thenlper/gte-large"`) as dict key
5. **Trust Remote Code**: Use `trust_remote_code=True` when loading models
6. **Progress Bars**: Use `show_progress_bar=True` for user feedback
7. **Median-Based Metrics**: Use median values from multiple runs (not mean) to reduce outlier impact
8. **Warmup Runs**: First run is warmup (smaller dataset), excluded from final results
9. **Multiple Repeats**: Run each model 3 times, calculate medians for reproducibility
10. **Task Scoring**: Use `calculate_score()` with median times for consistent comparison
11. **Interactive Flow**: Use `input()` prompts for user selection of benchmarks
12. **Power Monitoring**: Use `PowerMonitor` context manager for easy integration, optional sudo on macOS
13. **Image Encoding**: VLM images must be base64-encoded for LM Studio multimodal API

## Environment Configuration

Create a `.env` file in the root directory:
```env
HF_HUB_ENABLE_HF_TRANSFER=1  # Speeds up HuggingFace model downloads

# LLM Settings (for text-only inference)
LLM_API_KEY=api-key  # API key for LM Studio (default works for local)
LLM_BASE_URL=http://127.0.0.1:1234/v1  # LM Studio endpoint
LLM_MODEL_NAME=gpt-oss-20b  # Model name for LLM benchmarks

# VLM Settings (for multimodal vision-language inference)
# Falls back to LLM settings if not specified
VLM_API_KEY=api-key  # API key for VLM (defaults to LLM_API_KEY)
VLM_BASE_URL=http://127.0.0.1:1234/v1  # VLM endpoint (defaults to LLM_BASE_URL)
VLM_MODEL_NAME=qwen/qwen3-vl-8b  # VLM model name (default: qwen/qwen3-vl-8b)
```

## Makefile Utilities

Project includes a Makefile for common tasks:

```bash
make          # Generate benchmark results tables (default)
make generate # Generate benchmark results tables from JSON files
make format   # Run pre-commit hooks on all files
make lint     # Run ruff linter only
make clean    # Clean Python cache files
make help     # Show available commands
```

The `make generate` command processes `results/*.json` files and creates markdown tables for the README.

## LM Studio Integration

For LLM benchmarks, you need LM Studio with the `lms` CLI:

**Manual Setup:**
1. Install LM Studio from https://lmstudio.ai/
2. Download a GGUF model (e.g., `bartowski/QwQ-32B-Preview-GGUF`)
3. Start server: `lms server start`
4. Load model: `lms load <model-path>`

**Automated Setup (Recommended):**
- When prompted, select "y" for auto-setup
- The script will download, start, and load the model automatically
- Cleanup is automatic after benchmark completes

**Verification:**
- Server running: `lms server status`
- Model loaded: `lms ps`
- Test inference: Check `http://127.0.0.1:1234/v1/chat/completions`

## Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce batch size in executor calls
2. **MPS fallback to CPU**: Check PyTorch MPS support for your macOS version
3. **LM Studio connection errors**: Verify server is running and model is loaded
4. **Missing dependencies**: Run `uv sync` to install all packages
5. **Pre-commit failures**: Run `uv run pre-commit run --all-files` to see details

**Performance Tips:**

- Use batch_size=16 for embeddings on 32GB+ RAM systems
- For LLMs, use quantized models (Q4_K_M) for faster inference
- Clear GPU memory between runs using `clear_memory()`
- First run is always slower (warmup), subsequent runs are more representative
- user precommit without uv
