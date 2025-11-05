from dotenv import load_dotenv
import os
from src.system_info.device_info import get_device_info

load_dotenv()

# Detect device type for model selection
_device_info = get_device_info()
_is_apple_silicon = _device_info.get("device") == "mps"

# Backend Selection
# AUTO - Try LM Studio first, fallback to Ollama (uses one)
# LM_STUDIO - Use only LM Studio
# OLLAMA - Use only Ollama
# BOTH - Run benchmarks on both LM Studio AND Ollama (for comparison)
LLM_BACKEND = os.getenv("LLM_BACKEND", "BOTH")  # AUTO, LM_STUDIO, OLLAMA, or BOTH
LLM_API_KEY = os.getenv("LLM_API_KEY", "api-key")
LLM_DATA_SIZE = int(os.getenv("LLM_DATA_SIZE", "10"))

VLM_BACKEND = os.getenv("VLM_BACKEND", "BOTH")  # AUTO, LM_STUDIO, OLLAMA, or BOTH
VLM_API_KEY = os.getenv("VLM_API_KEY", os.getenv("LLM_API_KEY", "api-key"))
VLM_DATA_SIZE = int(os.getenv("VLM_DATA_SIZE", "10"))
# =============================================================================
# Embedding Model Settings
# =============================================================================
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "nomic-ai/modernbert-embed-base"
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DATA_SIZE = int(os.getenv("EMBEDDING_DATA_SIZE", "3000"))
EMBEDDING_MAX_LEN = int(os.getenv("EMBEDDING_MAX_LEN", "1024"))
# =============================================================================
# LM Studio Settings
# =============================================================================

# LM Studio Base URLs
LMS_LLM_BASE_URL = os.getenv("LMS_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LMS_VLM_BASE_URL = os.getenv("LMS_VLM_BASE_URL", LMS_LLM_BASE_URL)

# LM Studio Model Names - Device-aware defaults (MLX for Apple Silicon, GGUF for others)

# LLM Models
LMS_MLX_LLM_MODEL_NAME = os.getenv("LMS_MLX_LLM_MODEL_NAME", "openai/gpt-oss-20b")
LMS_GGUF_LLM_MODEL_NAME = os.getenv("LMS_GGUF_LLM_MODEL_NAME", "openai/gpt-oss-20b")

# VLM Models
LMS_MLX_VLM_MODEL_NAME = os.getenv("LMS_MLX_VLM_MODEL_NAME", "qwen/qwen3-vl-8b")
LMS_GGUF_VLM_MODEL_NAME = os.getenv("LMS_GGUF_VLM_MODEL_NAME", "qwen/qwen3-vl-8b")

# Active model selection based on device
LMS_LLM_MODEL_NAME = (
    LMS_MLX_LLM_MODEL_NAME if _is_apple_silicon else LMS_GGUF_LLM_MODEL_NAME
)
LMS_VLM_MODEL_NAME = (
    LMS_MLX_VLM_MODEL_NAME if _is_apple_silicon else LMS_GGUF_VLM_MODEL_NAME
)

# =============================================================================
# Ollama Settings
# =============================================================================

# Ollama Base URLs (typically http://127.0.0.1:11434)
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_VLM_BASE_URL = os.getenv("OLLAMA_VLM_BASE_URL", OLLAMA_LLM_BASE_URL)

# Ollama Model Names
OLLAMA_LLM_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL_NAME", "gpt-oss:20b")
OLLAMA_VLM_MODEL_NAME = os.getenv("OLLAMA_VLM_MODEL_NAME", "qwen3-vl:8b")

# =============================================================================
# Legacy compatibility (deprecated - use LMS_* instead)
# =============================================================================

LLM_BASE_URL = LMS_LLM_BASE_URL
LLM_MODEL_NAME = LMS_LLM_MODEL_NAME
VLM_BASE_URL = LMS_VLM_BASE_URL
VLM_MODEL_NAME = LMS_VLM_MODEL_NAME
