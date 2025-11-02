from dotenv import load_dotenv
import os

load_dotenv()

# Backend Selection
# AUTO - Try LM Studio first, fallback to Ollama (uses one)
# LM_STUDIO - Use only LM Studio
# OLLAMA - Use only Ollama
# BOTH - Run benchmarks on both LM Studio AND Ollama (for comparison)
LLM_BACKEND = os.getenv("LLM_BACKEND", "BOTH")  # AUTO, LM_STUDIO, OLLAMA, or BOTH
VLM_BACKEND = os.getenv("VLM_BACKEND", "BOTH")  # AUTO, LM_STUDIO, OLLAMA, or BOTH

# LM Studio Settings
LLM_API_KEY = os.getenv("LLM_API_KEY", "api-key")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss-20b")

# VLM Settings (defaults to LLM settings if not specified)
VLM_API_KEY = os.getenv("VLM_API_KEY", os.getenv("LLM_API_KEY", "api-key"))
VLM_BASE_URL = os.getenv(
    "VLM_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
)
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "qwen/qwen3-vl-8b")

# Ollama Settings
# For Ollama, BASE_URL is typically http://127.0.0.1:11434
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_LLM_MODEL_NAME = os.getenv("OLLAMA_LLM_MODEL_NAME", "gpt-oss:20b")

OLLAMA_VLM_BASE_URL = os.getenv(
    "OLLAMA_VLM_BASE_URL", os.getenv("OLLAMA_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
)
OLLAMA_VLM_MODEL_NAME = os.getenv("OLLAMA_VLM_MODEL_NAME", "qwen3-vl:8b")
