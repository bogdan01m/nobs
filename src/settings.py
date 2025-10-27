from dotenv import load_dotenv
import os

load_dotenv()

# LLM Settings
LLM_API_KEY = os.getenv("LLM_API_KEY", "api-key")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss-20b")

# VLM Settings (defaults to LLM settings if not specified)
VLM_API_KEY = os.getenv("VLM_API_KEY", os.getenv("LLM_API_KEY", "api-key"))
VLM_BASE_URL = os.getenv(
    "VLM_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
)
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "qwen/qwen3-vl-8b")
