from dotenv import load_dotenv
import os
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "api-key")
LM_STUDIO_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss-20b")