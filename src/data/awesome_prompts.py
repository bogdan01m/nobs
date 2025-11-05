from datasets import load_dataset
from src.settings import LLM_DATA_SIZE

_dataset = load_dataset("fka/awesome-chatgpt-prompts")
dataset = _dataset["train"][:LLM_DATA_SIZE]
