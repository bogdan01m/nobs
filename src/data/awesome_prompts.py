from datasets import load_dataset

_dataset = load_dataset("fka/awesome-chatgpt-prompts")
dataset = _dataset["train"][:1]
