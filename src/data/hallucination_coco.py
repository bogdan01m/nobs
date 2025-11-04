from datasets import load_dataset

ds = load_dataset("DogNeverSleep/Hallucination_COCO")
dataset = ds["train"][:1]
