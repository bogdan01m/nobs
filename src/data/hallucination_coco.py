from datasets import load_dataset
from src.settings import VLM_DATA_SIZE

ds = load_dataset("DogNeverSleep/Hallucination_COCO")
dataset = ds["train"][:VLM_DATA_SIZE]
