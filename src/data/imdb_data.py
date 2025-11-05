from datasets import load_dataset
from src.settings import EMBEDDING_DATA_SIZE

imdb = load_dataset("imdb")
dataset = imdb["train"][:EMBEDDING_DATA_SIZE]
