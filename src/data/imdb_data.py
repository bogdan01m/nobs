from datasets import load_dataset

imdb = load_dataset("imdb")
dataset = imdb["train"]