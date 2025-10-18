import time
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from src.data.imdb_data import dataset
from src.memory_cleaner import clear_memory
from src.device_info import get_device_info


def run_embeddings(model_name: str, model_key: str, texts: list, batch_size: int = 16):
    """
    Загружает модель, векторизует тексты, замеряет время и очищает память

    Args:
        model_name: Имя модели на HuggingFace
        model_key: Ключ для сохранения результатов
        texts: Список текстов для векторизации
        batch_size: Размер батча

    Returns:
        dict: Результаты бенчмарка
    """
    print("=" * 50)
    print(f"{model_key.upper()} MODEL")
    print("=" * 50)

    # Загрузка модели
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print(f"Model loaded, max_seq_length: {model.max_seq_length}")

    # Векторизация с замером времени
    print("Encoding texts...")
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    encoding_time = time.time() - start_time

    print(f"Encoding completed in {encoding_time:.2f}s")
    print(f"Speed: {len(texts) / encoding_time:.2f} texts/sec")

    # Результаты
    results = {
        "model_name": model_name,
        "encoding_time_seconds": round(encoding_time, 2),
        "texts_per_second": round(len(texts) / encoding_time, 2)
    }

    # Очистка памяти
    print(f"\nClearing {model_key} model from memory...")
    del model, embeddings
    clear_memory()

    return results


# Информация об устройстве
device_info = get_device_info()
print("=" * 50)
print("DEVICE INFO")
print("=" * 50)
print(json.dumps(device_info, indent=2))
print()

print(f"Dataset loaded: {dataset}")
print(f"Number of rows: {len(dataset)}\n")

# Подготовка результатов
texts = dataset['text'][:1000]  # Берем первые 1000 строк для теста
results = {
    "timestamp": datetime.now().isoformat(),
    "device_info": device_info,
    "dataset_size": len(texts),
    "models": {}
}

# Запуск бенчмарков
results["models"]["gte"] = run_embeddings(
    model_name="thenlper/gte-large",
    model_key="gte",
    texts=texts,
    batch_size=16
)

results["models"]["modernbert"] = run_embeddings(
    model_name="nomic-ai/modernbert-embed-base",
    model_key="modernbert",
    texts=texts,
    batch_size=16
)

# Сохранение финального отчета
print("=" * 50)
print("SAVING REPORT")
print("=" * 50)
report_path = "benchmark_report.json"
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Report saved to: {report_path}\n")
print("=" * 50)
print("BENCHMARK RESULTS")
print("=" * 50)
print(json.dumps(results, indent=2))
print("=" * 50)