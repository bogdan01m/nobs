"""Data loading utilities for benchmark results."""

import json
from pathlib import Path
from typing import Any


def _normalize_processor_name(processor: str) -> str:
    """Extract clean CPU model name from processor string.

    Handles cases where processor field contains full lscpu output
    (e.g., from VMs with both "Model name:" and "BIOS Model name:").

    Args:
        processor: Raw processor string from device_info

    Returns:
        Cleaned CPU model name
    """
    if not processor:
        return ""

    # Если это короткая строка (уже нормализована), возвращаем как есть
    if len(processor) < 200:
        return processor.strip()

    # Длинная строка - скорее всего полный вывод lscpu
    # Ищем "Model name:" используя split по "Model name:"
    if "Model name:" in processor:
        # Находим первое вхождение "Model name:" (игнорируем "BIOS Model name:")
        parts = processor.split("Model name:", 1)
        if len(parts) > 1:
            # Берём текст после "Model name:" до следующего поля (обычно разделено словом с заглавной буквы)
            # Ищем следующее поле lscpu (обычно "BIOS" или "CPU family")
            rest = parts[1].strip()
            # Берём до следующего известного ключевого слова lscpu
            for keyword in [
                " BIOS ",
                " CPU family:",
                " Model:",
                " Thread(s)",
                " Core(s)",
                " Socket(s)",
            ]:
                if keyword in rest:
                    rest = rest.split(keyword)[0]
                    break
            return rest.strip()

    # Фолбэк: возвращаем первые 100 символов
    return processor[:100].strip() + "..."


def _normalize_host_name(host: str) -> str:
    """Extract clean system model name from verbose host strings.

    Examples:
        "ASUSTeK COMPUTER INC. ASUS Vivobook Pro 15 N6506MV_N6506MV 1.0"
        -> "ASUS Vivobook Pro 15 N6506MV"

    Args:
        host: Raw host/system name from device_info

    Returns:
        Cleaned system model name
    """
    if not host:
        return ""

    host = host.strip()

    # Split into words
    words = host.split()

    # Filter out verbose corporate markers and version numbers
    filtered = []
    skip_next = False

    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue

        # Skip corporate markers
        word_upper = word.upper()
        if word_upper in {"INC.", "CORPORATION", "CORP.", "LTD.", "CO.,", "INC", "LTD"}:
            continue

        # Skip standalone version numbers (e.g., "1.0", "2.0")
        if word.replace(".", "").isdigit() and word.count(".") <= 2:
            continue

        # Skip words that look like manufacturer codes before actual brand
        # (e.g., "ASUSTeK" when "ASUS" follows)
        if i < len(words) - 1:
            next_word = words[i + 1]
            # If current word contains the next word (e.g., "ASUSTeK" contains "ASUS")
            if len(word) > 4 and len(next_word) > 3 and next_word.upper() in word_upper:
                continue

        filtered.append(word)

    result = " ".join(filtered)

    # Clean up duplicate model codes in format "CODE_CODE" -> "CODE"
    result_words = []
    for word in result.split():
        if "_" in word:
            parts = word.split("_")
            if len(parts) == 2 and parts[0] == parts[1]:
                result_words.append(parts[0])
            else:
                result_words.append(word)
        else:
            result_words.append(word)

    return " ".join(result_words).strip()


def load_results(results_dir: Path = Path("results")) -> list[dict[str, Any]]:
    """Load all result JSON files from the results directory.

    Args:
        results_dir: Directory containing report_*.json files

    Returns:
        List of result dictionaries sorted by host name
    """
    results = []
    for json_file in results_dir.glob("report_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            # Нормализуем processor name для старых JSON с полным выводом lscpu
            if "device_info" in data and "processor" in data["device_info"]:
                data["device_info"]["processor"] = _normalize_processor_name(
                    data["device_info"]["processor"]
                )
            # Нормализуем host name для читаемости в графиках
            if "device_info" in data and "host" in data["device_info"]:
                data["device_info"]["host"] = _normalize_host_name(
                    data["device_info"]["host"]
                )
            results.append(data)
    return sorted(results, key=lambda x: x["device_info"]["host"])
