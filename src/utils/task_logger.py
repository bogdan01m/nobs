"""
Task logging utility for benchmark tasks.

Logs task inputs and outputs to separate log files in logs/ directory:
- logs/embeddings.log - for embedding tasks
- logs/llm.log - for LLM tasks
- logs/vlm.log - for VLM tasks
"""

from datetime import datetime
from pathlib import Path


def _get_log_path(task_type: str) -> Path:
    """Get the path to the log file for the given task type."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir / f"{task_type}.log"


def _write_log(log_path: Path, message: str):
    """Append message to log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def log_embedding_task(text: str, embedding_preview: list, shape: tuple):
    """
    Log embedding task: input text preview, output embedding preview, and shape.

    Args:
        text: Input text (will be truncated to 50 chars)
        embedding_preview: First few values of the embedding vector
        shape: Shape of the embedding array (e.g., (768,))
    """
    log_path = _get_log_path("embeddings")
    text_preview = text[:50] + ("..." if len(text) > 50 else "")
    embedding_str = ", ".join(f"{v:.4f}" for v in embedding_preview[:3])

    message = (
        f"INPUT: '{text_preview}' | "
        f"OUTPUT: [{embedding_str}...] | "
        f"SHAPE: {shape}"
    )
    _write_log(log_path, message)


def log_llm_task(prompt: str, response: str):
    """
    Log LLM task: input prompt preview and output response preview.

    Args:
        prompt: Input prompt (will be truncated to 50 chars)
        response: Model response (will be truncated to 200 chars)
    """
    log_path = _get_log_path("llm")
    prompt_preview = prompt[:50] + ("..." if len(prompt) > 50 else "")
    response_preview = response[:200] + ("..." if len(response) > 200 else "")

    message = f"PROMPT: '{prompt_preview}' | RESPONSE: '{response_preview}'"
    _write_log(log_path, message)


def log_vlm_task(prompt: str, image_b64_preview: str | None, response: str):
    """
    Log VLM task: input prompt preview, base64 image preview, and output response preview.

    Args:
        prompt: Input prompt (will be truncated to 50 chars)
        image_b64_preview: First 5 lines of base64 encoded image (or None if text-only)
        response: Model response (will be truncated to 200 chars)
    """
    log_path = _get_log_path("vlm")
    prompt_preview = prompt[:50] + ("..." if len(prompt) > 50 else "")
    response_preview = response[:200] + ("..." if len(response) > 200 else "")

    if image_b64_preview:
        # Take first 5 lines (assuming ~80 chars per line)
        lines = [
            image_b64_preview[i : i + 80]
            for i in range(0, min(400, len(image_b64_preview)), 80)
        ]
        b64_preview = "\n    ".join(lines[:5])
        message = (
            f"PROMPT: '{prompt_preview}' | "
            f"IMAGE_B64 (preview):\n    {b64_preview}... | "
            f"RESPONSE: '{response_preview}'"
        )
    else:
        message = (
            f"PROMPT: '{prompt_preview}' | "
            f"IMAGE_B64: None | "
            f"RESPONSE: '{response_preview}'"
        )

    _write_log(log_path, message)
