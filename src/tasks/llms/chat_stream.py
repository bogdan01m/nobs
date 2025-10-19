# --- добавлено: импорт таймера ---
import time

from openai import OpenAI
from src.settings import LLM_API_KEY, LLM_MODEL_NAME, LLM_BASE_URL

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def stream_with_results(prompt: str):
    t0_stream = time.perf_counter()
    stream = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2048,
        temperature=0,
        top_p=0.95,
        stream=True,
        stream_options={"include_usage": True},
    )

    first_token_ts = None
    chunk_count = 0
    final_usage = None
    full_text = ""

    for chunk in stream:
        # Check if this chunk has content (first token arrives)
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                if first_token_ts is None:
                    first_token_ts = time.perf_counter()
                chunk_count += 1
                full_text += delta.content
                # Removed print for benchmark - we don't need to see the response

        # Final chunk may contain usage info
        if hasattr(chunk, "usage") and chunk.usage:
            final_usage = chunk.usage

    t1_stream = time.perf_counter()

    total_latency_s_stream = t1_stream - t0_stream
    ttft_s = (first_token_ts - t0_stream) if first_token_ts else None
    generation_time_s = (t1_stream - first_token_ts) if first_token_ts else None

    input_tokens_stream = final_usage.prompt_tokens if final_usage else None
    output_tokens_stream = final_usage.completion_tokens if final_usage else None

    # Calculate pure generation time (excluding TTFT/prompt processing)
    pure_generation_time = (
        (total_latency_s_stream - ttft_s) if ttft_s else total_latency_s_stream
    )

    char_count = len(full_text)
    chars_per_sec = (
        round(char_count / pure_generation_time, 4)
        if pure_generation_time and pure_generation_time > 0
        else None
    )
    tokens_per_sec = (
        round(output_tokens_stream / pure_generation_time, 4)
        if (output_tokens_stream and pure_generation_time and pure_generation_time > 0)
        else None
    )

    result = {
        "total_latency_s": round(total_latency_s_stream, 4),
        "ttft_s": round(ttft_s, 4) if ttft_s else None,
        "generation_time_s": round(generation_time_s, 4) if generation_time_s else None,
        "input_tokens": input_tokens_stream,
        "output_tokens": output_tokens_stream,
        "output_chars": char_count,
        "tokens_per_sec": tokens_per_sec,
        "chars_per_sec": chars_per_sec,
    }

    return result


# --- конец добавленного streaming-блока ---
