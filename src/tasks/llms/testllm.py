# --- добавлено: импорт таймера ---
import time

from openai import OpenAI
from src.settings import (LLM_API_KEY, 
                      LLM_MODEL_NAME, 
                      LM_STUDIO_URL
                    )
client = OpenAI(api_key=LLM_API_KEY,
                base_url=LM_STUDIO_URL)

prompt = """
Почему небо голубое
"""
# --- добавлено: STREAMING-замер TTFT и чистой скорости генерации ---

t0_stream = time.perf_counter()
stream = client.chat.completions.create(
    model=LLM_MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=2048,
    temperature=0,
    top_p=0.95,
    stream=True,
    stream_options={"include_usage": True}
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
            print(delta.content, end='', flush=True)

    # Final chunk may contain usage info
    if hasattr(chunk, 'usage') and chunk.usage:
        final_usage = chunk.usage

print()  # Newline after response

t1_stream = time.perf_counter()

total_latency_s_stream = t1_stream - t0_stream
ttft_s = (first_token_ts - t0_stream) if first_token_ts else None
generation_time_s = (t1_stream - first_token_ts) if first_token_ts else None

input_tokens_stream = final_usage.prompt_tokens if final_usage else None
output_tokens_stream = final_usage.completion_tokens if final_usage else None

tokens_per_sec_generation = (
    (output_tokens_stream / generation_time_s)
    if (output_tokens_stream and generation_time_s and generation_time_s > 0)
    else None
)

print("\n" + "="*60)
print(f"Model: {LLM_MODEL_NAME}")
print("="*60)

char_count = len(full_text)
chars_per_sec = round(char_count / generation_time_s, 2) if generation_time_s else None
tokens_per_sec = round(output_tokens_stream / generation_time_s, 2) if (output_tokens_stream and generation_time_s) else None

print({
    "total_latency_s": round(total_latency_s_stream, 3),
    "ttft_s": round(ttft_s, 3) if ttft_s else None,
    "generation_time_s": round(generation_time_s, 3) if generation_time_s else None,
    "input_tokens": input_tokens_stream,
    "output_tokens": output_tokens_stream,
    "output_chars": char_count,
    "tokens_per_sec": tokens_per_sec,
    "chars_per_sec": chars_per_sec,
})
# --- конец добавленного streaming-блока ---