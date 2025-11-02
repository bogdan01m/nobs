import time
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from src.settings import VLM_API_KEY, VLM_MODEL_NAME, VLM_BASE_URL


def stream_with_results(
    prompt: str, image=None, model_name: str | None = None, base_url: str | None = None
):
    # Use provided values or fallback to settings
    if model_name is None:
        model_name = VLM_MODEL_NAME
    if base_url is None:
        base_url = VLM_BASE_URL

    # Create client with appropriate base_url
    client = OpenAI(api_key=VLM_API_KEY, base_url=base_url)

    t0_stream = time.perf_counter()
    # For VLM, need to pass image as base64 (LM Studio requirement)
    if image:
        # image can be a PIL Image object or a path string
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        content: list[dict[str, object]] | str = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
    else:
        content = prompt

    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
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

    # print("usage", final_usage) # Debug
    t1_stream = time.perf_counter()

    total_latency_s_stream = t1_stream - t0_stream
    ttft_s = (first_token_ts - t0_stream) if first_token_ts else None
    generation_time_s = (t1_stream - first_token_ts) if first_token_ts else None

    input_tokens_stream = final_usage.prompt_tokens if final_usage else None
    output_tokens_stream = final_usage.completion_tokens if final_usage else None
    tokens_per_sec = (
        round(output_tokens_stream / generation_time_s, 4)
        if (output_tokens_stream and generation_time_s and generation_time_s > 0)
        else None
    )

    result = {
        "total_latency_s": round(total_latency_s_stream, 4),
        "ttft_s": round(ttft_s, 4) if ttft_s else None,
        "tg_s": round(generation_time_s, 4) if generation_time_s else None,
        "input_tokens": input_tokens_stream,
        "output_tokens": output_tokens_stream,
        "tokens_per_sec": tokens_per_sec,
    }

    return result
