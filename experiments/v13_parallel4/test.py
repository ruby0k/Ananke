import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline import post, run

MODEL = "v11-experts23"
PARALLEL = 4
MAX_TOKENS = 256
OUT = Path(__file__).parent / "report.json"
PROMPTS = [
    "Explain why the sky is blue in detail.",
    "Explain how binary search works in detail.",
    "Explain why seasons occur on Earth in detail.",
    "Explain how a hash table works in detail.",
]


def generate(instance: str, prompt: str) -> dict:
    return post("/api/v0/chat/completions", {
        "model": instance, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 0, "reasoning_effort": "low",
    })


def main() -> None:
    run("lms", "server", "start", check=False)
    run("lms", "unload", "--all", check=False)
    loaded = post("/api/v1/models/load", {
        "model": MODEL, "context_length": 1024 * PARALLEL, "parallel": PARALLEL, "flash_attention": True,
        "offload_kv_cache_to_gpu": True, "num_experts": 2, "echo_load_config": True,
    })
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        responses = list(pool.map(lambda prompt: generate(loaded["instance_id"], prompt), PROMPTS))
    elapsed = time.perf_counter() - start
    tokens = sum(response["usage"]["completion_tokens"] for response in responses)
    report = {"model": MODEL, "parallel": PARALLEL, "elapsed_seconds": elapsed, "completion_tokens": tokens,
              "aggregate_tokens_per_second": tokens / elapsed,
              "stream_tokens_per_second": [response["stats"]["tokens_per_second"] for response in responses]}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
