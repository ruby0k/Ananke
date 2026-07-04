import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from architecture import ExecutableAnanke
from baseline import MODEL, post, run
from benchmark_weights import enable_moe_offload, set_active_experts


PROMPTS = {
    "science": "Explain in one sentence why the daytime sky looks blue.",
    "arithmetic": "Which is larger, 9.11 or 9.9? Answer with only the larger value.",
    "code": "Write a one-line Python function named is_even that returns whether n is even.",
}


def harmony_text(raw: str) -> tuple[str | None, str]:
    analysis_marker = "<|channel|>analysis<|message|>"
    final_marker = "<|channel|>final<|message|>"
    reasoning = None
    if analysis_marker in raw:
        reasoning = raw.split(analysis_marker, 1)[1].split("<|end|>", 1)[0].strip()
    output = raw.split(final_marker, 1)[1].split("<|return|>", 1)[0].strip() if final_marker in raw else raw
    return reasoning, output


def custom_samples(tokenizer) -> list[dict]:
    enable_moe_offload()
    Path(".offload").mkdir(exist_ok=True)
    model = ExecutableAnanke.from_pretrained(
        MODEL, torch_dtype="auto", device_map="auto", offload_folder=".offload"
    )
    set_active_experts(model, 2)
    rows = []
    for name, prompt in PROMPTS.items():
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="low",
        ).to(model.device)
        input_tokens = inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        output_ids = generated[0, input_tokens:]
        raw = tokenizer.decode(output_ids, skip_special_tokens=False)
        reasoning, output = harmony_text(raw)
        rows.append(
            {
                "id": name,
                "prompt": prompt,
                "output": output,
                "reasoning": reasoning,
                "raw_output": raw,
                "tokens": len(output_ids),
                "tokens_per_second": len(output_ids) / seconds,
            }
        )
        print(f"custom {name}: {rows[-1]['tokens_per_second']:.3f} tok/s", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def lm_studio_samples() -> list[dict]:
    run("lms", "server", "start", check=False)
    run("lms", "unload", MODEL, check=False)
    loaded = post(
        "/api/v1/models/load",
        {
            "model": MODEL,
            "context_length": 4096,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": True,
            "num_experts": 2,
        },
    )
    rows = []
    for name, prompt in PROMPTS.items():
        response = post(
            "/api/v0/chat/completions",
            {
                "model": loaded["instance_id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 48,
                "temperature": 0,
                "seed": 0,
                "reasoning_effort": "low",
            },
        )
        message = response["choices"][0]["message"]
        rows.append(
            {
                "id": name,
                "prompt": prompt,
                "output": message["content"],
                "reasoning": message.get("reasoning"),
                "tokens": response["usage"]["completion_tokens"],
                "tokens_per_second": response["stats"]["tokens_per_second"],
            }
        )
        print(f"lm-studio {name}: {rows[-1]['tokens_per_second']:.3f} tok/s", flush=True)
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run("lms", "unload", "--all", check=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    custom = custom_samples(tokenizer)
    lm_studio = lm_studio_samples()
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"weights": MODEL, "active_experts": 2, "max_new_tokens": 48, "decoding": "greedy"},
        "custom": custom,
        "lm_studio": lm_studio,
    }
    out = Path("experiments/sample_comparison.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Custom architecture vs LM Studio", ""]
    for ours, lm in zip(custom, lm_studio):
        lines += [
            f"## {ours['id']}",
            "",
            f"**Prompt:** {ours['prompt']}",
            "",
            f"**Custom ({ours['tokens_per_second']:.3f} tok/s):** {ours['output']}",
            "",
            f"**LM Studio ({lm['tokens_per_second']:.3f} tok/s):** {lm['output']}",
            "",
        ]
    Path("experiments/sample_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved experiments/sample_comparison.{json,md}")


if __name__ == "__main__":
    main()
