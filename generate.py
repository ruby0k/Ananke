import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from architecture import ExecutableAnanke
from benchmark_weights import enable_moe_offload, set_active_experts


MODEL = "openai/gpt-oss-20b"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate with weight-loaded Ananke")
    parser.add_argument("--prompt", default="Reply with exactly ANANKE and nothing else.")
    parser.add_argument("--experts", type=int, choices=(2, 4), default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    enable_moe_offload()
    Path(".offload").mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = ExecutableAnanke.from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="auto",
        offload_folder=".offload",
    )
    set_active_experts(model, args.experts)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    ).to(model.device)
    input_tokens = inputs["input_ids"].shape[-1]

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    output_ids = generated[0, input_tokens:]
    print(tokenizer.decode(output_ids, skip_special_tokens=False))
    print(f"\n{len(output_ids) / seconds:.3f} tok/s ({len(output_ids)} tokens in {seconds:.2f}s)")


if __name__ == "__main__":
    main()
