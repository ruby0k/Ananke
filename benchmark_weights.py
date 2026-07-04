import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from accelerate import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_utils


MODEL = "openai/gpt-oss-20b"


def enable_moe_offload() -> None:
    # ponytail: remove when Transformers forwards preload_module_classes to Accelerate.
    def dispatch(model, _, device_map, offload_folder, offload_index, offload_buffers):
        dispatch_model(
            model,
            device_map=device_map,
            offload_dir=offload_folder,
            offload_index=offload_index,
            offload_buffers=offload_buffers,
            skip_keys=model._skip_keys_device_placement,
            preload_module_classes=["GptOssMLP"],
        )

    modeling_utils.accelerate_dispatch = dispatch


def set_active_experts(model, count: int) -> None:
    routers = [module for module in model.modules() if module.__class__.__name__ == "GptOssTopKRouter"]
    assert len(routers) == model.config.num_hidden_layers
    for router in routers:
        router.top_k = count


def generate(model, inputs, tokens: int) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - started, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark directly loaded GPT-OSS safetensors")
    parser.add_argument("--experts", default="2,4")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("experiments/direct_weights.json"))
    args = parser.parse_args()
    expert_counts = [int(value) for value in args.experts.split(",")]

    enable_moe_offload()
    offload = Path(".offload")
    offload.mkdir(exist_ok=True)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="auto",
        offload_folder=offload,
    )
    load_seconds = time.perf_counter() - load_started
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with the numbers one through ten."}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    ).to(model.device)
    input_tokens = inputs["input_ids"].shape[-1]

    rows = []
    for count in expert_counts:
        set_active_experts(model, count)
        generate(model, inputs, 1)  # compile/warm the kernel shape outside the measurement
        torch.cuda.reset_peak_memory_stats()
        seconds, output = generate(model, inputs, args.tokens)
        output_ids = output[0, input_tokens:]
        rows.append(
            {
                "active_experts": count,
                "tokens": len(output_ids),
                "seconds": seconds,
                "tokens_per_second": len(output_ids) / seconds,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                "output": tokenizer.decode(output_ids, skip_special_tokens=False),
            }
        )
        print(f"experts={count}: {rows[-1]['tokens_per_second']:.3f} tok/s")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "transformers_direct_safetensors",
        "model": MODEL,
        "load_seconds": load_seconds,
        "device_map": {name: str(device) for name, device in model.hf_device_map.items()},
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
