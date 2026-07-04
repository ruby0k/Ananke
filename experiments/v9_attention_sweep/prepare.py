import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/v7_hidden2560/hf"
LLAMA = ROOT / ".tools/llama.cpp"
QUANTIZE = ROOT / ".tools/llama.cpp/build-quant/bin/llama-quantize.exe"
VARIANTS = {
    "v9a_gqa4": {"num_key_value_heads": 4},
    "v9b_mqa": {"num_key_value_heads": 1},
    "v9c_full6": {"layer_types": ["full_attention" if i % 4 == 3 else "sliding_attention" for i in range(24)]},
    "v9d_window256": {"sliding_window": 256},
    "v9e_window64": {"sliding_window": 64},
}


def reduce_kv(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.reshape(heads, 8 // heads, 64, tensor.shape[1]).mean(dim=1).reshape(heads * 64, tensor.shape[1])
    return tensor.reshape(heads, 8 // heads, 64).mean(dim=1).reshape(heads * 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    args = parser.parse_args()
    variant = args.variant
    directory = Path(__file__).parent / variant
    hf = directory / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    config.update(VARIANTS[variant])
    if variant.startswith("v9d_") or variant.startswith("v9e_"):
        hf.rmdir()
        source = ROOT / "experiments/v7_hidden2560/ananke-v7-hidden2560-head-q8.gguf"
        final = directory / f"{variant}-head-q8.gguf"
        subprocess.run([str(QUANTIZE), "--override-kv",
                        f"gpt-oss.attention.sliding_window=int:{config['sliding_window']}",
                        str(source), str(final), "COPY"], check=True)
        print(final)
        return
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "generation_config.json"):
        shutil.copy2(SOURCE / name, hf / name)
    (hf / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    index = json.loads((SOURCE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    kv_heads = config["num_key_value_heads"]
    total_size = index["metadata"]["total_size"] if kv_heads == 8 else 0
    for shard in sorted(set(index["weight_map"].values())):
        destination = hf / shard
        if kv_heads == 8:
            if destination.exists():
                destination.unlink()
            os.link(SOURCE / shard, destination)
        else:
            state = load_file(SOURCE / shard, device="cpu")
            state = {name: reduce_kv(tensor, kv_heads) if name.endswith(("k_proj.weight", "k_proj.bias",
                                                                          "v_proj.weight", "v_proj.bias")) else tensor
                     for name, tensor in state.items()}
            with safe_open(SOURCE / shard, framework="pt") as source:
                metadata = source.metadata()
            save_file(state, destination, metadata=metadata)
            total_size += sum(t.numel() * t.element_size() for t in state.values())
    index["metadata"]["total_size"] = int(total_size)
    (hf / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    raw = directory / f"{variant}-MXFP4.gguf"
    final = directory / f"{variant}-head-q8.gguf"
    subprocess.run([str(LLAMA / ".venv/Scripts/python.exe"), str(LLAMA / "convert_hf_to_gguf.py"), str(hf),
                    "--outfile", str(raw), "--outtype", "auto"], check=True)
    subprocess.run([str(QUANTIZE), "--output-tensor-type", "Q8_0", "--token-embedding-type", "Q8_0",
                    str(raw), str(final), "COPY"], check=True)
    print(final)


if __name__ == "__main__":
    main()
