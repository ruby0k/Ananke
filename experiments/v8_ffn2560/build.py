import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/v7_hidden2560/hf"
OUT = Path(__file__).parent / "hf"
WIDTH = 2560


def choose_groups(index: dict) -> torch.Tensor:
    scores = torch.zeros(90)
    for shard in sorted(set(index["weight_map"].values())):
        state = load_file(SOURCE / shard, device="cpu")
        for name, scales in state.items():
            magnitude = torch.exp2((scales.float() - 127).clamp(-30, 30))
            if name.endswith("experts.down_proj_scales"):
                scores += magnitude.mean(dim=(0, 1))
            elif name.endswith("experts.gate_up_proj_scales"):
                paired = magnitude.reshape(32, 2880, 2, 80)
                scores += paired.reshape(32, 90, 32, 2, 80).mean(dim=(0, 2, 3, 4))
    return torch.topk(scores, WIDTH // 32).indices.sort().values


def prune(name: str, tensor: torch.Tensor, neurons: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    if name.endswith("experts.down_proj_blocks") or name.endswith("experts.down_proj_scales"):
        return tensor.index_select(2, groups)
    if name.endswith("experts.gate_up_proj_bias"):
        paired = torch.stack((neurons * 2, neurons * 2 + 1), dim=1).flatten()
        return tensor.index_select(1, paired)
    if name.endswith("experts.gate_up_proj_blocks") or name.endswith("experts.gate_up_proj_scales"):
        paired = torch.stack((neurons * 2, neurons * 2 + 1), dim=1).flatten()
        return tensor.index_select(1, paired)
    return tensor


def main() -> None:
    index = json.loads((SOURCE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    kept_groups = choose_groups(index)
    kept_neurons = torch.cat([torch.arange(group * 32, group * 32 + 32) for group in kept_groups])
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "generation_config.json"):
        shutil.copy2(SOURCE / name, OUT / name)
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    config["intermediate_size"] = WIDTH
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    total_size = 0
    for shard in sorted(set(index["weight_map"].values())):
        state = load_file(SOURCE / shard, device="cpu")
        state = {name: prune(name, tensor, kept_neurons, kept_groups) for name, tensor in state.items()}
        with safe_open(SOURCE / shard, framework="pt") as source:
            metadata = source.metadata()
        save_file(state, OUT / shard, metadata=metadata)
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        print(f"wrote {shard}", flush=True)
    index["metadata"]["total_size"] = total_size
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    report = {
        "version": "v8",
        "change": "intermediate_size 2880 -> 2560",
        "kept_groups": kept_groups.tolist(),
        "removed_groups": sorted(set(range(90)) - set(kept_groups.tolist())),
        "hf_tensor_bytes": total_size,
    }
    (Path(__file__).parent / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("built V8 with hidden=2560, intermediate=2560")


if __name__ == "__main__":
    main()
