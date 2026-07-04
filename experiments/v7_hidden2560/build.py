import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/v1_vocab64k/hf"
OUT = Path(__file__).parent / "hf"
WIDTH = 2560


def prune(name: str, tensor: torch.Tensor, channels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return tensor.index_select(1, channels)
    if name == "model.norm.weight" or name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
        return tensor.index_select(0, channels)
    if name.endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                      "mlp.router.weight")):
        return tensor.index_select(1, channels)
    if name.endswith("self_attn.o_proj.weight"):
        return tensor.index_select(0, channels)
    if name.endswith("self_attn.o_proj.bias"):
        return tensor.index_select(0, channels)
    if name.endswith(("experts.down_proj_bias", "experts.down_proj_blocks", "experts.down_proj_scales")):
        return tensor.index_select(1, channels)
    if name.endswith(("experts.gate_up_proj_blocks", "experts.gate_up_proj_scales")):
        return tensor.index_select(2, groups)
    return tensor


def main() -> None:
    calibration = json.loads((Path(__file__).parent / "calibration.json").read_text(encoding="utf-8"))
    scores = torch.tensor(calibration["group_scores"])
    kept_groups = torch.topk(scores, WIDTH // 32).indices.sort().values
    kept_channels = torch.cat([torch.arange(group * 32, group * 32 + 32) for group in kept_groups])
    OUT.mkdir(parents=True, exist_ok=True)

    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "generation_config.json"):
        shutil.copy2(SOURCE / name, OUT / name)
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    config["hidden_size"] = WIDTH
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    index = json.loads((SOURCE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    total_size = 0
    for shard in sorted(set(index["weight_map"].values())):
        state = load_file(SOURCE / shard, device="cpu")
        state = {name: prune(name, tensor, kept_channels, kept_groups) for name, tensor in state.items()}
        with safe_open(SOURCE / shard, framework="pt") as source:
            metadata = source.metadata()
        save_file(state, OUT / shard, metadata=metadata)
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        print(f"wrote {shard}", flush=True)
    index["metadata"]["total_size"] = total_size
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    report = {
        "version": "v7",
        "change": "hidden_size 2880 -> 2560",
        "calibration_tokens": calibration["tokens"],
        "kept_groups": kept_groups.tolist(),
        "removed_groups": sorted(set(range(90)) - set(kept_groups.tolist())),
        "hf_tensor_bytes": total_size,
    }
    (Path(__file__).parent / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert load_file(OUT / index["weight_map"]["lm_head.weight"])["lm_head.weight"].shape == (64000, WIDTH)
    print(f"built V7 with {WIDTH} hidden channels")


if __name__ == "__main__":
    main()
