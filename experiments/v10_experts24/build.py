import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/v7_hidden2560/hf"
OUT = Path(__file__).parent / "hf"
EXPERTS = 24
VERSION = "v10"
CALIBRATION = Path(__file__).parent / "calibration.json"


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    selected = [torch.tensor(row).topk(EXPERTS).indices for row in calibration["usage"]]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "generation_config.json"):
        shutil.copy2(SOURCE / name, OUT / name)
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    config["num_local_experts"] = EXPERTS
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    index = json.loads((SOURCE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    total_size = 0
    for shard in sorted(set(index["weight_map"].values())):
        state = load_file(SOURCE / shard, device="cpu")
        for name, tensor in list(state.items()):
            match = re.match(r"model\.layers\.(\d+)\.mlp\.(router|experts)\.", name)
            if match:
                state[name] = tensor.index_select(0, selected[int(match.group(1))])
        with safe_open(SOURCE / shard, framework="pt") as source:
            metadata = source.metadata()
        save_file(state, OUT / shard, metadata=metadata)
        total_size += sum(t.numel() * t.element_size() for t in state.values())
        print(f"wrote {shard}", flush=True)
    index["metadata"]["total_size"] = total_size
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (Path(__file__).parent / "build_report.json").write_text(json.dumps({
        "version": VERSION, "change": f"stored experts 32 -> {EXPERTS}", "calibration_tokens": calibration["tokens"],
        "selected_experts": [row.tolist() for row in selected], "hf_tensor_bytes": total_size,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
