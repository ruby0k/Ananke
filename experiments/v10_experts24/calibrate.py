import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_weights import enable_moe_offload

SOURCE = ROOT / "experiments/v7_hidden2560/hf"
CORPUS = Path(r"D:\Hegemony\Calliope\data\v3_general_gpt2")


def main() -> None:
    enable_moe_offload()
    source_tokenizer = AutoTokenizer.from_pretrained(CORPUS / "tokenizer")
    data = np.memmap(CORPUS / "train.bin", dtype=np.uint16, mode="r")
    text = source_tokenizer.decode(data[:800].tolist())
    tokenizer = AutoTokenizer.from_pretrained(SOURCE)
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE, torch_dtype="auto", device_map="auto", offload_folder=ROOT / ".offload/v10-calibration"
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, output_hidden_states=True)
        routed = []
        index = json.loads((SOURCE / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        for layer, state in enumerate(output.hidden_states[:-1]):
            weight_name = f"model.layers.{layer}.mlp.router.weight"
            bias_name = f"model.layers.{layer}.mlp.router.bias"
            with safe_open(SOURCE / index[weight_name], framework="pt", device="cpu") as source:
                weight, bias = source.get_tensor(weight_name).float(), source.get_tensor(bias_name).float()
            logits = torch.nn.functional.linear(state.reshape(-1, state.shape[-1]).cpu().float(), weight, bias)
            routed.append(logits.topk(2, dim=-1).indices)
    selected = []
    counts = []
    for indices in routed:
        top = indices.flatten()
        usage = torch.bincount(top, minlength=32)
        selected.append(usage.topk(24).indices.sort().values.tolist())
        counts.append(usage.tolist())
    report = {"tokens": inputs["input_ids"].shape[1], "usage": counts, "selected_experts": selected}
    (Path(__file__).parent / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"calibrated {report['tokens']} tokens across {len(selected)} routers")


if __name__ == "__main__":
    main()
