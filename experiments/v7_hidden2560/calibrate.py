import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark_weights import enable_moe_offload


SOURCE = ROOT / "experiments/v1_vocab64k/hf"
TEXT = """Explain why the daytime sky is blue in one sentence.
Write a Python function that checks whether an integer is even.
Calculate 17 multiplied by 19 and compare 9.11 with 9.9.
Summarize how photosynthesis converts light energy into chemical energy.
Describe a cautious plan for debugging a CUDA out-of-memory error.
The rain in Spain falls mainly on the plain, and the train returns again.
"""


def main() -> None:
    enable_moe_offload()
    offload = ROOT / ".offload/v7-calibration"
    offload.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(SOURCE)
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE, torch_dtype="auto", device_map="auto", offload_folder=offload
    )
    inputs = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.inference_mode():
        hidden_states = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
    scores = sum(state.float().square().mean(dim=(0, 1)).cpu() for state in hidden_states)
    groups = scores.reshape(90, 32).mean(dim=1)
    report = {
        "tokens": inputs["input_ids"].shape[1],
        "channel_scores": scores.tolist(),
        "group_scores": groups.tolist(),
    }
    (Path(__file__).parent / "calibration.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    print(f"calibrated {report['tokens']} tokens across {len(hidden_states)} residual states")


if __name__ == "__main__":
    main()
