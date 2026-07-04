import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from architecture import snapshot_path

OUT = Path(__file__).parent / "hf"
FILES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "chat_template.jinja", "generation_config.json"}


def link(source: Path, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    source = snapshot_path()
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["num_experts_per_tok"] = config["experts_per_token"] = 2
    config["max_position_embeddings"] = 262144
    config["rope_scaling"]["factor"] = 64.0
    config["ananke_version"] = "v20-context262k"
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for name in FILES | {"model.safetensors.index.json"} | set(index["weight_map"].values()):
        link(source / name, OUT / name)
    report = {"version": "v20", "base": "openai/gpt-oss-20b", "active_experts": 2,
              "context_length": 262144, "rope_type": "yarn", "rope_factor": 64.0,
              "original_context_length": 4096, "weight_change": False}
    (Path(__file__).parent / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
