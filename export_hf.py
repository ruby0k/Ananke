import hashlib
import json
import shutil
from pathlib import Path

from architecture import snapshot_path


OUT = Path("dist/ananke-gpt-oss-20b-top2")
COPY = {
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
}


def write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    source = snapshot_path()
    OUT.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config.update(
        {
            "architectures": ["AnankeForCausalLM"],
            "model_type": "ananke",
            "num_experts_per_tok": 2,
            "experts_per_token": 2,
            "auto_map": {
                "AutoConfig": "configuration_ananke.AnankeConfig",
                "AutoModelForCausalLM": "modeling_ananke.AnankeForCausalLM",
            },
        }
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for name in sorted(COPY | set(index["weight_map"].values())):
        src = source / name
        if src.exists():
            dst = OUT / name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)

    write(
        OUT / "configuration_ananke.py",
        '''from transformers import GptOssConfig


class AnankeConfig(GptOssConfig):
    model_type = "ananke"''',
    )
    write(
        OUT / "modeling_ananke.py",
        '''from transformers import GptOssForCausalLM

from .configuration_ananke import AnankeConfig


class AnankeForCausalLM(GptOssForCausalLM):
    config_class = AnankeConfig''',
    )
    write(OUT / "__init__.py", "from .configuration_ananke import AnankeConfig\nfrom .modeling_ananke import AnankeForCausalLM")
    write(OUT / ".gitattributes", "*.safetensors filter=lfs diff=lfs merge=lfs -text")
    write(
        OUT / "README.md",
        '''---
license: apache-2.0
library_name: transformers
base_model: openai/gpt-oss-20b
pipeline_tag: text-generation
tags:
  - gpt-oss
  - mxfp4
  - mixture-of-experts
  - custom_code
---

# Ananke GPT-OSS-20B Top-2

Weight-compatible GPT-OSS-20B architecture experiment using two active experts per token instead
of four. The checkpoint weights and tokenizer come from `openai/gpt-oss-20b`; only routing width
and the custom Hugging Face architecture entrypoint change.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "YOUR_USERNAME/ananke-gpt-oss-20b-top2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)
```

Requires a current PyTorch/Transformers stack plus MXFP4-compatible Triton kernels. The native
checkpoint is approximately 13.8 GB and normally targets at least 16 GB combined accelerator
memory. On the development RTX 5050 Laptop GPU (8 GB), Transformers CPU offloading generated at
roughly 0.3 tok/s; LM Studio's llama.cpp runtime was substantially faster.

This is an architecture experiment, not a newly trained model. Top-2 routing needs broader
quality evaluation before production use.

The repository also includes `ananke-gpt-oss-20b-top2-MXFP4.gguf`, independently converted from
these safetensors with llama.cpp. It contains 459 tensors and declares two active experts:

```powershell
lms import -L --user-repo USERNAME/ananke-gpt-oss-20b-top2 ananke-gpt-oss-20b-top2-MXFP4.gguf
```''',
    )

    manifest = {}
    for path in sorted(file for file in OUT.iterdir() if file.is_file() and file.name != "MANIFEST.json"):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        manifest[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Packed {len(manifest)} files ({sum(row['bytes'] for row in manifest.values()) / 1e9:.2f} GB) in {OUT}")


if __name__ == "__main__":
    main()
