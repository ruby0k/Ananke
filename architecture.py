import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn


MODEL_CACHE = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots"


@dataclass(frozen=True)
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_local_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    attention_bias: bool
    layer_types: list[str]
    sliding_window: int

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    def __init__(self, config: Config, device: str):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.empty(config.hidden_size, dtype=torch.bfloat16, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, config: Config, device: str):
        super().__init__()
        dtype = torch.bfloat16
        q = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q, config.attention_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, kv, config.attention_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, kv, config.attention_bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(q, config.hidden_size, config.attention_bias, device=device, dtype=dtype)
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads, dtype=dtype, device=device))


class Router(nn.Module):
    def __init__(self, config: Config, device: str):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size, dtype=torch.bfloat16, device=device)
        )
        self.bias = nn.Parameter(torch.empty(config.num_local_experts, dtype=torch.bfloat16, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores, indices = torch.topk(nn.functional.linear(x, self.weight, self.bias), self.top_k, dim=-1)
        return nn.functional.softmax(scores, dim=-1), indices


class PackedExperts(nn.Module):
    def __init__(self, config: Config, device: str):
        super().__init__()
        experts, hidden, intermediate = config.num_local_experts, config.hidden_size, config.intermediate_size
        input_blocks, expert_blocks = hidden // 32, intermediate // 32
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(experts, 2 * intermediate, dtype=torch.bfloat16, device=device)
        )
        self.register_buffer(
            "gate_up_proj_blocks",
            torch.empty(experts, 2 * intermediate, input_blocks, 16, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "gate_up_proj_scales",
            torch.empty(experts, 2 * intermediate, input_blocks, dtype=torch.uint8, device=device),
        )
        self.down_proj_bias = nn.Parameter(torch.empty(experts, hidden, dtype=torch.bfloat16, device=device))
        self.register_buffer(
            "down_proj_blocks", torch.empty(experts, hidden, expert_blocks, 16, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "down_proj_scales", torch.empty(experts, hidden, expert_blocks, dtype=torch.uint8, device=device)
        )

    def forward(self, *_):
        raise NotImplementedError("MXFP4 expert execution is the custom-kernel boundary")


class MLP(nn.Module):
    def __init__(self, config: Config, device: str):
        super().__init__()
        self.router = Router(config, device)
        self.experts = PackedExperts(config, device)


class Block(nn.Module):
    def __init__(self, config: Config, device: str):
        super().__init__()
        self.input_layernorm = RMSNorm(config, device)
        self.self_attn = Attention(config, device)
        self.mlp = MLP(config, device)
        self.post_attention_layernorm = RMSNorm(config, device)


class Backbone(nn.Module):
    def __init__(self, config: Config, device: str):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList([Block(config, device) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config, device)


class Ananke(nn.Module):
    def __init__(self, config: Config, device: str = "meta"):
        super().__init__()
        self.config = config
        self.model = Backbone(config, device)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=torch.bfloat16
        )


def snapshot_path(path: Path | None = None) -> Path:
    if path:
        return path
    snapshots = list(MODEL_CACHE.glob("*"))
    if not snapshots:
        raise FileNotFoundError("GPT-OSS-20B is not present in the Hugging Face cache")
    return snapshots[0]


def verify(model: Ananke, snapshot: Path) -> None:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    expected = model.state_dict()
    errors = []
    for name in sorted(set(expected) | set(index)):
        if name not in expected or name not in index:
            errors.append(f"{name}: {'unexpected' if name not in expected else 'missing'}")
            continue
        with safe_open(snapshot / index[name], framework="pt", device="cpu") as shard:
            tensor = shard.get_slice(name)
            if list(expected[name].shape) != tensor.get_shape():
                errors.append(f"{name}: {list(expected[name].shape)} != {tensor.get_shape()}")
            dtype = {torch.bfloat16: "BF16", torch.uint8: "U8"}[expected[name].dtype]
            if dtype != tensor.get_dtype():
                errors.append(f"{name}: {dtype} != {tensor.get_dtype()}")
    if errors:
        raise ValueError("Checkpoint mapping failed:\n" + "\n".join(errors))
    print(f"Verified {len(index)} tensors: names, shapes, and dtypes match exactly.")


def materialize(model: Ananke, snapshot: Path, device: str) -> None:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    loaded = set()
    for shard_path in sorted({snapshot / shard for shard in index.values()}):
        state = load_file(shard_path, device=device)
        model.load_state_dict(state, strict=False, assign=True)
        loaded.update(state)
    if loaded != set(model.state_dict()):
        raise ValueError(f"Loaded {len(loaded)} of {len(model.state_dict())} tensors")
    if any(t.is_meta for t in model.state_dict().values()):
        raise ValueError("Materialization left tensors on the meta device")
    print(f"Mapped {len(loaded)} checkpoint tensors onto the custom architecture on {device}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map GPT-OSS-20B weights into Ananke")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    snapshot = snapshot_path(args.checkpoint)
    model = Ananke(Config.load(snapshot / "config.json"))
    verify(model, snapshot)
    if args.materialize:
        materialize(model, snapshot, args.device)


if __name__ == "__main__":
    main()
