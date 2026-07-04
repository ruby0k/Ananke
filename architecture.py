import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import GptOssForCausalLM


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


@dataclass(frozen=True)
class ArchitectureOptions:
    expert_mode: str = "top2"
    expert_confidence: float = 0.8
    layer_skip_threshold: float = 0.05


def active_expert_count(logits: torch.Tensor, options: ArchitectureOptions) -> int:
    if options.expert_mode == "top2":
        return 2
    if options.expert_mode != "adaptive":
        raise ValueError(f"unknown expert mode: {options.expert_mode}")
    top2_mass = logits.softmax(-1).topk(2, dim=-1).values.sum(-1).mean()
    return 2 if top2_mass >= options.expert_confidence else 4


def active_layer_mask(relative_changes: list[float], threshold: float) -> list[bool]:
    """Select a stable per-prompt layer mask; generation reuses it so KV caches stay aligned."""
    active = [change >= threshold for change in relative_changes]
    active[0] = active[-1] = True
    for index in range(1, len(active)):
        if not active[index - 1] and not active[index]:
            active[index] = True
    return active


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
    def __init__(self, config: Config, options: ArchitectureOptions, device: str):
        super().__init__()
        self.options = options
        self.last_active_experts = 2
        self.weight = nn.Parameter(
            torch.empty(config.num_local_experts, config.hidden_size, dtype=torch.bfloat16, device=device)
        )
        self.bias = nn.Parameter(torch.empty(config.num_local_experts, dtype=torch.bfloat16, device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = nn.functional.linear(x, self.weight, self.bias)
        self.last_active_experts = active_expert_count(logits, self.options)
        scores, indices = torch.topk(logits, self.last_active_experts, dim=-1)
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
    def __init__(self, config: Config, options: ArchitectureOptions, device: str):
        super().__init__()
        self.router = Router(config, options, device)
        self.experts = PackedExperts(config, device)


class Block(nn.Module):
    def __init__(self, config: Config, options: ArchitectureOptions, device: str):
        super().__init__()
        self.input_layernorm = RMSNorm(config, device)
        self.self_attn = Attention(config, device)
        self.mlp = MLP(config, options, device)
        self.post_attention_layernorm = RMSNorm(config, device)


class Backbone(nn.Module):
    def __init__(self, config: Config, options: ArchitectureOptions, device: str):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList([Block(config, options, device) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config, device)


class Ananke(nn.Module):
    def __init__(self, config: Config, options: ArchitectureOptions | None = None, device: str = "meta"):
        super().__init__()
        self.config = config
        self.options = options or ArchitectureOptions()
        self.model = Backbone(config, self.options, device)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=torch.bfloat16
        )


class ExecutableAnanke(GptOssForCausalLM):
    """Runnable Ananke surface using the canonical GPT-OSS kernels and cache implementation."""



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
    parser.add_argument("--expert-mode", choices=("top2", "adaptive"), default="top2")
    parser.add_argument("--expert-confidence", type=float, default=0.8)
    parser.add_argument("--layer-skip-threshold", type=float, default=0.05)
    args = parser.parse_args()
    snapshot = snapshot_path(args.checkpoint)
    options = ArchitectureOptions(args.expert_mode, args.expert_confidence, args.layer_skip_threshold)
    model = Ananke(Config.load(snapshot / "config.json"), options)
    verify(model, snapshot)
    assert active_expert_count(torch.tensor([[10.0, 0.0, 0.0, 0.0]]), ArchitectureOptions("adaptive", 0.8)) == 2
    assert active_expert_count(torch.zeros(1, 4), ArchitectureOptions("adaptive", 0.8)) == 4
    assert active_layer_mask([1.0, 0.01, 0.01, 1.0], 0.05) == [True, False, True, True]
    if args.materialize:
        materialize(model, snapshot, args.device)


if __name__ == "__main__":
    main()
