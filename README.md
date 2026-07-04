# Ananke

GPT-OSS-20B weight-mapping experiments, with a reproducible LM Studio baseline for later
comparison against custom architectures.

The installed `openai/gpt-oss-20b` is already MXFP4 (4-bit). The measured sweet spot on this
RTX 5050 Laptop GPU is 60% GPU offload; higher values spill VRAM and become slower.

```powershell
uv run python baseline.py
```

The runner starts LM Studio's server, reloads the model with a 4096-token context and one
prediction slot, then writes output, quantization, runtime, TTFT, and generation speed to
`results/baseline.json`.

## Custom architecture

Verify that every official checkpoint tensor maps exactly onto the Ananke topology without
allocating the 13.8 GB model:

```powershell
uv run python architecture.py
```

Actually materialize all three weight shards on CPU:

```powershell
lms unload ananke-baseline
uv run python architecture.py --materialize
```

The packed MXFP4 expert tensors remain packed. Their execution kernel is the intentional next
architecture boundary; attention, routing, normalization, embeddings, and tensor ownership are
already represented directly in Ananke.

Current weight-compatible controls:

- fixed top-2 routing (default),
- adaptive top-2/top-4 routing via router confidence,
- prompt-calibrated layer masks that never skip consecutive blocks and remain fixed during cached generation.

```powershell
uv run python architecture.py --expert-mode adaptive --expert-confidence 0.8 --layer-skip-threshold 0.05
```

## Experiments

Run the first controlled architectural ablation: one, two, and four active experts with fixed
prompts and deterministic decoding.

```powershell
uv run python experiment_experts.py
```

Per-run JSON and a comparison table are written under `experiments/expert_sweep/`.

Benchmark the original safetensors loaded directly through PyTorch/Transformers (not LM Studio):

```powershell
lms unload --all
uv run python benchmark_weights.py
```

Generate through the executable Ananke subclass with mapped weights and canonical MXFP4 kernels:

```powershell
lms unload --all
uv run python generate.py --experts 2
```

Build a push-ready Hugging Face repository under `dist/ananke-gpt-oss-20b-top2/`:

```powershell
uv run python export_hf.py
```
