# ananke-20b-v1

First Ananke mainline control model, derived from `openai/gpt-oss-20b` without weight surgery.

- 24 layers, hidden size 2880
- 32 stored experts
- **top-4 active experts per token**
- native 201,088-token vocabulary
- original MXFP4 weights and Harmony template

This version establishes the accuracy control for subsequent trained architectural changes.

## Baseline result

The standard deterministic smoke test passes **6/6** at **17.90 tok/s** on the RTX 5050 Laptop GPU with top-4 routing.

```powershell
uv run python models\ananke-20b-v1\build.py
uv run python models\ananke-20b-v1\convert.py
uv run python models\ananke-20b-v1\test.py
```
