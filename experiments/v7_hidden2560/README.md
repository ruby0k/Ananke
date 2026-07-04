# V7 — hidden size 2560

Activation-aware, group-aligned residual-width pruning from 2880 to 2560. Derived from V1/V3 with the 64K tokenizer and top-2 routing unchanged.

```powershell
uv run python experiments/v7_hidden2560/calibrate.py
uv run python experiments/v7_hidden2560/build.py
uv run python experiments/v7_hidden2560/convert.py
uv run python experiments/v7_hidden2560/quantize.py
uv run python experiments/v7_hidden2560/test.py
```

## Result — promising, needs recovery training

| model | GGUF size | fixed-cap pass rate | tok/s | visible chars/s |
|---|---:|---:|---:|---:|
| V3 hidden2880 | 11.044 GiB | 83% | 18.51 | 12.49 |
| V7 hidden2560 | 9.819 GiB | 67% | 37.59 | 18.43 |

V7 saves 1.225 GiB and is 2.03x faster in the same controlled run. It retains the tested factual, arithmetic, decimal, and science knowledge. The additional failure is the code prompt exhausting the 256-token cap during verbose reasoning; at a 512-token cap it returns the correct implementation at 37.32 tok/s after 447 reasoning tokens. Exact-instruction framing remains broken in both families.

Do not replace V3 yet. The width cut is recoverable, but V7 needs teacher distillation or reasoning-length calibration before acceptance.
