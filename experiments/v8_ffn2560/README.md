# V8 — expert FFN size 2560

Derived from V7. Expert intermediate width is reduced from 2880 to 2560 using MXFP4 scale saliency in 32-neuron groups. Residual width remains 2560.

```powershell
uv run python experiments/v8_ffn2560/build.py
uv run python experiments/v8_ffn2560/convert.py
uv run python experiments/v8_ffn2560/quantize.py
uv run python experiments/v8_ffn2560/test.py
```

## Result — rejected

| model | GGUF size | pass rate | tok/s | visible chars/s |
|---|---:|---:|---:|---:|
| V7 FFN2880 | 9.819 GiB | 67% | 34.17 | 16.95 |
| V8 FFN2560 | 8.884 GiB | 50% | 37.61 | 15.41 |

V8 saves 0.935 GiB and is 10.1% faster, but introduces a real arithmetic regression (`17 x 19 -> 287`) and retains V7's verbose-reasoning failures. Do not merge this cut into V7 without distillation. A less aggressive FFN width such as 2720 is the remaining no-training option.
