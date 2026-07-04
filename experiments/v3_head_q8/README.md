# V3 — Q8 embedding and output head

Single change from V1: `token_embd.weight` and `output.weight` are quantized from BF16 to Q8_0. The other 457 tensors, 64K tokenizer, and top-2 routing remain unchanged.

```powershell
uv run python experiments/v3_head_q8/quantize.py
uv run python experiments/v3_head_q8/test.py
```

## Result — accepted

| model | GGUF size | pass rate | tok/s | visible chars/s |
|---|---:|---:|---:|---:|
| V1 vocab64k | 11.366 GiB | 83% | 17.76 | 14.93 |
| V3 head Q8 | 11.044 GiB | 83% | 34.21 | 20.53 |

V3 changes exactly two of 459 tensors and saves 329.6 MiB. In the same controlled run it is 1.93× faster than V1 with the same smoke-test score.
