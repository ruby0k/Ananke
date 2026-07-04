# V6 — n-gram speculative decoding

V3 runs through the official standalone llama.cpp CUDA server with speculation disabled versus `ngram-simple`. No draft model is loaded.

```powershell
uv run python experiments/v6_ngram/test.py
```

## Result — rejected as default

| mode | mean tok/s | aggregate acceptance |
|---|---:|---:|
| no speculation | 23.52 | — |
| ngram-simple | 20.64 | 31.9% |

Default n-gram speculation is 12.3% slower across the mixed benchmark. It helps strongly repetitive code (`25.82 -> 35.96 tok/s`, +39.3%) but hurts ordinary and lightly repetitive prompts. Keep it disabled by default; it can be useful as an opt-in coding/repetition mode.
