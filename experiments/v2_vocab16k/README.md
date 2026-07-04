# V2 — 16K vocabulary

Single change from V1: vocabulary reduced from 64,000 to 16,000 tokens. Token selection, top-2 experts, corpus, and tests remain unchanged.

```powershell
uv run python experiments/v2_vocab16k/build.py
uv run python experiments/v2_vocab16k/convert.py
uv run python experiments/v2_vocab16k/test.py
```

## Result — rejected

| model | GGUF size | corpus token inflation | pass rate | tok/s |
|---|---:|---:|---:|---:|
| top-2 control | 12.845 GiB | — | 83% | 11.54 |
| V1 vocab64k | 11.366 GiB | 0.076% | 83% | 29.40 |
| V2 vocab16k | 10.848 GiB | 14.4% | 17% | 34.03 |

V2 fails the quality gate: five of six prompts fail and several outputs contain malformed Harmony/channel fragments. The small speed gain over V1 does not justify the severe regression. Keep V1; test a midpoint as a new experiment.
