# V1 — 64K vocabulary

Single change from the Ananke top-2 control: vocabulary reduced from 201,088 to 64,000 tokens.

- Selection corpus: Calliope `v3_general_gpt2` (general English, code, and knowledge text).
- Protected: all byte tokens, all 21 Harmony/special tokens, and BPE merge-parent closure.
- Mapping: retained embedding and LM-head rows are copied exactly; all other tensors are unchanged.
- Acceptance: generation works, effective characters/second improves, and fixed-prompt behavior remains usable.

```powershell
uv run python experiments/v1_vocab64k/build.py
uv run python experiments/v1_vocab64k/convert.py
uv run python experiments/v1_vocab64k/test.py
```

## Result

| model | GGUF size | pass rate | tok/s | visible chars/s |
|---|---:|---:|---:|---:|
| top-2 control | 12.845 GiB | 83% | 15.82 | 8.10 |
| V1 vocab64k | 11.366 GiB | 83% | 35.59 | 20.54 |

V1 retains `64,000 / 201,088` vocabulary rows and adds only 0.076% tokens on the sampled corpus. It passes the initial smoke gate; broader quality evaluation is still required before treating it as a replacement for the control.
