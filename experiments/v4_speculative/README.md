# V4 — speculative decoding

V3 is the target. The draft retains eight evenly distributed V3 layers and the exact same 64K tokenizer.

```powershell
uv run python experiments/v4_speculative/build_draft.py
uv run python experiments/v4_speculative/test.py
```

## Result — rejected

| mode | tok/s | draft acceptance |
|---|---:|---:|
| V3 without speculation | 25.29 | — |
| V3 + 8-layer draft | 5.69 | 0.0% |

The directly pruned draft proposes no tokens accepted by V3, making generation 4.4× slower. LM Studio also crashed on the second consecutive speculative request during the broader run. A useful draft requires distillation/training; layer deletion alone is insufficient.
