# V5 — LFM 2.5 draft

V3 target paired with local `liquid/lfm2.5-1.2b` as the speculative draft.

```powershell
uv run python experiments/v5_lfm_draft/test.py
```

## Result — incompatible

LM Studio rejected the pair before generation:

```text
draft model bos tokens must match target model
target BOS: 63979; draft BOS: 1
```

LFM 2.5 does not share Ananke V3's tokenizer or token IDs, so it cannot be used directly as its speculative draft. Making it compatible would require training LFM with Ananke's tokenizer/output vocabulary; metadata remapping alone would destroy its token semantics.
