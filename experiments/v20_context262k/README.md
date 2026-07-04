# V20 — top-2, 262K context

Base GPT-OSS-20B weights with top-2 routing and the native context doubled from 131,072 to 262,144 tokens. YaRN keeps its 4,096-token original window and increases its scaling factor from 32 to 64.

No weights are changed. This is an extrapolation experiment and must pass retrieval beyond 131K before it is accepted.

## Initial result

- GGUF metadata: 262,144 context, YaRN factor 64, top-2
- Full-window LM Studio load: pass
- Short arithmetic generation: pass (`323`)
- Speed with the full window allocated: **7.47 tok/s**
- VRAM: **7,549 MiB used**, 362 MiB free
- Retrieval beyond 131K: not yet tested
