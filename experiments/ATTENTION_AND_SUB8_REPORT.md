# Attention and sub-8 GB report

Direct GQA/MQA and sliding-window surgery does not produce a usable or materially smaller model.

| model | size GB | smoke | tok/s | long retrieval | decision |
|---|---:|---:|---:|---|---|
| V7 baseline | 10.542 | 4/6 | 37.64 | pass at 128 | reference |
| GQA-4 | ~10.48 | 0/6 | 36.25 | fail | reject |
| MQA | ~10.43 | 0/6 | 35.44 | fail | reject |
| window 256 | 10.542 | 3/6 | 33.97 | fail | reject |
| window 64 | 10.542 | 3/6 | 33.34 | pass | reject |
| V10 experts24 | 8.279 | 4/6 | 35.92 | pass at 512 | sub-8 GiB only |
| V11 experts23 | 7.996 | 4/6 | 37.06 | pass at 512 | best strict sub-8 GB candidate |

Six-full-layer attention cannot be represented by the standard GPT-OSS GGUF runtime because it does not store a per-layer attention map. V11 reaches the target by calibrated per-layer expert pruning while keeping top-2 routing, GQA-8, the trained 128-token sliding window, and 12 full-attention layers.
