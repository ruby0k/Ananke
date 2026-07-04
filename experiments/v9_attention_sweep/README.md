# V9 — attention sweep

Isolated V7-derived tests for GQA-4, MQA, six full-attention layers, sliding window 256, and sliding window 64. Generated HF/GGUF staging is removed after each test because the five models exceed available disk space.

## Results

| variant | size GiB | smoke | tok/s | 3,174-token retrieval | TTFT s | result |
|---|---:|---:|---:|---:|---:|---|
| V7 baseline | 9.819 | 67% | 37.64 | pass | 2.994 | keep |
| GQA-4 | 9.757 | 0% | 36.25 | fail | 2.758 | reject |
| MQA | 9.713 | 0% | 35.44 | fail | 2.730 | reject |
| six full layers | 9.819 | — | — | — | — | unsupported by standard GPT-OSS GGUF runtime |
| window 256 | 9.819 | 50% | 33.97 | fail | 2.885 | reject |
| window 64 | 9.819 | 50% | 33.34 | pass | 3.001 | reject |

Direct K/V-head averaging destroys learned attention specialization. Changing the trained 128-token sliding window in either direction also reduces quality and does not reduce model file size. None reaches the sub-8 GiB target.
