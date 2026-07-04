# V10 — 24 stored experts

Sub-8 GiB candidate derived from V7. Each layer keeps its 24 most-used experts from a 512-token Calliope router calibration; top-2 routing is unchanged.

## Result

- Size: 8.279 GB / 7.710 GiB
- Smoke: 4/6, matching V7
- Speed: 35.92 tok/s
- 3,174-token retrieval: succeeds with a 512-token generation budget, but exhausts 128 tokens

V10 meets a binary 8 GiB target but not a strict 8.000 decimal GB target.
