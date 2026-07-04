# V11 — 23 stored experts

Strict sub-8 decimal GB candidate derived from V7. Each layer keeps its 23 highest-usage experts from V10's 512-token router calibration; top-2 routing is unchanged.

## Result — sub-8 GB candidate

| model | size GB | size GiB | smoke | tok/s | long retrieval |
|---|---:|---:|---:|---:|---|
| V7 | 10.542 | 9.819 | 4/6 | 37.64 | pass at 128 tokens |
| V11 | 7.996 | 7.447 | 4/6 | 37.06 | pass at 512 tokens |

V11 preserves the tested factual, arithmetic, decimal, and science behavior while meeting the strict file-size target. It is 1.6% slower than V7 in the smoke run and needs a larger reasoning budget for long retrieval. Treat it as the best sub-8 GB candidate, not a quality-equivalent replacement, until broader evaluation or distillation.
