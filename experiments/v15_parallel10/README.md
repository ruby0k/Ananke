# V15 — ten concurrent streams

Unmodified V11 top-2 with ten LM Studio slots. Result: **98.11 aggregate tok/s end-to-end** and **121.3 tok/s summed steady decode**, at **12.13 tok/s per stream**. The GPU is saturated; ten slots do not materially beat eight.
