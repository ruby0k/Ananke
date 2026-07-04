# V18 — top-1 recovery

Router-only recovery is mathematically unable to repair top-1: softmax over one selected expert is always 1, so the router receives zero gradient. The available MXFP4/GGUF artifact is an inference checkpoint; useful recovery requires trainable adapters or dequantized expert/transformer weights. V12 remains the empirical control at **45.99 tok/s, 0/6 pass**.
