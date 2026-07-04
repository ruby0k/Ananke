# V17 — frozen MTP heads

Rejected before training: the installed LM Studio/llama.cpp runtime does not implement an MTP graph for GPT-OSS. Its load test rejects V11 because no supported bundled MTP head exists. Training a head that the target runtime cannot execute would not produce a testable model.
