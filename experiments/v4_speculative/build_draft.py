import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUANTIZE = ROOT / ".tools/llama.cpp/build-quant/bin/llama-quantize.exe"
SOURCE = ROOT / "experiments/v3_head_q8/ananke-v3-vocab64k-head-q8.gguf"
OUT = Path(__file__).parent / "ananke-v4-draft-8l.gguf"
PRUNED = "1,2,4,5,6,8,9,11,12,14,15,17,18,19,21,22"


if __name__ == "__main__":
    subprocess.run([str(QUANTIZE), "--prune-layers", PRUNED, str(SOURCE), str(OUT), "COPY"], check=True)
