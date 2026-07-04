import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QUANTIZE = ROOT / ".tools/llama.cpp/build-quant/bin/llama-quantize.exe"
SOURCE = ROOT / "experiments/v1_vocab64k/ananke-v1-vocab64k-MXFP4.gguf"
OUT = Path(__file__).parent / "ananke-v3-vocab64k-head-q8.gguf"


if __name__ == "__main__":
    subprocess.run(
        [
            str(QUANTIZE),
            "--output-tensor-type",
            "Q8_0",
            "--token-embedding-type",
            "Q8_0",
            str(SOURCE),
            str(OUT),
            "COPY",
        ],
        check=True,
    )
