import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / ".tools/llama.cpp/build-quant/bin/llama-quantize.exe"
SOURCE = Path(__file__).parent / "ananke-v10-experts24-MXFP4.gguf"
OUT = Path(__file__).parent / "ananke-v10-experts24-head-q8.gguf"

if __name__ == "__main__":
    subprocess.run([str(Q), "--output-tensor-type", "Q8_0", "--token-embedding-type", "Q8_0",
                    str(SOURCE), str(OUT), "COPY"], check=True)
