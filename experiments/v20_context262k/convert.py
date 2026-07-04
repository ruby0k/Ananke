import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / ".tools/llama.cpp"
OUT = Path(__file__).parent / "ananke-v20-top2-context262k-MXFP4.gguf"

if __name__ == "__main__":
    subprocess.run([
        str(LLAMA / ".venv/Scripts/python.exe"), str(LLAMA / "convert_hf_to_gguf.py"),
        str(Path(__file__).parent / "hf"), "--outfile", str(OUT), "--outtype", "auto",
    ], check=True)
    print(OUT)
