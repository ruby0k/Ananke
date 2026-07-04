import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / ".tools/llama.cpp"
CONVERTER_BASE = LLAMA / "conversion/base.py"
TOKENIZER_HASH = "f882ccd78951d01935c3807a71f3ae789fea20804219f06cd358c36010b584bb"
OUT = Path(__file__).parent / "ananke-v1-vocab64k-MXFP4.gguf"
VERSION = "V1"


def main() -> None:
    source = CONVERTER_BASE.read_text(encoding="utf-8")
    registration = f'        if chkhsh == "{TOKENIZER_HASH}":\n            res = "gpt-4o"  # Ananke {VERSION} prunes vocabulary only.\n'
    if TOKENIZER_HASH not in source:
        marker = "        if res is None:\n"
        source = source.replace(marker, registration + "\n" + marker, 1)
        if TOKENIZER_HASH not in source:
            raise RuntimeError("llama.cpp tokenizer-hash insertion point changed")
        CONVERTER_BASE.write_text(source, encoding="utf-8")

    python = LLAMA / ".venv/Scripts/python.exe"
    subprocess.run(
        [
            str(python),
            str(LLAMA / "convert_hf_to_gguf.py"),
            str(Path(__file__).parent / "hf"),
            "--outfile",
            str(OUT),
            "--outtype",
            "auto",
        ],
        check=True,
    )
    print(OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
