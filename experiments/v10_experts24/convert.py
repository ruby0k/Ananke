import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v1_vocab64k import convert

convert.OUT = Path(__file__).parent / "ananke-v10-experts24-MXFP4.gguf"
convert.VERSION = "V10"
convert.__file__ = str(Path(__file__))

if __name__ == "__main__":
    convert.main()
