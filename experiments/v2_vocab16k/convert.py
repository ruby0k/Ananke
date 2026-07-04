import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v1_vocab64k import convert


convert.TOKENIZER_HASH = "8260b806c4be446bc28a9923dc1909d69a545d0ff8f23505c91980f486158d1a"
convert.OUT = Path(__file__).parent / "ananke-v2-vocab16k-MXFP4.gguf"
convert.VERSION = "V2"
convert.__file__ = str(Path(__file__))


if __name__ == "__main__":
    convert.main()
