import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v1_vocab64k import build


build.OUT = Path(__file__).parent / "hf"
build.TARGET_VOCAB = 16_000
build.TARGET_BASE = build.TARGET_VOCAB - build.SPECIAL_COUNT
build.VERSION = "v2"
build.CHANGE = "vocabulary 201088 -> 16000"


if __name__ == "__main__":
    build.main()
