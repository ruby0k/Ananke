import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v1_vocab64k import test


test.MODELS = {
    "control": "ananke-gpt-oss-20b-top2",
    "v1_vocab64k": "ananke-v1-vocab64k",
    "v2_vocab16k": "v2-vocab16k",
}
test.VERSION = "V2"
test.__file__ = str(Path(__file__))


if __name__ == "__main__":
    test.main()
