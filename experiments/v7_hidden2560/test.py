import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v1_vocab64k import test


test.MODELS = {
    "v3_head_q8": "v3-head",
    "v7_hidden2560": "v7-hidden2560",
}
test.VERSION = "V7"
test.__file__ = str(Path(__file__))


if __name__ == "__main__":
    test.main()
