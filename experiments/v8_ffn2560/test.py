import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v1_vocab64k import test

test.MODELS = {"v7_hidden2560": "v7-hidden2560", "v8_ffn2560": "v8-ffn2560"}
test.VERSION = "V8"
test.__file__ = str(Path(__file__))

if __name__ == "__main__":
    test.main()
