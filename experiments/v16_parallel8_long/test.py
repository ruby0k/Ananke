import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v13_parallel4 import test

test.PARALLEL = 8
test.MAX_TOKENS = 512
test.PROMPTS *= 2
test.OUT = Path(__file__).parent / "report.json"

if __name__ == "__main__":
    test.main()
