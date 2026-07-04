import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.v10_experts24 import build

build.OUT = Path(__file__).parent / "hf"
build.EXPERTS = 23
build.VERSION = "v11"
build.CALIBRATION = ROOT / "experiments/v10_experts24/calibration.json"
build.__file__ = str(Path(__file__))

if __name__ == "__main__":
    build.main()
