import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import experiment_experts as test

test.MODEL = "v4-draft-8l"

if __name__ == "__main__":
    report = test.evaluate(2)
    (Path(__file__).parent / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    print(f'V19: {summary["mean_tokens_per_second"]:.2f} tok/s, {summary["pass_rate"]:.0%} pass')
