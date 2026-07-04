import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline import post, run


def main() -> None:
    run("lms", "server", "start", check=False)
    run("lms", "unload", "--all", check=False)
    run("lms", "load", "v3-head", "--gpu", "0.60", "--context-length", "4096", "--parallel", "1",
        "--identifier", "v5-target", "--yes")
    payload = {
        "model": "v5-target",
        "draft_model": "liquid/lfm2.5-1.2b",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer with only the city."}],
        "max_tokens": 128,
        "temperature": 0,
        "seed": 0,
    }
    try:
        response = post("/api/v0/chat/completions", payload)
        report = {"status": "ran", "stats": response["stats"], "output": response["choices"][0]["message"]["content"]}
    except RuntimeError as error:
        report = {"status": "incompatible", "error": str(error)}
    path = Path(__file__).parent / "test_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
