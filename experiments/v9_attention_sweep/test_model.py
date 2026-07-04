import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline import post, run
from experiment_experts import CASES, passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    identifier = "v9-test"
    run("lms", "unload", "--all", check=False)
    run("lms", "load", args.model, "--gpu", "0.60", "--context-length", "4096", "--parallel", "1",
        "--identifier", identifier, "--yes")
    rows = []
    for case in CASES:
        try:
            response = post("/api/v0/chat/completions", {
                "model": identifier, "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": 256, "temperature": 0, "seed": 0, "reasoning_effort": "low",
            })
        except RuntimeError as error:
            rows.append({"id": case["id"], "passed": False, "error": str(error)})
            print(f"{case['id']}: ERROR", flush=True)
            continue
        message = response["choices"][0]["message"]
        rows.append({"id": case["id"], "passed": passed(case, message["content"]),
                     "tokens_per_second": response["stats"]["tokens_per_second"], "output": message["content"]})
        print(f"{case['id']}: {'pass' if rows[-1]['passed'] else 'FAIL'}", flush=True)
    filler = "The archive records ordinary weather observations and routine maintenance notes. " * 280
    prompt = filler[:len(filler)//2] + " The secret code is ORCHID-7319. " + filler[len(filler)//2:] + \
             "\nWhat is the secret code? Reply with only the code."
    try:
        response = post("/api/v0/chat/completions", {
            "model": identifier, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128, "temperature": 0, "seed": 0, "reasoning_effort": "low",
        })
        content = response["choices"][0]["message"]["content"]
        context = {"passed": "ORCHID-7319" in content, "output": content,
                   "prompt_tokens": response["usage"]["prompt_tokens"],
                   "time_to_first_token": response["stats"]["time_to_first_token"],
                   "tokens_per_second": response["stats"]["tokens_per_second"]}
    except RuntimeError as error:
        context = {"passed": False, "error": str(error)}
    measured = [row for row in rows if "tokens_per_second" in row]
    report = {"model": args.model, "smoke_pass_rate": sum(r["passed"] for r in rows) / len(rows),
              "mean_tokens_per_second": sum(r["tokens_per_second"] for r in measured) / len(measured) if measured else None,
              "cases": rows, "context": context}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("smoke_pass_rate", "mean_tokens_per_second", "context")}, indent=2))


if __name__ == "__main__":
    main()
