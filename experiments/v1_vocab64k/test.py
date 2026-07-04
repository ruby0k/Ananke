import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline import post, run
from experiment_experts import CASES, passed


MODELS = {
    "control": "ananke-gpt-oss-20b-top2",
    "v1_vocab64k": "ananke-v1-vocab64k",
}
VERSION = "V1"


def test_model(label: str, model_key: str) -> dict:
    identifier = f"{VERSION.lower()}-{label}"
    run("lms", "unload", "--all", check=False)
    run(
        "lms",
        "load",
        model_key,
        "--gpu",
        "0.60",
        "--context-length",
        "4096",
        "--parallel",
        "1",
        "--identifier",
        identifier,
        "--yes",
    )
    rows = []
    for case in CASES:
        try:
            response = post(
                "/api/v0/chat/completions",
                {
                    "model": identifier,
                    "messages": [{"role": "user", "content": case["prompt"]}],
                    "max_tokens": 256,
                    "temperature": 0,
                    "seed": 0,
                    "reasoning_effort": "low",
                },
            )
        except RuntimeError as error:
            rows.append({"id": case["id"], "passed": False, "error": str(error)})
            continue
        message = response["choices"][0]["message"]
        seconds = response["stats"]["generation_time"]
        rows.append(
            {
                "id": case["id"],
                "passed": passed(case, message["content"]),
                "output": message["content"],
                "reasoning": message.get("reasoning"),
                "completion_tokens": response["usage"]["completion_tokens"],
                "tokens_per_second": response["stats"]["tokens_per_second"],
                "characters_per_second": len(message["content"]) / seconds,
                "finish_reason": response["choices"][0]["finish_reason"],
            }
        )
        print(f"{label} {case['id']}: {'pass' if rows[-1]['passed'] else 'FAIL'}", flush=True)
    measured = [row for row in rows if "tokens_per_second" in row]
    return {
        "model": model_key,
        "gpu_offload": 0.60,
        "summary": {
            "pass_rate": sum(row["passed"] for row in rows) / len(rows),
            "mean_tokens_per_second": sum(row["tokens_per_second"] for row in measured) / len(measured),
            "mean_characters_per_second": sum(row["characters_per_second"] for row in measured) / len(measured),
            "parser_failures": len(rows) - len(measured),
        },
        "cases": rows,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run("lms", "server", "start", check=False)
    results = {label: test_model(label, model) for label, model in MODELS.items()}
    out = Path(__file__).parent / "test_report.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# {VERSION} test report",
        "",
        "| model | pass rate | tok/s | visible chars/s | parser failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, report in results.items():
        summary = report["summary"]
        lines.append(
            f"| {label} | {summary['pass_rate']:.0%} | {summary['mean_tokens_per_second']:.2f} | "
            f"{summary['mean_characters_per_second']:.2f} | {summary['parser_failures']} |"
        )
    (Path(__file__).parent / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
