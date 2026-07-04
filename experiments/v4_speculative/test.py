import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline import post, run
from experiment_experts import CASES, passed


TARGET = "v3-head"
DRAFT = "v4-draft-8l"
TEST_CASES = CASES[:1]  # The untrained draft crashes LM Studio on the second request.


def run_cases(identifier: str, draft: str | None) -> dict:
    rows = []
    for case in TEST_CASES:
        payload = {
            "model": identifier,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 256,
            "temperature": 0,
            "seed": 0,
            "reasoning_effort": "low",
        }
        if draft:
            payload["draft_model"] = draft
        try:
            response = post("/api/v0/chat/completions", payload)
        except RuntimeError as error:
            if draft and "Model reloaded" in str(error):
                response = post("/api/v0/chat/completions", payload)
            else:
                raise
        message = response["choices"][0]["message"]
        stats = response["stats"]
        usage = response["usage"]
        rows.append(
            {
                "id": case["id"],
                "passed": passed(case, message["content"]),
                "output": message["content"],
                "tokens_per_second": stats["tokens_per_second"],
                "draft_tokens": stats.get("total_draft_tokens_count", usage.get("total_draft_tokens_count", 0)),
                "accepted_draft_tokens": stats.get("accepted_draft_tokens_count", usage.get("accepted_draft_tokens_count", 0)),
            }
        )
        print(f"{draft or 'no_spec'} {case['id']}: {'pass' if rows[-1]['passed'] else 'FAIL'}", flush=True)
    drafted = sum(row["draft_tokens"] for row in rows)
    accepted = sum(row["accepted_draft_tokens"] for row in rows)
    return {
        "summary": {
            "pass_rate": sum(row["passed"] for row in rows) / len(rows),
            "mean_tokens_per_second": sum(row["tokens_per_second"] for row in rows) / len(rows),
            "draft_acceptance": accepted / drafted if drafted else None,
            "draft_tokens": drafted,
            "accepted_draft_tokens": accepted,
        },
        "cases": rows,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run("lms", "server", "start", check=False)
    run("lms", "unload", "--all", check=False)
    identifier = "v4-no-spec"
    run("lms", "load", TARGET, "--gpu", "0.30", "--context-length", "4096", "--parallel", "1",
        "--identifier", identifier, "--yes")
    results = {"v3_no_spec": run_cases(identifier, None)}
    run("lms", "unload", "--all", check=False)
    identifier = "v4-spec"
    run("lms", "load", TARGET, "--gpu", "0.30", "--context-length", "4096", "--parallel", "1",
        "--identifier", identifier, "--yes")
    results["v4_spec_8l"] = run_cases(identifier, DRAFT)
    out = Path(__file__).parent / "test_report.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# V4 test report",
        "",
        "| mode | pass rate | tok/s | draft acceptance |",
        "|---|---:|---:|---:|",
    ]
    for label, result in results.items():
        summary = result["summary"]
        acceptance = "—" if summary["draft_acceptance"] is None else f"{summary['draft_acceptance']:.1%}"
        lines.append(f"| {label} | {summary['pass_rate']:.0%} | {summary['mean_tokens_per_second']:.2f} | {acceptance} |")
    (Path(__file__).parent / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
