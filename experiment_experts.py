import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from baseline import MODEL, post, run


CASES = [
    {"id": "capital", "prompt": "What is the capital of France? Answer with only the city.", "requires": ["paris"]},
    {"id": "multiply", "prompt": "What is 17 multiplied by 19? Answer with only the number.", "requires": ["323"]},
    {"id": "decimal", "prompt": "Which is larger, 9.11 or 9.9? Answer with only the larger value.", "requires": ["9.9"]},
    {
        "id": "code",
        "prompt": "Write a one-line Python function named is_even that returns whether n is even.",
        "requires": ["def is_even", "% 2"],
    },
    {
        "id": "science",
        "prompt": "Explain in one sentence why the daytime sky looks blue.",
        "requires": ["scatter", "blue"],
    },
    {"id": "instruction", "prompt": "Reply with exactly ANANKE and nothing else.", "exact": "ANANKE"},
]


def repetition(text: str, n: int = 4) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0


def passed(case: dict, output: str) -> bool:
    clean = output.strip()
    if "exact" in case:
        return clean == case["exact"]
    lower = clean.lower()
    return all(term in lower for term in case["requires"])


def unload() -> None:
    run("lms", "unload", MODEL, check=False)


def load(experts: int) -> dict:
    return post(
        "/api/v1/models/load",
        {
            "model": MODEL,
            "context_length": 4096,
            "flash_attention": True,
            "offload_kv_cache_to_gpu": True,
            "num_experts": experts,
            "echo_load_config": True,
        },
    )


def evaluate(experts: int) -> dict:
    unload()
    loaded = load(experts)
    rows = []
    for case in CASES:
        print(f"experts={experts} case={case['id']}...", flush=True)
        try:
            response = post(
                "/api/v0/chat/completions",
                {
                    "model": loaded["instance_id"],
                    "messages": [{"role": "user", "content": case["prompt"]}],
                    "max_tokens": 128,
                    "temperature": 0,
                    "seed": 0,
                    "reasoning_effort": "low",
                },
            )
        except RuntimeError as error:
            rows.append({"id": case["id"], "passed": False, "error": str(error)})
            continue
        message = response["choices"][0]["message"]
        rows.append(
            {
                "id": case["id"],
                "passed": passed(case, message["content"]),
                "tokens_per_second": response["stats"]["tokens_per_second"],
                "time_to_first_token": response["stats"]["time_to_first_token"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "repetition_score": repetition(message["content"]),
                "reasoning": message.get("reasoning"),
                "output": message["content"],
            }
        )
    measured = [row for row in rows if "tokens_per_second" in row]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "active_experts": experts,
        "load": loaded,
        "summary": {
            "pass_rate": sum(row["passed"] for row in rows) / len(rows),
            "mean_tokens_per_second": sum(row["tokens_per_second"] for row in measured) / len(measured),
            "mean_time_to_first_token": sum(row["time_to_first_token"] for row in measured) / len(measured),
            "mean_repetition_score": sum(row["repetition_score"] for row in measured) / len(measured),
            "parser_failures": len(rows) - len(measured),
        },
        "cases": rows,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ablate GPT-OSS active expert count")
    parser.add_argument("--experts", default="1,2,4", help="comma-separated active expert counts")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/expert_sweep"))
    args = parser.parse_args()
    experts = [int(value) for value in args.experts.split(",")]
    if any(value not in (1, 2, 4) for value in experts):
        parser.error("expert counts must be 1, 2, or 4")

    run("lms", "server", "start", check=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for count in experts:
        report = evaluate(count)
        reports.append(report)
        (args.out_dir / f"experts_{count}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary = report["summary"]
        print(f"experts={count}: {summary['mean_tokens_per_second']:.2f} tok/s, {summary['pass_rate']:.0%} pass")

    lines = [
        "# Active-expert sweep",
        "",
        "| experts | pass rate | tok/s | TTFT | parser failures | repetition |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        summary = report["summary"]
        lines.append(
            f"| {report['active_experts']} | {summary['pass_rate']:.0%} | "
            f"{summary['mean_tokens_per_second']:.2f} | {summary['mean_time_to_first_token']:.2f}s | "
            f"{summary['parser_failures']} | {summary['mean_repetition_score']:.4f} |"
        )
    (args.out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
