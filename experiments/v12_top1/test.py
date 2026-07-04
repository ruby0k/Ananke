import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline import post, run
from experiment_experts import CASES, passed

MODEL = "v11-experts23"


def main() -> None:
    run("lms", "server", "start", check=False)
    run("lms", "unload", "--all", check=False)
    loaded = post("/api/v1/models/load", {
        "model": MODEL, "context_length": 1024, "flash_attention": True,
        "offload_kv_cache_to_gpu": True, "num_experts": 1, "echo_load_config": True,
    })
    rows = []
    for case in CASES:
        response = post("/api/v0/chat/completions", {
            "model": loaded["instance_id"], "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 128, "temperature": 0, "seed": 0, "reasoning_effort": "low",
        })
        message = response["choices"][0]["message"]
        row = {"id": case["id"], "passed": passed(case, message["content"]),
               "tokens_per_second": response["stats"]["tokens_per_second"], "output": message["content"]}
        rows.append(row)
        print(f'{row["id"]}: {row["tokens_per_second"]:.2f} tok/s, pass={row["passed"]}', flush=True)
    report = {"model": MODEL, "change": "top-2 -> top-1 routing", "load": loaded,
              "mean_tokens_per_second": sum(x["tokens_per_second"] for x in rows) / len(rows),
              "pass_rate": sum(x["passed"] for x in rows) / len(rows), "cases": rows}
    (Path(__file__).parent / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f'V12: {report["mean_tokens_per_second"]:.2f} tok/s, {report["pass_rate"]:.0%} pass')


if __name__ == "__main__":
    main()
