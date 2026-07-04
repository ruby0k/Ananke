import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / ".tools/llama-b9861-cuda12"
SERVER = TOOLS / "llama-server.exe"
MODEL = ROOT / "experiments/v3_head_q8/ananke-v3-vocab64k-head-q8.gguf"
VENDOR = Path.home() / ".lmstudio/extensions/backends/vendor/win-llama-cuda12-vendor-v2"
PORT = 1235
CASES = [
    ("capital", "What is the capital of France? Answer with only the city."),
    ("multiply", "What is 17 multiplied by 19? Answer with only the number."),
    ("copy_text", "Repeat exactly and nothing else: The rain in Spain falls mainly on the plain, and the train returns again."),
    ("copy_code", "Repeat exactly and nothing else: for item in items:\n    print(item)\nfor item in items:\n    print(item)"),
]


def request(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as response:
        return json.load(response)


def run_mode(mode: str) -> dict:
    log_path = Path(__file__).parent / f"server_{mode}.log"
    env = os.environ.copy()
    env["PATH"] = f"{VENDOR};{TOOLS};{env['PATH']}"
    args = [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
            "-c", "4096", "-np", "1", "-ngl", "14", "--spec-type", mode, "--no-webui"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=TOOLS, env=env, stdout=log, stderr=log)
        try:
            for _ in range(120):
                time.sleep(0.5)
                try:
                    if request("/health").get("status") == "ok":
                        break
                except Exception:
                    pass
            else:
                raise RuntimeError(f"server failed to start; see {log_path}")

            rows = []
            for case_id, prompt in CASES:
                response = request("/completion", {
                    "prompt": prompt,
                    "n_predict": 128,
                    "temperature": 0,
                    "seed": 0,
                })
                rows.append({
                    "id": case_id,
                    "tokens_per_second": response["timings"]["predicted_per_second"],
                    "tokens": response["tokens_predicted"],
                    "output": response["content"],
                })
                print(f"{mode} {case_id}: {rows[-1]['tokens_per_second']:.2f} tok/s", flush=True)
        finally:
            process.terminate()
            process.wait(timeout=30)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    counts = [(int(accepted), int(generated)) for accepted, generated in
              re.findall(r"\(\s*(\d+) accepted /\s*(\d+) generated\)", log_text)]
    return {
        "summary": {
            "mean_tokens_per_second": sum(row["tokens_per_second"] for row in rows) / len(rows),
            "mean_draft_acceptance": sum(a for a, _ in counts) / sum(g for _, g in counts) if counts else None,
        },
        "cases": rows,
    }


def main() -> None:
    results = {mode: run_mode(mode) for mode in ("none", "ngram-simple")}
    out = Path(__file__).parent / "test_report.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# V6 test report", "", "| mode | mean tok/s | mean draft acceptance |", "|---|---:|---:|"]
    for mode, result in results.items():
        summary = result["summary"]
        acceptance = "—" if summary["mean_draft_acceptance"] is None else f"{summary['mean_draft_acceptance']:.1%}"
        lines.append(f"| {mode} | {summary['mean_tokens_per_second']:.2f} | {acceptance} |")
    (Path(__file__).parent / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
