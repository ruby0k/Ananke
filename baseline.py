import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MODEL = "openai/gpt-oss-20b"
IDENTIFIER = "ananke-baseline"
API = "http://localhost:1234"


def run(*args: str, check: bool = True) -> None:
    subprocess.run(args, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post(path: str, payload: dict) -> dict:
    request = Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8", errors="replace")) from error


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="GPT-OSS-20B MXFP4 baseline through LM Studio")
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu", type=float, default=0.60)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()

    if not 0 <= args.gpu <= 1:
        parser.error("--gpu must be between 0 and 1")

    run("lms", "server", "start", check=False)
    run("lms", "unload", IDENTIFIER, check=False)
    run(
        "lms",
        "load",
        MODEL,
        "--gpu",
        str(args.gpu),
        "--context-length",
        str(args.context_length),
        "--parallel",
        "1",
        "--no-speculative-draft-mtp",
        "--identifier",
        IDENTIFIER,
        "--yes",
    )

    response = post(
        "/api/v0/chat/completions",
        {
            "model": IDENTIFIER,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_new_tokens,
            "temperature": 0,
            "seed": 0,
            "reasoning_effort": "low",
        },
    )
    message = response["choices"][0]["message"]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "load_settings": {
            "quantization": response["model_info"]["quant"],
            "gpu_offload_ratio": args.gpu,
            "context_length": args.context_length,
            "parallel": 1,
            "speculative_decoding": False,
        },
        "prompt": args.prompt,
        "usage": response["usage"],
        "stats": response["stats"],
        "runtime": response["runtime"],
        "reasoning": message.get("reasoning"),
        "output": message["content"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result["output"])
    print(
        f"\n{result['stats']['tokens_per_second']:.2f} tok/s; "
        f"{result['stats']['time_to_first_token']:.2f}s TTFT; saved {args.output}"
    )


if __name__ == "__main__":
    main()
