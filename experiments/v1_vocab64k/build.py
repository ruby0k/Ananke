import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dist/ananke-gpt-oss-20b-top2"
OUT = Path(__file__).resolve().parent / "hf"
CORPUS = Path(r"D:\Hegemony\Calliope\data\v3_general_gpt2")
TARGET_VOCAB = 64_000
SPECIAL_COUNT = 21
TARGET_BASE = TARGET_VOCAB - SPECIAL_COUNT
SAMPLE_TOKENS = 2_000_000
VERSION = "v1"
CHANGE = "vocabulary 201088 -> 64000"


def sample_corpus() -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(CORPUS / "tokenizer")
    data = np.memmap(CORPUS / "train.bin", dtype=np.uint16, mode="r")
    chunk = 20_000
    starts = np.linspace(0, len(data) - chunk, SAMPLE_TOKENS // chunk, dtype=int)
    return [tokenizer.decode(data[start : start + chunk].tolist()) for start in starts]


def select_tokens(tokenizer_json: dict, counts: Counter[int]) -> list[int]:
    vocab = tokenizer_json["model"]["vocab"]
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    parents = {}
    for left, right in tokenizer_json["model"]["merges"]:
        merged = left + right
        if merged in vocab:
            parents[vocab[merged]] = (vocab[left], vocab[right])

    def closure(token_id: int) -> set[int]:
        needed = {token_id}
        stack = [token_id]
        while stack:
            current = stack.pop()
            for parent in parents.get(current, ()):
                if parent not in needed:
                    needed.add(parent)
                    stack.append(parent)
        return needed

    selected = set(range(256))
    ranked = sorted(range(len(vocab)), key=lambda token_id: (-counts[token_id], token_id))
    for token_id in ranked:
        needed = closure(token_id) - selected
        if len(selected) + len(needed) <= TARGET_BASE:
            selected.update(needed)
        if len(selected) == TARGET_BASE:
            break
    if len(selected) != TARGET_BASE:
        raise RuntimeError(f"selected {len(selected)} base tokens, expected {TARGET_BASE}")
    return sorted(selected)


def remap_tokenizer(selected: list[int]) -> tuple[dict, dict[int, int]]:
    tokenizer_json = json.loads((SOURCE / "tokenizer.json").read_text(encoding="utf-8"))
    old_vocab = tokenizer_json["model"]["vocab"]
    id_to_token = {token_id: token for token, token_id in old_vocab.items()}
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected)}
    tokenizer_json["model"]["vocab"] = {id_to_token[old_id]: old_to_new[old_id] for old_id in selected}
    kept_tokens = set(tokenizer_json["model"]["vocab"])
    tokenizer_json["model"]["merges"] = [
        [left, right]
        for left, right in tokenizer_json["model"]["merges"]
        if left in kept_tokens and right in kept_tokens and left + right in kept_tokens
    ]
    for offset, token in enumerate(tokenizer_json["added_tokens"]):
        old_to_new[token["id"]] = TARGET_BASE + offset
        token["id"] = TARGET_BASE + offset
    return tokenizer_json, old_to_new


def rewrite_metadata(old_to_new: dict[int, int], tokenizer_json: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in ("chat_template.jinja", "special_tokens_map.json"):
        shutil.copy2(SOURCE / name, OUT / name)

    tokenizer_config = json.loads((SOURCE / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_config["added_tokens_decoder"] = {
        str(old_to_new[int(old_id)]): value for old_id, value in tokenizer_config["added_tokens_decoder"].items()
    }
    (OUT / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    config.update({"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "vocab_size": TARGET_VOCAB})
    config.pop("auto_map", None)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in config:
            config[key] = old_to_new[config[key]]
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    generation = json.loads((SOURCE / "generation_config.json").read_text(encoding="utf-8"))
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in generation:
            value = generation[key]
            generation[key] = [old_to_new[token_id] for token_id in value] if isinstance(value, list) else old_to_new[value]
    (OUT / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")


def slice_weights(old_to_new: dict[int, int]) -> None:
    index = json.loads((SOURCE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    changed_shard = weight_map["model.embed_tokens.weight"]
    assert changed_shard == weight_map["lm_head.weight"]
    for shard in set(weight_map.values()) - {changed_shard}:
        destination = OUT / shard
        if not destination.exists():
            os.link(SOURCE / shard, destination)

    state = load_file(SOURCE / changed_shard, device="cpu")
    retained_old_ids = [old_id for old_id, _ in sorted(old_to_new.items(), key=lambda item: item[1])]
    rows = torch.tensor(retained_old_ids)
    state["model.embed_tokens.weight"] = state["model.embed_tokens.weight"].index_select(0, rows)
    state["lm_head.weight"] = state["lm_head.weight"].index_select(0, rows)
    with safe_open(SOURCE / changed_shard, framework="pt") as source:
        metadata = source.metadata()
    save_file(state, OUT / changed_shard, metadata=metadata)
    index["metadata"]["total_size"] = sum(tensor.numel() * tensor.element_size() for tensor in state.values()) + sum(
        (OUT / shard).stat().st_size for shard in set(weight_map.values()) - {changed_shard}
    )
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    texts = sample_corpus()
    original = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True)
    counts = Counter()
    original_token_count = 0
    for text in texts:
        ids = original.encode(text, add_special_tokens=False)
        counts.update(ids)
        original_token_count += len(ids)

    source_json = json.loads((SOURCE / "tokenizer.json").read_text(encoding="utf-8"))
    selected = select_tokens(source_json, counts)
    tokenizer_json, old_to_new = remap_tokenizer(selected)
    rewrite_metadata(old_to_new, tokenizer_json)
    slice_weights(old_to_new)

    pruned = AutoTokenizer.from_pretrained(OUT)
    pruned_token_count = sum(len(pruned.encode(text, add_special_tokens=False)) for text in texts)
    report = {
        "version": VERSION,
        "change": CHANGE,
        "corpus": str(CORPUS / "train.bin"),
        "sample_source_tokens": SAMPLE_TOKENS,
        "original_tokens": original_token_count,
        "pruned_tokens": pruned_token_count,
        "token_inflation": pruned_token_count / original_token_count,
        "retained_old_ids": [old_id for old_id, _ in sorted(old_to_new.items(), key=lambda item: item[1])],
    }
    (Path(__file__).parent / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert len(pruned) == TARGET_VOCAB
    assert load_file(OUT / "model-00002-of-00002.safetensors")["lm_head.weight"].shape == (TARGET_VOCAB, 2880)
    print(f"built {VERSION}: {TARGET_VOCAB} tokens, {report['token_inflation']:.3f}x token inflation")


if __name__ == "__main__":
    main()
