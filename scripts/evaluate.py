"""Compare the fine-tuned adapter against the prompt-only baseline on the frozen test set.

Writes results/<name>/:
  predictions.jsonl  every output with automatic checks
  summary.json       automatic metrics per system
  ratings.csv        blind A/B sheet for a human to fill in (rating 1-10)
  ratings_key.json   which of A/B is which system (do not show to the rater)
"""
import argparse
import csv
import json
import random
import re
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from number_check import unsupported_numbers

FORMAT_VIOLATION = re.compile(r"^\s*(#|[-*•]\s|\d+\.\s|```|[{\[])", re.MULTILINE)


def repetition_rate(text: str, n: int = 4) -> float:
    words = text.lower().split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def generate(model, tokenizer, messages, max_new_tokens):
    # enable_thinking=False only matters for Qwen3-style templates; other templates ignore it.
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    seconds = time.perf_counter() - start
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), seconds, len(new_tokens)


def score(source: str, output: str) -> dict:
    return {
        "unsupported_numbers": sorted(unsupported_numbers(source, output)),
        "format_violation": bool(FORMAT_VIOLATION.search(output)),
        "empty": not output.strip(),
        "repetition": round(repetition_rate(output), 3),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "pct_with_unsupported_numbers": round(100 * sum(bool(r["unsupported_numbers"]) for r in rows) / len(rows), 1),
        "pct_format_violations": round(100 * sum(r["format_violation"] for r in rows) / len(rows), 1),
        "pct_empty": round(100 * sum(r["empty"] for r in rows) / len(rows), 1),
        "mean_repetition": round(statistics.mean(r["repetition"] for r in rows), 3),
        "median_seconds": round(statistics.median(r["seconds"] for r in rows), 2),
        "mean_new_tokens": round(statistics.mean(r["new_tokens"] for r in rows), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="artifacts/adapters/v1/final")
    ap.add_argument("--test-file", default="data/processed/test.jsonl")
    ap.add_argument("--baseline-prompt", default="configs/baseline_prompt.txt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"results/{Path(args.adapter).parent.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    baseline_prompt = Path(args.baseline_prompt).read_text(encoding="utf-8").strip()
    examples = [json.loads(l) for l in Path(args.test_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    examples = examples[: args.limit] if args.limit else examples

    rows = []
    for i, ex in enumerate(examples):
        user_input = ex["prompt"][-1]["content"]
        reference = ex["completion"][0]["content"]
        baseline_messages = [{"role": "system", "content": baseline_prompt}, {"role": "user", "content": user_input}]
        with model.disable_adapter():
            base_out, base_s, base_n = generate(model, tokenizer, baseline_messages, args.max_new_tokens)
        ft_out, ft_s, ft_n = generate(model, tokenizer, ex["prompt"], args.max_new_tokens)
        for system, output, seconds, n in (("baseline", base_out, base_s, base_n), ("fine_tuned", ft_out, ft_s, ft_n)):
            rows.append({"example_id": i, "source_id": ex.get("source_id"), "system": system, "input": user_input,
                         "reference": reference, "output": output, "seconds": round(seconds, 3),
                         "new_tokens": n, **score(user_input, output)})
        print(f"[{i + 1}/{len(examples)}] done")

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {s: summarize([r for r in rows if r["system"] == s]) for s in ("baseline", "fine_tuned")}
    summary["base_model"] = args.base_model
    summary["adapter"] = args.adapter
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rng = random.Random(0)
    key = {}
    with open(out_dir / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "input", "candidate", "output", "rating_1_to_10", "notes"])
        for i in range(len(examples)):
            pair = [r for r in rows if r["example_id"] == i]
            rng.shuffle(pair)
            key[i] = {"A": pair[0]["system"], "B": pair[1]["system"]}
            for label, r in zip("AB", pair):
                w.writerow([i, r["input"], label, r["output"], "", ""])
    (out_dir / "ratings_key.json").write_text(json.dumps(key, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Results in {out_dir}. Give ratings.csv (not ratings_key.json) to a human rater.")


if __name__ == "__main__":
    main()
