"""Validate reviewed examples, split them by source, and write TRL prompt-completion JSONL.

Input: data/raw/examples.jsonl, one object per line:
  {"source_id": "acme", "input": "...", "output": "...", "evidence": "...", "reviewed_by": "jp"}
  - source_id: the company/document the example came from (used for splitting)
  - evidence: optional extra source text that backs claims in the output
  - reviewed_by: required; the human who checked the output against the source

Rights: data/raw/sources.jsonl needs one approved record per source_id:
  {"source_id": "acme", "source_url": "...", "owner": "...", "rights_status": "approved",
   "commercial_use_permitted": true, "rights_reviewed_at": "2026-10-08"}

Output: data/processed/{train,validation,test}.jsonl and data/splits/manifest.json.
Sources already in the manifest keep their split, so the test set never changes;
new sources go to train.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from number_check import unsupported_numbers

REQUIRED = ("source_id", "input", "output", "reviewed_by")


def load_examples(path: Path) -> list[dict]:
    examples, errors = [], []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {line_no}: invalid JSON ({e})")
            continue
        missing = [k for k in REQUIRED if not str(ex.get(k, "")).strip()]
        if missing:
            errors.append(f"line {line_no}: missing {', '.join(missing)}")
            continue
        ex["_line"] = line_no
        examples.append(ex)
    if errors:
        sys.exit("Fix these examples first:\n  " + "\n  ".join(errors))
    return examples


def check_rights(path: Path, source_ids: set[str]) -> None:
    if not path.exists():
        sys.exit(f"Missing {path}: add a rights record for every source before training on it.")
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["source_id"]] = rec
    unapproved = sorted(s for s in source_ids if records.get(s, {}).get("rights_status") != "approved")
    if unapproved:
        sys.exit(f"No approved rights record for: {', '.join(unapproved)}")


def check_numbers(examples: list[dict]) -> list[str]:
    problems = []
    for ex in examples:
        source = ex["input"] + "\n" + ex.get("evidence", "")
        bad = unsupported_numbers(source, ex["output"])
        if bad:
            problems.append(f"line {ex['_line']} ({ex['source_id']}): {sorted(bad)}")
    return problems


def assign_splits(sources: dict[str, int], existing: dict[str, str], seed: int) -> dict[str, str]:
    """Greedy split of whole sources toward 80/10/10 of examples; existing assignments are kept."""
    assignment = {s: existing[s] for s in sources if s in existing}
    new = [s for s in sources if s not in existing]
    if existing:
        assignment.update({s: "train" for s in new})
        return assignment
    rng = random.Random(seed)
    rng.shuffle(new)
    total = sum(sources.values())
    targets = {"test": 0.1 * total, "validation": 0.1 * total}
    filled = {"test": 0, "validation": 0}
    for s in new:
        split = next((k for k in ("test", "validation") if filled[k] < targets[k]), "train")
        assignment[s] = split
        if split != "train":
            filled[split] += sources[s]
    return assignment


def to_record(ex: dict, system_prompt: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["input"]},
        ],
        "completion": [{"role": "assistant", "content": ex["output"]}],
        "source_id": ex["source_id"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/examples.jsonl")
    ap.add_argument("--sources", default="data/raw/sources.jsonl")
    ap.add_argument("--system-prompt", default="configs/system_prompt.txt")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--manifest", default="data/splits/manifest.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw)
    examples = load_examples(raw)
    check_rights(Path(args.sources), {ex["source_id"] for ex in examples})
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()

    problems = check_numbers(examples)
    if problems:
        sys.exit(
            "Outputs contain numbers not found in input or evidence. A human must fix these "
            "(remove the number or add the supporting evidence); do not auto-correct:\n  "
            + "\n  ".join(problems)
        )

    sources: dict[str, int] = {}
    for ex in examples:
        sources[ex["source_id"]] = sources.get(ex["source_id"], 0) + 1
    if len(sources) < 3:
        sys.exit(f"Need at least 3 distinct source_id values to split by source; found {len(sources)}.")

    manifest_path = Path(args.manifest)
    existing = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())["assignment"]
    assignment = assign_splits(sources, existing, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        rows = [to_record(ex, system_prompt) for ex in examples if assignment[ex["source_id"]] == split]
        if not rows:
            sys.exit(f"Split '{split}' is empty; add more sources.")
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(rows)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "seed": args.seed,
        "counts": counts,
        "sources": len(sources),
        "assignment": dict(sorted(assignment.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("raw_sha256", "counts", "sources")}, indent=2))
    if len(sources) < 10:
        print(f"WARNING: only {len(sources)} sources; validation/test scores will be noisy.")


if __name__ == "__main__":
    main()
