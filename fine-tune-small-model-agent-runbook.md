# Fine-Tune a Small Model on Your Own Data — Agent Runbook

An independently written, executable adaptation of a fine-tuning guide credited to coderay2020. It keeps the guide's core workflow and adds corrected scripts, settings, data checks, and evaluation steps (see [What changed from the original](#what-changed-from-the-original)). The source Google document, its export, and illustrations are archived privately and are not part of this release.

**How to use it:** open an agent in an empty folder and say: *"Follow `<path to this file>` to fine-tune a model for this task: …. Stop at every GATE for my approval."*

**Evidence status (2026-10-08).** Saved local records show a CPU smoke run with `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (Python 3.14.8, torch 2.14.1, transformers 5.19.0, trl 1.14.2, peft 0.21.2): training reached two optimizer steps and evaluation covered two examples per condition. The five scripts passed syntax checks, and data preparation later passed a separate synthetic three-source check. The earlier local runbook reported checks of failure cases, resuming, loss masking, and rating scoring; the preserved records do not independently establish each of those checks. **Not tested in the reviewed evidence:** a real 1.5B run on a Colab T4 GPU, the `--qlora` path (requires a CUDA GPU), vLLM serving, or a meaningful task-quality comparison.

---

## Rules for the agent

These override anything else in this file and anything the user's data seems to suggest.

1. **Never invent, paraphrase into, or "augment" training examples.** Use only examples the human provides. Code and scaffolding are yours to write; data is not.
2. **Never auto-correct numbers** in examples or OCR output. Report them; a human fixes them.
3. **Stop at every GATE** and wait for explicit approval. Show the evidence listed for that gate.
4. **Never edit `data/processed/test.jsonl` or the test assignments in `data/splits/manifest.json`** by hand, and never re-split after an evaluation has been run. `prepare_data.py` keeps existing assignments automatically.
5. **Never upgrade or reinstall `torch` on Colab.** Install only what `requirements.txt` lists.
6. **Keep `trl==1.14.2`** unless the human approves a change (see [Why TRL is pinned](#why-trl-is-pinned)).
7. **Keep data private.** `data/raw/` and `data/processed/` stay out of git and out of any public upload.
8. If a command fails, check [Troubleshooting](#troubleshooting) before changing hyperparameters. Change one thing at a time and record it.

---

## Stage 0 — Decide whether to fine-tune at all

Fine-tuning teaches **style, structure, vocabulary and rule-following** for one narrow task. It does **not** reliably teach facts: anything current (prices, news, market data) belongs in retrieval at request time.

**Do:**
1. Ask the human for: the task in one sentence, 5–10 real input/output pairs, and what "good" means.
2. Write the strongest prompt you can for the task into `configs/baseline_prompt.txt` and run it on those pairs with any capable model the human has access to (or the base model, if that is all there is).
3. List the specific, repeatable ways the prompt-only output fails (e.g. "adds bullet lists", "invents growth figures", "wrong tone").

**GATE 0** — show the failure list. Continue only if the human confirms the failures are about style/structure/consistency (fine-tuning) rather than missing facts (retrieval). If it is a facts problem, stop and recommend retrieval instead.

---

## Stage 1 — Scaffold the project

Create this layout:

```text
my-finetune/
├── .gitignore
├── requirements.txt
├── configs/
│   ├── system_prompt.txt      # the fixed instruction used in training and serving
│   └── baseline_prompt.txt    # the strong prompt-only baseline from Stage 0
├── data/
│   ├── raw/                   # PRIVATE: examples.jsonl, sources.jsonl
│   ├── processed/             # PRIVATE: train/validation/test.jsonl (generated)
│   └── splits/manifest.json   # which source went to which split (generated)
├── scripts/
│   ├── number_check.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── score_ratings.py
├── artifacts/adapters/        # trained adapters (generated)
└── results/                   # evaluation outputs (generated)
```

Write each file exactly as given in [Files](#files). `configs/system_prompt.txt` is one short instruction for the task, for example:

```text
Write grounded English funding-pitch copy. Return plain text only. Do not invent metrics, customers, revenue, market size, or traction.
```

**Verify:** `python -c "import ast,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('scripts').glob('*.py')]"` exits 0.

---

## Stage 2 — Collect data with rights

**Base model choice.** Default `Qwen/Qwen2.5-1.5B-Instruct` (not gated; no Hugging Face login needed). Alternatives: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `Qwen/Qwen3-0.6B` (a "thinking" model; `evaluate.py` already disables thinking mode, and your examples must not contain `<think>` blocks). Do not start at 7B+.

**Rights first.** For every source (company, document set, mailbox…), the human adds one line to `data/raw/sources.jsonl`:

```json
{"source_id": "acme", "source_url": "https://...", "owner": "Acme Inc.", "rights_status": "approved", "commercial_use_permitted": true, "rights_reviewed_at": "2026-10-08"}
```

Public is not the same as permitted. The safest data is the human's own (past emails, tickets, proposals, reports). `prepare_data.py` refuses any source without an approved record.

**Examples.** One line per example in `data/raw/examples.jsonl`:

```json
{"source_id": "acme", "input": "Startup brief:\nWe help independent clinics reduce missed appointments...", "output": "Independent clinics lose revenue...", "evidence": "optional extra source text backing claims in the output", "reviewed_by": "jp"}
```

- `input` is exactly what the model will receive at serving time; `output` is the ideal plain-text answer.
- Every number in `output` must appear in `input` or `evidence`. If the source does not back a number, the number does not go in.
- `reviewed_by` names the human who checked the output against the source.
- Aim for 200–500 examples from at least 10 different `source_id`s. Quality beats quantity.

If the source material is PDFs, extract text with normal extraction first, then layout-aware OCR (Docling, PaddleOCR) for near-empty pages, then a human. Never drop empty pages silently, and never let a model "fix" OCR'd numbers.

**GATE 2** — show the count of examples and sources, and confirm every source has an approved rights record.

---

## Stage 3 — Prepare and split the data

```bash
python scripts/prepare_data.py
```

What it does:
- validates every line, checks rights, and rejects any example whose output contains a number not found in its input/evidence (normalized, so `$12.5M` = `12.5 million`, `40%` = `40 percent`, `twelve` = `12`);
- splits **by `source_id`, not by example**, toward 80/10/10 train/validation/test, so no company appears in both training and test;
- writes TRL **prompt-completion** JSONL (`prompt` = system + user messages, `completion` = assistant message), so loss is computed on the answer only;
- records hashes and the split assignment in `data/splits/manifest.json`. On later runs, existing sources keep their split and new sources go to train, so the test set stays frozen.

**If it exits with "Outputs contain numbers not found…":** show the human the listed lines. Do not edit them yourself.

**GATE 3** — show the printed counts and the manifest's `assignment`. Warn if there are fewer than 10 sources.

---

## Stage 4 — Local smoke test (optional, CPU, ~1 minute)

Proves the pipeline runs before spending GPU time. Uses a tiny random model, so outputs are gibberish by design.

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install torch -r requirements.txt
python scripts/train.py --base-model trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --max-steps 2 --output-dir artifacts/adapters/smoke
python scripts/evaluate.py --base-model trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --adapter artifacts/adapters/smoke/final --out-dir results/smoke --limit 2 --max-new-tokens 20
```

**Verify:** `artifacts/adapters/smoke/final/adapter_model.safetensors` and `results/smoke/summary.json` exist. Then delete `artifacts/adapters/smoke` and `results/smoke`.

---

## Stage 5 — Train on Google Colab

**GATE 5** — before starting, tell the human: base model, number of training examples, optimizer steps per epoch (`ceil(train_examples / 16)`), epochs (2), and where outputs will be saved. Wait for approval.

1. The human copies the project folder (including `data/`) to Google Drive, e.g. `MyDrive/my-finetune`.
2. In [Colab](https://colab.research.google.com): new notebook → Runtime → Change runtime type → T4 GPU. Run these cells:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/my-finetune
!nvidia-smi
```

```python
# Does not touch torch: Colab's preinstalled torch matches its CUDA.
%pip install -q -r requirements.txt
```

```python
# 1-minute GPU smoke test
!python scripts/train.py --base-model trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --max-steps 2 --output-dir artifacts/adapters/smoke
```

```python
# Real run: plain LoRA, eval + checkpoint every epoch, best epoch kept
!python scripts/train.py --output-dir artifacts/adapters/v1
```

If Colab disconnects: reconnect, rerun the first two cells, then
`!python scripts/train.py --output-dir artifacts/adapters/v1 --resume`.

**What to expect:** with 400 examples there are about 25 optimizer steps per epoch, so the run is short (expected minutes, not hours, for 1.5B on a T4; not yet measured). Watch `eval_loss` once per epoch. Training loss falling while `eval_loss` rises means memorization; the script keeps the best epoch automatically.

**Defaults and what they mean:**

| Setting | Default | Meaning |
|---|---|---|
| `--lora-r` | 16 | Size of the adapter. `lora_alpha` is set to 2×r. |
| `--lr` | 2e-4 | Step size; standard for LoRA. |
| `--epochs` | 2 | Passes over the data. Raise only if `eval_loss` is still falling at the end. |
| `--batch-size` × `--grad-accum` | 2 × 8 | Effective batch of 16. Lower `--batch-size` to 1 (and raise `--grad-accum` to 16) if out of memory. |
| `--max-length` | 2048 | Longer examples are truncated. |
| `--qlora` | off | 4-bit loading. Not needed at ≤3B (1.5B in fp16 is ~3 GB); use for 7B+. Requires `pip install bitsandbytes`. |

**Verify:** `artifacts/adapters/v1/final/` contains `adapter_model.safetensors` and `training_record.json` (base model, data hashes, hyperparameters, library versions, best checkpoint, eval loss).

---

## Stage 6 — Evaluate against the prompt-only baseline

```python
!python scripts/evaluate.py --adapter artifacts/adapters/v1/final
```

It runs the **untouched test set** through both the fine-tuned adapter and the base model with `configs/baseline_prompt.txt`, using greedy decoding, and writes to `results/v1-<timestamp>/`:

- `summary.json`: per system, % of outputs with unsupported numbers, % format violations (headings, bullets, code fences, JSON), % empty, mean repetition (repeated 4-grams), median seconds, mean new tokens;
- `predictions.jsonl`: every output with its checks;
- `ratings.csv`: a blind A/B sheet. The human fills `rating_1_to_10` without seeing `ratings_key.json`.

After the human rates:

```bash
python scripts/score_ratings.py results/v1-<timestamp>
```

The number check is a filter, not proof: a human should still read every flagged output and a sample of unflagged ones.

**GATE 6** — show both `summary.json` and the ratings summary side by side. Recommend shipping only if the fine-tuned model is better on the human rating **and** not worse on unsupported numbers. Claim an improvement only if the test set was never used for tuning decisions.

If the fine-tuned model is not better: fix the examples (consistency, coverage, unsupported claims) before touching hyperparameters, or reconsider retrieval (Stage 0).

---

## Stage 7 — Serve it

For real traffic, use vLLM (not tested here; check the current vLLM docs first):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --enable-lora \
  --lora-modules mytask=artifacts/adapters/v1/final \
  --max-lora-rank 16 \
  --max-model-len 2048
```

Call it through the OpenAI-compatible API with `"model": "mytask"`, and send the **same system prompt** as `configs/system_prompt.txt`, since that is what the adapter was trained with. `--max-lora-rank` must be at least the `--lora-r` used in training.

Never expose the model server to the internet directly. Put a small gateway (a cheap VPS is fine) in front for HTTPS, authentication, rate limiting and a request queue; the GPU machine only runs the model. For low traffic or testing, a small FastAPI app loading the adapter with `PeftModel.from_pretrained` (as in `evaluate.py`) is enough.

---

## Stage 8 — Monitor and retrain

- Log per request: latency, errors, empty outputs, tokens, retries, user reports of invented claims, and the adapter version that answered.
- Do not collect user inputs for retraining without explicit opt-in consent, and strip personal and confidential details (names, emails, phones, private financials) before storing.
- Retrain only with a meaningful batch of new human-reviewed examples and a named failure pattern to fix. New sources go to train automatically; the test set stays frozen so versions remain comparable.
- Keep a version log: `v0` prompt-only baseline, `v1` ~200 reviewed examples, `v2` ~1,000, `v3` larger set plus retrieval. Each `training_record.json` plus the matching `results/` folder is that version's record. Never deploy without a known rollback adapter.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'triton'` or a Triton/kernel error during training | TRL ≥ 1.15 installed | `pip install "trl==1.14.2"` |
| CUDA out of memory | Batch or length too large | `--batch-size 1 --grad-accum 16`, then `--max-length 1024` |
| `loss` is `nan` on a T4 | fp16 overflow | Add `--fp32-weights` |
| Training "finished" with no eval or checkpoint | Old step-based settings | Use this `train.py`: it evaluates and saves every epoch |
| Great test scores, poor real-world results | Same source in train and test | Check `manifest.json`; examples must be split by `source_id` |
| Model invents numbers | Training outputs contained unsupported numbers | `prepare_data.py` blocks these; audit the `evidence` fields too |
| Repetitive output or copied training text | Too many epochs / too little variety | Fewer epochs (the best epoch is kept), more varied examples |
| Fine-tuned not better than baseline | Inconsistent examples, or a facts problem | Fix the examples; reconsider retrieval |
| Evaluation hangs while loading the model on a Mac | `device_map="auto"` on Apple silicon | Already handled: `evaluate.py` only uses it on CUDA |

### Why TRL is pinned

TRL 1.15.0 (released after 1.14.2) computes the training loss with a Triton GPU kernel and no fallback. Triton only ships on Linux and needs a supported GPU, so 1.15 cannot train on a Mac or CPU at all, and its T4 support is unverified. TRL 1.14.2 uses plain PyTorch for the loss and ran cleanly in testing. Upgrade only after a GPU smoke test (Stage 5) passes on the new version.

---

## Cost

A first run on free Colab costs $0. On a rented GPU (RunPod, Modal, Lambda, Vast.ai; check live prices) a few hundred short examples on a 1.5B model should take well under an hour. The real cost is iteration: rebuilding data, re-running evaluation, trying another base model. Budget several times the first run.

---

## What changed from the original

| Original guide | This runbook | Why |
|---|---|---|
| `eval_steps=50`, `save_steps=50`, `logging_steps=10` | Evaluate and save every epoch; log every step; keep best epoch | With 200–500 examples a run is ~25–60 steps, so step-50 evaluation/checkpoints barely or never fire |
| `packing=True` | `packing=False` | Packing without FlashAttention (unavailable on T4) can let examples attend to each other; no benefit at this size |
| QLoRA by default | Plain LoRA by default, `--qlora` flag for 7B+ | A 1.5B model fits a T4 without 4-bit; plain LoRA is simpler and usually faster |
| `messages` format (loss on system + user + answer) | Prompt-completion format (loss on answer only) | Training on the prompt wastes capacity; verified that only the answer tokens get labels |
| `%pip install -U torch …` | `requirements.txt` without torch | Upgrading torch on Colab can break its CUDA build |
| Unpinned TRL | `trl==1.14.2` | TRL 1.15 requires Triton GPU kernels (see above) |
| Raw-regex number check | Normalized check (scales, %, "percent", spelled-out numbers, no trailing-period false positives) | The original flagged `1,000.`, `12.5 million` and `40 percent` falsely and missed "twelve" |
| Number check only at evaluation | Also blocks unsupported numbers in training data, and requires rights records per source | Enforces the guide's own rules instead of only stating them |
| No code to load and run the adapter | `evaluate.py` with baseline comparison and blind human-rating sheet | The original's evaluation step had no inference code |
| Split by company (described) | Split by `source_id` (implemented), test set frozen across re-runs | Prevents leakage and accidental test-set changes |
| ~3 hours, ~$1.70 for the first run | Minutes to under an hour | A 1.5B LoRA on a few hundred short examples is small |
| vLLM command | Adds `--max-lora-rank` and the system-prompt requirement | vLLM rejects adapters above its max rank; output depends on matching the training prompt |

---

## Files

Write each file exactly as shown, at the path in its heading.

### `.gitignore`

```gitignore
data/raw/
data/processed/
artifacts/
results/
.venv/
__pycache__/
```

### `requirements.txt`

```text
# torch is intentionally absent: use Colab's preinstalled build, or install it yourself locally.
trl==1.14.2
transformers>=4.56.2
peft
datasets
accelerate
# bitsandbytes   # only for --qlora, CUDA only
```

### `scripts/number_check.py`

```python
"""Find numbers in text and normalize them so "$12.5M" and "12.5 million" compare equal."""
import re
from decimal import Decimal, InvalidOperation

SCALES = {
    "k": 10**3, "thousand": 10**3,
    "m": 10**6, "mm": 10**6, "mn": 10**6, "million": 10**6,
    "b": 10**9, "bn": 10**9, "billion": 10**9,
    "t": 10**12, "tn": 10**12, "trillion": 10**12,
}
# "one" is left out on purpose: it is too often a pronoun ("one workflow").
WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "dozen": 12,
}
SCALE_RE = r"(?:thousand|million|billion|trillion|mm|mn|bn|tn|k|m|b|t)(?![a-z])"
UNIT_RE = r"(?:%|percent(?![a-z])|per cent(?![a-z])|x(?![a-z]))"
DIGIT_RE = re.compile(
    rf"(?<![\w.])[$€£₹]?\s?(\d+(?:,\d{{3}})*(?:\.\d+)?)\s?({SCALE_RE})?\s?({UNIT_RE})?",
    re.IGNORECASE,
)
WORD_RE = re.compile(
    rf"\b({'|'.join(WORDS)})(?:-({'|'.join(WORDS)}))?\b(?:\s({SCALE_RE}))?\s?({UNIT_RE})?",
    re.IGNORECASE,
)


def _canonical(value: Decimal, scale: str | None, unit: str | None) -> str:
    if scale:
        value *= SCALES[scale.lower()]
    text = format(value.normalize(), "f")
    if unit:
        unit = unit.lower()
        text += "%" if unit.startswith("per") or unit == "%" else "x"
    return text


def extract_numbers(text: str) -> set[str]:
    """Return the set of canonical numbers in text, e.g. {"12500000", "40%", "10x"}."""
    found = set()
    for m in DIGIT_RE.finditer(text):
        try:
            value = Decimal(m.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        found.add(_canonical(value, m.group(2), m.group(3)))
    for m in WORD_RE.finditer(text):
        value = Decimal(WORDS[m.group(1).lower()])
        if m.group(2):
            value += WORDS[m.group(2).lower()]
        found.add(_canonical(value, m.group(3), m.group(4)))
    return found


def unsupported_numbers(source: str, generated: str) -> set[str]:
    """Numbers in generated that do not appear anywhere in source."""
    return extract_numbers(generated) - extract_numbers(source)
```

### `scripts/prepare_data.py`

```python
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
```

### `scripts/train.py`

```python
"""Train a LoRA (or QLoRA) adapter with TRL on data/processed/{train,validation}.jsonl.

Loss is computed on the assistant completion only (prompt-completion format).
Evaluates and checkpoints once per epoch and keeps the best epoch by validation loss.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import peft
import torch
import transformers
import trl
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--output-dir", default="artifacts/adapters/v1")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--qlora", action="store_true", help="Load the base model in 4-bit (for 7B+ models).")
    ap.add_argument("--fp32-weights", action="store_true", help="Load weights in fp32 (fixes fp16 NaN loss on T4).")
    ap.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --output-dir.")
    ap.add_argument("--max-steps", type=int, default=-1, help="Smoke tests only.")
    args = ap.parse_args()

    versions = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
    }
    print("versions:", versions)

    cuda = torch.cuda.is_available()
    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    load_dtype = torch.float32 if args.fp32_weights else compute_dtype
    print(f"device: {torch.cuda.get_device_name(0) if cuda else 'cpu'}, bf16={use_bf16}, fp16={use_fp16}")

    data_dir = Path(args.data_dir)
    dataset = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "validation.jsonl"),
    })
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in ("prompt", "completion")])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"dtype": load_dtype}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    output_dir = Path(args.output_dir)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        gradient_checkpointing=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    steps_per_epoch = -(-len(dataset["train"]) // (args.batch_size * args.grad_accum))
    print(f"train examples: {len(dataset['train'])}, optimizer steps per epoch: {steps_per_epoch}")

    has_checkpoint = any(output_dir.glob("checkpoint-*"))
    if args.resume and not has_checkpoint:
        print("No checkpoint found; starting from scratch.")
    trainer.train(resume_from_checkpoint=args.resume and has_checkpoint)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = trainer.evaluate()
    record = {
        "base_model": args.base_model,
        "train_sha256": file_sha256(data_dir / "train.jsonl"),
        "validation_sha256": file_sha256(data_dir / "validation.jsonl"),
        "train_examples": len(dataset["train"]),
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("data_dir", "output_dir")},
        "versions": versions,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "final_eval_loss": metrics.get("eval_loss"),
    }
    (final_dir / "training_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
```

### `scripts/evaluate.py`

```python
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
```

### `scripts/score_ratings.py`

```python
"""Unblind a filled-in ratings.csv and report the mean human rating per system."""
import csv
import json
import statistics
import sys
from pathlib import Path


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results/latest")
    key = json.loads((results / "ratings_key.json").read_text())
    scores = {"baseline": [], "fine_tuned": []}
    missing = 0
    with open(results / "ratings.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["rating_1_to_10"].strip():
                missing += 1
                continue
            system = key[row["example_id"]][row["candidate"]]
            scores[system].append(float(row["rating_1_to_10"]))
    report = {s: {"n": len(v), "mean": round(statistics.mean(v), 2) if v else None} for s, v in scores.items()}
    report["unrated_rows"] = missing
    (results / "ratings_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```
