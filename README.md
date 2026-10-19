# Fine-tune a small model: corrected agent runbook

An executable small-model fine-tuning workflow by Jean Paul Ruiz, adapted from a guide credited to coderay2020. The [agent runbook](fine-tune-small-model-agent-runbook.md) contains the full method and explains the corrections to training, data preparation, and evaluation. The original Google document and illustrations are not included. See [NOTICE.md](NOTICE.md) for provenance.

This repository includes five scripts, requirements, and editable prompt examples. It excludes raw and processed examples, source documents, adapter weights, and historical experiment artifacts. The implementation is open source under [MIT](LICENSE).

## What changed

- Training evaluates and checkpoints every epoch, so short runs reach validation.
- Prompt-completion data and completion-only loss target the answer text.
- The initial path uses LoRA without quantized loading or example packing; QLoRA remains optional.
- Preparation checks source rights records, normalizes selected number formats, and freezes source-group split assignments.
- Evaluation compares the adapter with a prompt-only baseline and produces a human rating sheet.

Read the runbook for caveats and full commands. These choices are intended to make a small experiment easier to inspect; they are not evidence that an adapter improves quality.

## Start locally

Use Python 3.10 or newer and install PyTorch appropriate for your machine before installing the other dependencies. This machine's system `python3` is 3.9; use `python3.14` here. The recorded smoke-test environment used Python 3.14.8, PyTorch 2.14.1, Transformers 5.19.0, TRL 1.14.2, and PEFT 0.21.2; those are historical values, not a fresh compatibility guarantee.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install torch -r requirements.txt
```

Edit the two files in `configs/` for your own task. Add reviewed examples and source-rights records under ignored `data/raw/` as described in the runbook. Then run:

```sh
python scripts/prepare_data.py
```

The [runbook](fine-tune-small-model-agent-runbook.md) gives the CPU smoke-test commands, GPU training route, evaluation, and human review gates. A real training run requires your own approved data, model downloads, and compute. The project does not bundle a dataset.

## Evidence and limits

The local work exercised the pipeline on a tiny test model. The best-supported recovered run used 33 training examples and at most two optimizer steps; evaluation had two examples per condition. That establishes a pipeline smoke test, not useful task-quality gains. GPU training, the QLoRA path, and production serving were not verified in the inspected artifacts. No new training was performed for this repository release.

This release contains the runnable project and its runbook. Further task-quality evaluation remains future work.
