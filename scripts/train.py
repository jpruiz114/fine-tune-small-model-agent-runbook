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
