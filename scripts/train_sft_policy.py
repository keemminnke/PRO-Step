#!/usr/bin/env python3
"""Train policy model with SFT on all-GOOD trajectories.

Only GOOD step tokens contribute to loss (documents masked out).
No ref_model needed — standard cross-entropy.

Usage:
    # 2-GPU DDP
    torchrun --nproc_per_node=2 scripts/train_sft_policy.py \\
        --input outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \\
        --input outputs/regenerated_critic_scored.jsonl \\
        --output-dir outputs/sft_policy_v1 \\
        --wandb-run-name sft_v1

    # Single GPU
    python scripts/train_sft_policy.py \\
        --input outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \\
        --output-dir outputs/sft_policy_v1 --no-wandb
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from prmrag.training.sft_trainer import SFTDataPreparer, SFTDataCollator

HF_CACHE_DIR = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on all-GOOD trajectories")

    parser.add_argument(
        "--input", type=Path, action="append", default=[],
        help="Combined JSONL file(s). Can be specified multiple times.",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=str, default="outputs/sft_policy_v1")

    # Training
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Logging
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="prmrag-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = local_rank in (-1, 0)

    if not args.input:
        print("Error: provide at least one --input file")
        sys.exit(1)
    for p in args.input:
        if not p.exists():
            print(f"Error: not found: {p}")
            sys.exit(1)

    if is_main:
        print("=" * 70)
        print("SFT POLICY TRAINING (all-GOOD trajectories)")
        print("=" * 70)
        num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        eff = args.batch_size * args.gradient_accumulation * num_gpus
        for p in args.input:
            print(f"  Input:  {p}")
        print(f"  Model:  {args.model_name}")
        print(f"  Output: {args.output_dir}")
        print(f"  LR:     {args.learning_rate}")
        print(f"  Batch:  {args.batch_size} x {args.gradient_accumulation} x {num_gpus} = {eff} eff")
        print(f"  LoRA:   r={args.lora_r}, alpha={args.lora_alpha}")
        print()

    # =========================================================
    # 1. Tokenizer
    # =========================================================
    os.environ["HF_HOME"] = HF_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # =========================================================
    # 2. Dataset
    # =========================================================
    if is_main:
        print("Preparing SFT dataset (all-GOOD only)...")
    preparer = SFTDataPreparer(tokenizer)
    dataset = preparer.prepare_dataset(
        input_paths=[str(p) for p in args.input],
        filter_all_good=True,
        limit=args.limit,
    )
    if is_main:
        print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: no training samples!")
        sys.exit(1)

    # =========================================================
    # 3. Model
    # =========================================================
    gpu_id = max(local_rank, 0)
    if is_main:
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE_DIR,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # =========================================================
    # 4. Training args
    # =========================================================
    report_to = []
    if not args.no_wandb and WANDB_AVAILABLE and is_main:
        report_to.append("wandb")
        run_name = args.wandb_run_name or f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "num_samples": len(dataset),
            },
            tags=["sft", "prmrag", "all-good"],
        )
    else:
        report_to.append("tensorboard")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        save_total_limit=2,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # =========================================================
    # 5. Train
    # =========================================================
    collator = SFTDataCollator(tokenizer, max_length=args.max_length)

    # Check for existing checkpoint
    checkpoint_dir = Path(args.output_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    ) if checkpoint_dir.exists() else []
    resume_checkpoint = str(checkpoints[-1]) if checkpoints else None
    if resume_checkpoint and is_main:
        print(f"Resuming from: {resume_checkpoint}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    if is_main:
        print(f"\n{'='*70}")
        print("STARTING SFT TRAINING")
        print(f"{'='*70}\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # =========================================================
    # 6. Save
    # =========================================================
    if is_main:
        final_path = Path(args.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        info = {
            "model_name": args.model_name,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_length": args.max_length,
            "input_files": [str(p) for p in args.input],
            "num_samples": len(dataset),
            "completed_at": datetime.now().isoformat(),
        }
        with open(Path(args.output_dir) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n{'='*70}")
        print(f"SFT TRAINING COMPLETED")
        print(f"Model saved to: {final_path}")
        print(f"{'='*70}")

    if not args.no_wandb and WANDB_AVAILABLE and is_main and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
