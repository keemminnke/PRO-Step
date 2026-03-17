#!/usr/bin/env python3
"""Train policy model with step-level KTO loss.

Usage:
    # Combined format (new pipeline: original + regenerated)
    python scripts/train_kto_policy.py \
        --input outputs/hotpotqa_generative_results_v8_per_trajectory.jsonl \
        --input outputs/regenerated_generative_scored.jsonl \
        --output-dir outputs/kto_policy_v2 \
        --truncate-after-bad \
        --wandb-run-name kto_policy_v2

    # Legacy format (two separate files)
    python scripts/train_kto_policy.py \
        --trajectories outputs/judge_labels_merged_hotpotqa_musique.jsonl \
        --generative-scores outputs/generative_scores_v8_binary_per_trajectory.jsonl \
        --output-dir outputs/kto_policy_v1 \
        --truncate-after-bad \
        --wandb-run-name kto_policy_v1

    # Small test
    python scripts/train_kto_policy.py \
        --input outputs/hotpotqa_generative_results_v8_per_trajectory.jsonl \
        --output-dir outputs/kto_policy_test \
        --limit 100 --no-wandb

Key ideas (ReARTeR-inspired):
  - KTO loss at step level (not trajectory level)
  - Document masking: <documents>...</documents> excluded from loss
  - Fixed lambda_U (lambda0) and desirable_weight for GOOD/BAD steps
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from prmrag.training.kto_trainer import (
    KTODataPreparer,
    StepLevelKTOTrainer,
    KTODebugCallback,
    EarlyCollapseCallback,
)

HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train policy model with step-level KTO loss"
    )

    # Data — new combined format (preferred)
    parser.add_argument(
        "--input", type=Path, action="append", default=[],
        help="Combined JSONL file(s) with steps + generative scores. "
             "Can be specified multiple times to combine data sources.",
    )

    # Data — legacy format (two separate files)
    parser.add_argument(
        "--trajectories", type=Path, default=None,
        help="[Legacy] Path to judge_labels JSONL (step content)",
    )
    parser.add_argument(
        "--generative-scores", type=Path, default=None,
        help="[Legacy] Path to generative_scores per_trajectory JSONL",
    )

    # Model
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune (can be a local merged model path)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/kto_policy_v1",
        help="Output directory",
    )

    # KTO hyperparameters (KTO paper recommendations)
    parser.add_argument("--beta", type=float, default=0.3, help="KTO temperature ([0.10,1.00] for non-SFT model)")
    parser.add_argument("--lambda0", type=float, default=10.9, help="Weight (lambda_U) for BAD steps; set s.t. λD*nD/(λU*nU)∈[1,1.5]")
    parser.add_argument("--desirable-weight", type=float, default=1.0, help="Weight (lambda_D) for GOOD step loss")

    # Training
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size (KTO needs >= 2)")
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate (KTO: 5e-6 with AdamW)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Data processing
    parser.add_argument(
        "--truncate-after-bad", action="store_true",
        help="Remove steps after first BAD step",
    )
    parser.add_argument(
        "--best-of-n", action="store_true",
        help="Use only best trajectory per question (highest generative_min)",
    )
    parser.add_argument(
        "--filter-no-gold", action="store_true",
        help="Remove trajectories where gold answer not found in any retrieved document",
    )

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="prmrag-kto", help="Wandb project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")

    # Debug / Sweep
    parser.add_argument("--limit", type=int, default=None, help="Limit data for debugging")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (-1=full training)")
    parser.add_argument("--sweep-result-file", type=str, default=None,
                        help="If set, write sweep result JSON here and skip model saving")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine data mode
    use_combined = len(args.input) > 0
    use_legacy = args.trajectories is not None and args.generative_scores is not None

    if not use_combined and not use_legacy:
        print("Error: Provide either --input (combined format) or "
              "--trajectories + --generative-scores (legacy format)")
        sys.exit(1)

    # Validate inputs
    if use_combined:
        for p in args.input:
            if not p.exists():
                print(f"Error: Input not found: {p}")
                sys.exit(1)
    if use_legacy:
        if not args.trajectories.exists():
            print(f"Error: Trajectories not found: {args.trajectories}")
            sys.exit(1)
        if not args.generative_scores.exists():
            print(f"Error: Generative scores not found: {args.generative_scores}")
            sys.exit(1)

    # DDP: determine local rank from environment (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in (-1, 0)

    if is_main_process:
        print("=" * 70)
        print("KTO POLICY TRAINING (Step-level PRM Loss)")
        print("=" * 70)
        print()
    if is_main_process:
        if use_combined:
            print(f"Data mode:       Combined ({len(args.input)} files)")
            for p in args.input:
                print(f"  Input:         {p}")
        else:
            print(f"Data mode:       Legacy (two files)")
            print(f"  Trajectories:  {args.trajectories}")
            print(f"  Generative scores: {args.generative_scores}")
        print(f"Model:           {args.model_name}")
        print(f"Output dir:      {args.output_dir}")
        print(f"Beta:            {args.beta}")
        print(f"Lambda0:         {args.lambda0}")
        print(f"Max length:      {args.max_length}")
        num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        eff_batch = args.batch_size * args.gradient_accumulation * num_gpus
        print(f"Batch size:      {args.batch_size} x {args.gradient_accumulation} accum x {num_gpus} GPUs = {eff_batch} eff")
        print(f"Learning rate:   {args.learning_rate}")
        print(f"LoRA:            r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"Truncate BAD:    {args.truncate_after_bad}")
        print(f"Filter no-gold:  {args.filter_no_gold}")
        print(f"Best-of-N:       {args.best_of_n}")
        print(f"Wandb:           {'Disabled' if args.no_wandb else args.wandb_project}")
        if args.limit:
            print(f"Limit:           {args.limit}")
        print()

    # =========================================================================
    # 1. Load tokenizer
    # =========================================================================
    print("Loading tokenizer...")
    os.environ["HF_HOME"] = HF_CACHE_DIR
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # =========================================================================
    # 2. Prepare dataset
    # =========================================================================
    print("\nPreparing KTO dataset...")
    preparer = KTODataPreparer(tokenizer)
    if use_combined:
        dataset = preparer.prepare_dataset_combined(
            input_paths=[str(p) for p in args.input],
            truncate_after_bad=args.truncate_after_bad,
            filter_no_gold_in_docs=args.filter_no_gold,
            limit=args.limit,
        )
    else:
        dataset = preparer.prepare_dataset(
            trajectories_path=str(args.trajectories),
            generative_scores_path=str(args.generative_scores),
            truncate_after_bad=args.truncate_after_bad,
            best_of_n=args.best_of_n,
            limit=args.limit,
        )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: No training samples produced!")
        sys.exit(1)

    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE TRAINING DATA")
    print("=" * 70)
    sample = dataset[0]
    print(f"Trajectory: {sample['trajectory_id']}")
    print(f"Prompt (last 200 chars): ...{sample['prompt'][-200:]}")
    print(f"Completion (first 300 chars): {sample['completion'][:300]}...")
    annotations = json.loads(sample["step_annotations_json"])
    print(f"\nStep annotations ({len(annotations)} steps):")
    for i, ann in enumerate(annotations):
        label_str = "GOOD" if ann["label"] else "BAD"
        masked = sum(ann["doc_mask"])
        total = len(ann["doc_mask"])
        print(f"  Step {i+1}: {label_str} (score={ann['generative_score']:.4f}), "
              f"tokens={total}, doc_masked={masked}")
    print()

    # Token length distribution
    print("Analyzing token lengths...")
    lengths = []
    for i in range(min(100, len(dataset))):
        s = dataset[i]
        full = s["prompt"] + s["completion"]
        toks = tokenizer.encode(full, add_special_tokens=False)
        lengths.append(len(toks))

    print(f"Token lengths (first {len(lengths)} samples):")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.0f}")
    exceeding = sum(1 for l in lengths if l > args.max_length)
    if exceeding:
        print(f"  Exceeding max_length: {exceeding} ({100*exceeding/len(lengths):.1f}%)")
    print()

    # =========================================================================
    # 3. Load models
    # =========================================================================
    # DDP: load each model directly onto this process's GPU
    # device_map={"": local_rank} → GPU:local_rank (GPU:0 if single-GPU)
    gpu_id = max(local_rank, 0)

    if is_main_process:
        print("Loading policy model (with LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE_DIR,
    )

    # Enable gradient checkpointing (use_reentrant=False required for DDP + LoRA)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if is_main_process:
        model.print_trainable_parameters()

    if is_main_process:
        print("\nLoading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE_DIR,
    )
    ref_model.eval()

    # =========================================================================
    # 4. Setup training
    # =========================================================================
    report_to = []
    if not args.no_wandb and WANDB_AVAILABLE and is_main_process:
        report_to.append("wandb")
        run_name = args.wandb_run_name or f"kto_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "beta": args.beta,
                "lambda0": args.lambda0,
                "desirable_weight": args.desirable_weight,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": args.batch_size * args.gradient_accumulation,
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "num_epochs": args.num_epochs,
                "truncate_after_bad": args.truncate_after_bad,
                "num_samples": len(dataset),
            },
            tags=["kto", "prmrag", "step-level"],
        )
        print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    else:
        report_to.append("tensorboard")

    sweep_mode = args.sweep_result_file is not None

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
        save_steps=args.save_steps if not sweep_mode else 99999,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        save_total_limit=2 if not sweep_mode else 0,
        report_to=report_to,
        remove_unused_columns=False,  # We need custom columns
        dataloader_pin_memory=True,
    )

    # =========================================================================
    # 5. Train
    # =========================================================================
    debug_callback = KTODebugCallback()
    collapse_callback = EarlyCollapseCallback(
        bad_loss_threshold=0.05,
        kl_threshold=-5.0,
        warmup_steps=30,
    )

    trainer = StepLevelKTOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=args.beta,
        lambda0=args.lambda0,
        desirable_weight=args.desirable_weight,
        max_length=args.max_length,
        args=training_args,
        train_dataset=dataset,
        callbacks=[debug_callback, collapse_callback],
    )

    # Check for checkpoint resume
    checkpoint_dir = Path(args.output_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    ) if checkpoint_dir.exists() else []

    resume_checkpoint = None
    if checkpoints:
        resume_checkpoint = str(checkpoints[-1])
        print(f"\nResuming from checkpoint: {resume_checkpoint}")

    print(f"\n{'='*70}")
    print("STARTING KTO TRAINING")
    print(f"{'='*70}\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # =========================================================================
    # 6. Save
    # =========================================================================
    if sweep_mode:
        # Sweep mode: write result JSON, skip model saving
        result = {
            "config": {
                "beta": args.beta,
                "lambda0": args.lambda0,
                "learning_rate": args.learning_rate,
                "desirable_weight": args.desirable_weight,
            },
            **collapse_callback.summary(),
        }
        with open(args.sweep_result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Sweep] Result written to: {args.sweep_result_file}")
        print(f"[Sweep] collapsed={result['collapsed']}, "
              f"final_bad_loss={result.get('final_bad_loss')}, "
              f"final_kl={result.get('final_kl')}")
    else:
        final_path = Path(args.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        info = {
            "model_name": args.model_name,
            "beta": args.beta,
            "lambda0": args.lambda0,
            "desirable_weight": args.desirable_weight,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_length": args.max_length,
            "truncate_after_bad": args.truncate_after_bad,
            "data_mode": "combined" if use_combined else "legacy",
            "input_files": [str(p) for p in args.input] if use_combined else [
                str(args.trajectories), str(args.generative_scores)
            ],
            "num_samples": len(dataset),
            "completed_at": datetime.now().isoformat(),
        }
        with open(Path(args.output_dir) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n{'='*70}")
        print("KTO TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Model saved to: {final_path}")
        print()
        print("To use the trained model:")
        print(f"  from peft import PeftModel")
        print(f"  model = PeftModel.from_pretrained(base_model, '{final_path}')")
        print()

    # Finish wandb
    if not args.no_wandb and WANDB_AVAILABLE and is_main_process and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
