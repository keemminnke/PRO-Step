#!/usr/bin/env python3
"""Train generative model for step-level evaluation.

Usage:
    python scripts/train_generative_model.py \
        --train-data outputs/test_judge_5q_newdata_merged_clean.jsonl \
        --output-dir outputs/generative_model_test \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --num-epochs 1

This script trains a generative model that can:
1. Generate rationale for step evaluation
2. Predict verdict (GOOD/BAD)

Following the loss function:
    L_generative = -E[log P(v_j | x, s_t, v_<j) + λ log P(r | x, s_t, v)]

Where:
    - v: Verdict (GOOD/BAD)
    - r: Rationale (reasoning)
    - x: Question
    - s_t: Current step
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prmrag.training.generative_trainer import (
    create_generative_trainer,
    GenerativeTrainingConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train generative model for step evaluation"
    )

    # Data
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to training data (merged JSONL with judge labels)"
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        help="Path to eval data (optional)"
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/generative_model",
        help="Output directory for checkpoints"
    )

    # Training
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )

    # LoRA
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (lower = smaller updates, less overfitting)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    # Data processing
    parser.add_argument(
        "--truncate-after-bad",
        action="store_true",
        help="Discard steps after first BAD step (PRM-style training)"
    )

    # Debug
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=100,
        help="Interval (steps) for sample prediction debugging"
    )

    # Wandb
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="prmrag-generative",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (auto-generated if not specified)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    args = parser.parse_args()

    # Validate
    if not args.train_data.exists():
        print(f"Error: Training data not found: {args.train_data}")
        sys.exit(1)

    print("=" * 70)
    print("GENERATIVE MODEL TRAINING")
    print("=" * 70)
    print()
    print(f"Training data:  {args.train_data}")
    if args.eval_data:
        print(f"Eval data:      {args.eval_data}")
    print(f"Model:          {args.model_name}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Wandb:          {'Disabled' if args.no_wandb else f'{args.wandb_project}'}")
    print()

    # Create config (unused, using create_generative_trainer instead)
    # config = GenerativeTrainingConfig(...)

    # Create trainer
    trainer = create_generative_trainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Prepare data
    train_dataset, eval_dataset = trainer.prepare_data(
        train_file=args.train_data,
        eval_file=args.eval_data,
        truncate_after_bad=args.truncate_after_bad,
    )

    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    print()

    # Count labels (1 = GOOD, 0 = BAD)
    label_counts = {1: 0, 0: 0, -1: 0}  # 1=GOOD, 0=BAD, -1=OTHER
    for sample in train_dataset:
        label = sample.get('label', -1)
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[-1] += 1

    print(f"Total samples: {len(train_dataset)}")
    print(f"Label distribution (1=GOOD, 0=BAD):")
    print(f"  - 1 (GOOD): {label_counts[1]} ({100*label_counts[1]/len(train_dataset):.1f}%)")
    print(f"  - 0 (BAD): {label_counts[0]} ({100*label_counts[0]/len(train_dataset):.1f}%)")
    if label_counts[-1] > 0:
        print(f"  - OTHER: {label_counts[-1]}")
    print()

    # Show detailed sample (full content, not truncated)
    print("=" * 70)
    print("SAMPLE TRAINING EXAMPLE (FULL)")
    print("=" * 70)
    sample_messages = train_dataset[0]['messages']
    for msg in sample_messages:
        print(f"\n[{msg['role'].upper()}]:")
        print("-" * 50)
        print(msg['content'])
        print("-" * 50)

    # Show second sample if available (different label if possible)
    if len(train_dataset) > 1:
        # Find a sample with different label
        first_label = train_dataset[0].get('label')
        second_idx = 1
        for i, sample in enumerate(train_dataset):
            if sample.get('label') != first_label:
                second_idx = i
                break

        print("\n" + "=" * 70)
        print(f"SECOND SAMPLE (label={train_dataset[second_idx].get('label')})")
        print("=" * 70)
        sample_messages = train_dataset[second_idx]['messages']
        for msg in sample_messages:
            print(f"\n[{msg['role'].upper()}]:")
            print("-" * 50)
            print(msg['content'])
            print("-" * 50)

    # Tokenize and show token counts for first sample
    print("\n" + "=" * 70)
    print("TOKEN ANALYSIS")
    print("=" * 70)
    sample_text = trainer.tokenizer.apply_chat_template(
        train_dataset[0]['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    tokens = trainer.tokenizer.encode(sample_text)
    print(f"Sample 1 token count: {len(tokens)}")
    print(f"Max sequence length: {args.max_length}")
    if len(tokens) > args.max_length:
        print(f"WARNING: Sample exceeds max_length by {len(tokens) - args.max_length} tokens!")

    # Check a few more samples for token length distribution
    token_lengths = []
    for i in range(min(100, len(train_dataset))):
        text = trainer.tokenizer.apply_chat_template(
            train_dataset[i]['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        token_lengths.append(len(trainer.tokenizer.encode(text)))

    print(f"\nToken length stats (first {len(token_lengths)} samples):")
    print(f"  Min: {min(token_lengths)}")
    print(f"  Max: {max(token_lengths)}")
    print(f"  Avg: {sum(token_lengths)/len(token_lengths):.0f}")
    exceeding = sum(1 for l in token_lengths if l > args.max_length)
    if exceeding > 0:
        print(f"  Exceeding max_length: {exceeding} ({100*exceeding/len(token_lengths):.1f}%)")
    print()

    # Train with debugging
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        debug_interval=args.debug_interval,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print()
    print(f"Model saved to: {args.output_dir}/final_model")
    print()
    print("To use the trained model:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{args.output_dir}/final_model')")
    print()


if __name__ == "__main__":
    main()
