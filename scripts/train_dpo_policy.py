#!/usr/bin/env python3
"""Train policy model with TRL official DPOTrainer + document masking.

Usage:
    # Small test
    torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
        --dpo-dataset outputs/dpo_dataset_bva.jsonl \
        --output-dir outputs/dpo_policy_test \
        --limit 50 --no-wandb

    # Full training (MCTS DPO with document masking)
    torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
        --dpo-dataset outputs/mcts_dpo_5000q.jsonl \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --output-dir outputs/dpo_policy_mcts_v1 \
        --num-epochs 1 --wandb-run-name dpo_mcts_v1

Key design:
  - Uses TRL DPOTrainer with document masking
  - <documents>...</documents> spans in completion are masked (label=-100)
    because documents are retriever output, not model-generated
  - ref_model=None: TRL disables LoRA adapters for reference forward pass
  - peft_config passed to DPOTrainer (TRL handles LoRA application)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

HF_CACHE_DIR = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

# Qwen2.5 token IDs for <documents> / </documents> boundary detection
_OPEN_LT = 27      # '<'
_CLOSE_LT = 522    # '</'
_DOC_TOKEN = 50778  # 'documents'


class DocMaskDPOTrainer(DPOTrainer):
    """DPOTrainer that masks <documents>...</documents> in completion_mask.

    Documents are retriever output (environment), not model-generated text.
    Masking them prevents the policy from being trained to produce documents.
    """

    _mask_log_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "completion_mask" in inputs and "input_ids" in inputs:
            before_active = inputs["completion_mask"].sum().item()
            masked_count = self._mask_document_spans(inputs)
            after_active = inputs["completion_mask"].sum().item()
            DocMaskDPOTrainer._mask_log_counter += 1
            if DocMaskDPOTrainer._mask_log_counter <= 3 or DocMaskDPOTrainer._mask_log_counter % 200 == 0:
                print(f"[DocMask] step={DocMaskDPOTrainer._mask_log_counter} "
                      f"before_active={before_active} after_active={after_active} "
                      f"masked={masked_count} shape={inputs['input_ids'].shape}")
        return super().compute_loss(
            model, inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    @staticmethod
    def _mask_document_spans(inputs):
        """Zero out completion_mask for <documents>...</documents> spans."""
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        batch_size, seq_len = input_ids.shape
        masked_total = 0

        for i in range(batch_size):
            ids = input_ids[i].tolist()
            in_doc = False
            j = 0
            while j < seq_len:
                # Detect <documents> start: token '<' (27) followed by 'documents' (50778)
                if (not in_doc and j + 1 < seq_len
                        and ids[j] == _OPEN_LT and ids[j + 1] == _DOC_TOKEN):
                    in_doc = True

                if in_doc:
                    completion_mask[i, j] = 0
                    masked_total += 1
                    # Detect </documents> end: '</' (522) followed by 'documents' (50778)
                    if (j + 1 < seq_len
                            and ids[j] == _CLOSE_LT and ids[j + 1] == _DOC_TOKEN):
                        # Mask '</documents>' tokens: </ + documents + >
                        completion_mask[i, j] = 0
                        if j + 1 < seq_len:
                            completion_mask[i, j + 1] = 0
                            masked_total += 1
                        if j + 2 < seq_len:
                            completion_mask[i, j + 2] = 0
                            masked_total += 1
                        j += 3
                        in_doc = False
                        continue
                j += 1

        return masked_total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train policy model with TRL DPOTrainer"
    )

    # Data
    parser.add_argument(
        "--dpo-dataset", type=Path, required=True,
        help="Pre-built DPO dataset JSONL (prompt/chosen/rejected).",
    )

    # Model
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune",

    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/dpo_policy_v1",
        help="Output directory",
    )


    # DPO
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta temperature")
    parser.add_argument(
        "--loss-type", type=str, default="sigmoid",
        choices=["sigmoid", "ipo", "hinge", "robust"],
        help="DPO loss type (sigmoid = standard DPO)",
    )

    # Training
    parser.add_argument("--max-length", type=int, default=8192, help="Max full sequence length")
    parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="prmrag-dpo", help="Wandb project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")

    # Debug
    parser.add_argument("--limit", type=int, default=None, help="Limit data for debugging")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save checkpoint every N steps")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve relative model path to absolute (prevents HF Hub lookup)
    if os.path.exists(args.model_name):
        args.model_name = os.path.abspath(args.model_name)

    if not args.dpo_dataset.exists():
        print(f"Error: Dataset not found: {args.dpo_dataset}")
        sys.exit(1)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["HF_HOME"] = HF_CACHE_DIR

    if local_rank == 0:
        print("=" * 70)
        print("DPO POLICY TRAINING (TRL official DPOTrainer)")
        print("=" * 70)
        print(f"Dataset:         {args.dpo_dataset}")
        print(f"Model:           {args.model_name}")
        print(f"Output dir:      {args.output_dir}")
        print(f"Beta:            {args.beta}  |  Loss: {args.loss_type}")
        print(f"Max length:      {args.max_length}")
        print(f"Batch size:      {args.batch_size} x {args.gradient_accumulation} accum")
        print(f"Learning rate:   {args.learning_rate}")
        print(f"LoRA:            r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"Wandb:           {'Disabled' if args.no_wandb else args.wandb_project}")
        if args.limit:
            print(f"Limit:           {args.limit}")
        print()

    # =========================================================================
    # 1. Load tokenizer
    # =========================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # =========================================================================
    # 2. Load dataset
    # =========================================================================
    records = []
    with open(args.dpo_dataset) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]

    # is_conversational() 체크: prompt가 list[dict] 이면 messages format으로 인식
    # → TRL이 apply_chat_template으로 처리 (tokenization boundary 문제 없음)
    sample = records[0]
    if isinstance(sample["prompt"], list):
        # messages format: prompt/chosen/rejected 모두 list[dict]
        dataset = HFDataset.from_list([
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in records
        ])
    else:
        # text format (fallback)
        dataset = HFDataset.from_list([
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in records
        ])

    if local_rank == 0:
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"\nSample prompt (last 100 chars): ...{sample['prompt'][-100:]}")
        print(f"Sample chosen (first 200 chars): {sample['chosen'][:200]}...")
        print()

    # =========================================================================
    # 3. Load model
    # =========================================================================
    if local_rank == 0:
        print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE_DIR,
    )

    # =========================================================================
    # 4. Setup wandb
    # =========================================================================
    report_to = []
    if not args.no_wandb and WANDB_AVAILABLE:
        report_to.append("wandb")
        if local_rank == 0:
            run_name = args.wandb_run_name or f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_name": args.model_name,
                    "method": "dpo_trl",
                    "beta": args.beta,
                    "loss_type": args.loss_type,
                    "dataset": str(args.dpo_dataset),
                    "num_pairs": len(dataset),
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "effective_batch_size": args.batch_size * args.gradient_accumulation * 2,
                    "max_length": args.max_length,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "num_epochs": args.num_epochs,
                },
                tags=["dpo", "trl", "prmrag"],
            )
            print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    else:
        report_to.append("tensorboard")

    # =========================================================================
    # 5. DPOConfig
    # =========================================================================
    training_args = DPOConfig(
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
        save_total_limit=2,
        report_to=report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # DPO-specific
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
    )

    # =========================================================================
    # 6. LoRA config
    # =========================================================================
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

    # =========================================================================
    # 7. DPOTrainer with document masking
    #   ref_model=None: TRL uses adapter-disabled model as reference
    #   (no extra GPU memory for a separate ref model)
    #   DocMaskDPOTrainer: masks <documents>...</documents> in loss
    # =========================================================================
    trainer = DocMaskDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Resume from checkpoint if available
    checkpoint_dir = Path(args.output_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    ) if checkpoint_dir.exists() else []

    resume_checkpoint = None
    if checkpoints:
        resume_checkpoint = str(checkpoints[-1])
        if local_rank == 0:
            print(f"Resuming from checkpoint: {resume_checkpoint}")

    if local_rank == 0:
        print(f"\n{'='*70}")
        print("STARTING DPO TRAINING")
        print(f"{'='*70}\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # =========================================================================
    # 8. Save
    # =========================================================================
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_path))
    if local_rank == 0:
        tokenizer.save_pretrained(str(final_path))

        info = {
            "model_name": args.model_name,
            "method": "dpo_trl_doc_masked",
            "beta": args.beta,
            "loss_type": args.loss_type,
            "dataset": str(args.dpo_dataset),
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_length": args.max_length,
            "num_pairs": len(dataset),
            "document_masking": True,
            "completed_at": datetime.now().isoformat(),
        }
        with open(Path(args.output_dir) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n{'='*70}")
        print("DPO TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Model saved to: {final_path}")

        if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
