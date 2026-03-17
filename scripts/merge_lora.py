#!/usr/bin/env python3
"""Merge LoRA adapters into base model.

Usage:
    python scripts/merge_lora.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --lora-path outputs/sft_policy_v3_5000q/final_model \
        --output-dir outputs/sft_policy_v3_5000q/merged_model
"""

import argparse
import os

HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HF_CACHE_DIR

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, cache_dir=HF_CACHE_DIR
    )

    print(f"Loading LoRA: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
