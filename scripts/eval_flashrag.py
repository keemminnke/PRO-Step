#!/usr/bin/env python3
"""Evaluate policy model using FlashRAG SearchR1Pipeline.

Usage:
    # DPO model
    python scripts/eval_flashrag.py \
        --model outputs/sft_policy_v1/merged_model \
        --lora outputs/dpo_policy_v1/final_model \
        --tag dpo_bva

    # Qwen base
    python scripts/eval_flashrag.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --tag qwen_base
"""

import argparse
import json
import os
import sys

os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE

from flashrag.config import Config
from flashrag.utils import get_dataset, get_retriever, get_generator
from flashrag.pipeline import SearchR1Pipeline
from flashrag.prompt import PromptTemplate


SYSTEM_PROMPT = """You are a helpful assistant who is good at answering questions with multi-turn search engine calling. To answer questions, you must first reason through the available information using <think> and </think>. If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with relevant documents enclosed in <documents> and </documents>. You can search as many times as you want. Once you have sufficient information or if you find no further external knowledge is needed, directly provide your final answer. Ensure your answer is concise, using nouns or short phrases whenever possible. Conclude with: "So the answer is <answer>answer</answer>"."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--lora", type=str, default=None)
    p.add_argument("--tag", type=str, default="eval")

    # Retriever
    p.add_argument("--index-path", type=str,
                   default="data/indexes/bge_Flat_IVF4096.index")
    p.add_argument("--corpus-path", type=str,
                   default="data/kilt/kilt_corpus_flashrag.jsonl")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--retriever-method", type=str, default="bge")
    p.add_argument("--retriever-model", type=str, default="BAAI/bge-base-en-v1.5")

    # Data
    p.add_argument("--data-dir", type=str, default="data/flashrag")
    p.add_argument("--dataset", type=str, default="hotpotqa")
    p.add_argument("--limit", type=int, default=None)

    # Generation
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-retrieval", type=int, default=10)

    # Tokens (customize for different models)
    p.add_argument("--doc-begin", type=str, default="<documents>")
    p.add_argument("--doc-end", type=str, default="</documents>")
    p.add_argument("--query-begin", type=str, default="<search>")
    p.add_argument("--query-end", type=str, default="</search>")

    # Prompt control
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument("--use-default-prompt", action="store_true",
                   help="Use SearchR1Pipeline built-in prompt (for Search-R1 model)")

    # GPU
    p.add_argument("--gpu-util", type=float, default=0.70)
    return p.parse_args()


def resolve_model_path(model_name: str) -> str:
    """Resolve HuggingFace model ID to local snapshot path if needed."""
    if os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "config.json")):
        return model_name
    # Try to resolve via huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(model_name, cache_dir=HF_CACHE)
        print(f"Resolved {model_name} -> {local_path}")
        return local_path
    except Exception:
        return model_name


def main():
    args = parse_args()

    # Resolve model path for FlashRAG (needs local config.json)
    model_path = resolve_model_path(args.model)

    output_dir = f"outputs/flashrag_{args.tag}"
    os.makedirs(output_dir, exist_ok=True)

    config_dict = {
        # Data
        "data_dir": args.data_dir,
        "dataset_name": args.dataset,
        "split": ["test"],
        "test_sample_num": args.limit,
        "random_sample": False,

        # Save
        "save_dir": output_dir,
        "save_intermediate_data": True,
        "save_metric_score": True,

        # Retriever
        "retrieval_method": args.retriever_method,
        "retrieval_model_path": args.retriever_model,
        "index_path": args.index_path,
        "corpus_path": args.corpus_path,
        "retrieval_topk": args.top_k,
        "retrieval_batch_size": 256,
        "retrieval_use_fp16": True,
        "retrieval_query_max_length": 256,
        "retrieval_pooling_method": "mean",
        "faiss_gpu": False,

        # Generator
        "framework": "vllm",
        "generator_model": model_path,
        "generator_model_path": model_path,
        "generator_max_input_len": 32768,
        "gpu_memory_utilization": args.gpu_util,
        "gpu_num": 2,
        "generation_params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 0.95,
        },

        # Metrics
        "metrics": ["em", "f1"],
        "metric_setting": {},

        # Misc
        "seed": 42,
        "gpu_id": "0,1",
    }

    # LoRA
    if args.lora:
        config_dict["generator_lora_path"] = args.lora

    config = Config(config_dict=config_dict)

    print("=" * 70)
    print("FlashRAG Evaluation")
    print("=" * 70)
    print(f"Model:      {model_path}")
    if args.lora:
        print(f"LoRA:       {args.lora}")
    print(f"Retriever:  bge-base-en-v1.5 (top_k={args.top_k})")
    print(f"Dataset:    {args.dataset} ({args.limit} questions)")
    print(f"Temp:       {args.temperature}")
    print(f"Output:     {output_dir}")
    print("=" * 70)

    # Load dataset
    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test questions")

    # Pipeline
    if args.use_default_prompt:
        # Use SearchR1Pipeline built-in prompt (for Search-R1 model)
        prompt_template = None
    else:
        sys_prompt = args.system_prompt if args.system_prompt else SYSTEM_PROMPT
        prompt_template = PromptTemplate(
            config=config,
            system_prompt=sys_prompt,
            user_prompt="Question: {question}\n",
        )

    pipeline = SearchR1Pipeline(
        config=config,
        prompt_template=prompt_template,
        max_retrieval_num=args.max_retrieval,
        begin_of_query_token=args.query_begin,
        end_of_query_token=args.query_end,
        begin_of_documents_token=args.doc_begin,
        end_of_documents_token=args.doc_end,
        begin_of_answer_token="<answer>",
        end_of_answer_token="</answer>",
    )

    if hasattr(pipeline.retriever.index, "nprobe"):
        pipeline.retriever.index.nprobe = 128
        print(f"[retriever] Set IVF nprobe = {pipeline.retriever.index.nprobe}")

    # Run
    result_dataset = pipeline.run(test_data, do_eval=True)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
