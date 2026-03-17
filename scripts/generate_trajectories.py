#!/usr/bin/env python3
"""Stage 1: Generate trajectories only (no Judge evaluation).

Generate N trajectories per question using Policy model.
Supports both KILT corpus (BGE-M3 + Reranker) and HotpotQA (BM25).

Usage:
    # KILT corpus with BGE-M3 + Reranker (fully GPU accelerated)
    python scripts/generate_trajectories.py \
        --data_path data/raw/questions/hotpotqa_train.jsonl \
        --output_path outputs/trajectories.jsonl \
        --num_samples 16 \
        --batch_size 50

    # HotpotQA validation with BM25 only (no rerank)
    python scripts/generate_trajectories.py \
        --data_path data/hotpot_dev_distractor_v1.json \
        --output_path outputs/trajectories_hotpot.jsonl \
        --num_samples 8 \
        --use_hotpotqa_corpus \
        --no_rerank
"""

import sys
import os
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.generation.adaptive_generator import SimpleTrajectoryGenerator
from prmrag.retrieval.bge_retriever import BGERetriever
from prmrag.retrieval.bge_reranker import BGEReranker
from prmrag.retrieval import BM25Retriever
from prmrag.retrieval.e5_retriever import E5Retriever


class DenseRetrieverWithReranker:
    """BGE Dense Retriever + Reranker (no BM25, fully GPU-accelerated)."""

    def __init__(self, bge_retriever, reranker, top_k_candidates: int = 50):
        self.bge = bge_retriever
        self.reranker = reranker
        self.top_k_candidates = top_k_candidates
        print(f"✓ Dense retriever initialized (BGE + Reranker)")

    def retrieve(self, query: str, top_k: int = 5):
        """Single query retrieval."""
        # Get candidates from BGE
        candidates = self.bge.retrieve(query, top_k=self.top_k_candidates)
        # Rerank
        if self.reranker:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    def batch_retrieve(self, queries, top_k: int = 5):
        """Batch retrieval - fully GPU accelerated."""
        if not queries:
            return []

        # Batch BGE retrieval (GPU)
        all_candidates = self.bge.batch_retrieve(queries, top_k=self.top_k_candidates)

        # Batch rerank (GPU)
        if self.reranker and hasattr(self.reranker, 'batch_rerank'):
            return self.reranker.batch_rerank(queries, all_candidates, top_k=top_k)
        elif self.reranker:
            # Fallback to sequential
            return [self.reranker.rerank(q, c, top_k=top_k)
                    for q, c in zip(queries, all_candidates)]
        else:
            return [c[:top_k] for c in all_candidates]


def build_hotpotqa_corpus(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build corpus from HotpotQA context paragraphs."""
    corpus = []
    seen = set()

    for item in data:
        for title, sentences in item.get('context', []):
            text = ' '.join(sentences)
            key = f"{title}::{text[:100]}"
            if key not in seen:
                corpus.append({
                    'id': f"hotpotqa_{len(corpus)}",
                    'title': title,
                    'text': text,
                })
                seen.add(key)

    print(f"  Built HotpotQA corpus: {len(corpus)} passages")
    return corpus


def get_truncated_text(doc_text_list, max_chars=1200):
    """Character-based truncation with truncation marker."""
    if not doc_text_list:
        return ""
    joined_text = ' '.join(doc_text_list)
    if len(joined_text) > max_chars:
        return joined_text[:max_chars] + " [TRUNCATED]"
    return joined_text


def load_kilt_corpus(corpus_file: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Load KILT Wikipedia corpus with smart truncation. Caches parsed corpus as pickle."""
    corpus_file = Path(corpus_file)
    cache_path = corpus_file.parent / (corpus_file.stem + "_parsed.pkl")

    # Try loading from pickle cache first (much faster than parsing 35GB JSONL)
    if cache_path.exists() and limit is None:
        print(f"Loading cached corpus from {cache_path}...")
        with open(cache_path, 'rb') as f:
            corpus = pickle.load(f)
        print(f"✓ Loaded {len(corpus):,} documents from cache")
        return corpus

    print(f"Loading KILT corpus from {corpus_file}...")

    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            doc = json.loads(line)

            # Support both raw format (_id/wikipedia_title) and pre-processed format (id/title)
            if 'id' in doc:
                doc_id = doc['id']
                title = doc['title']
            else:
                doc_id = doc['_id']
                title = doc['wikipedia_title']

            if isinstance(doc['text'], list):
                text = get_truncated_text(doc['text'], max_chars=1200)
            else:
                text = doc['text'][:1200] + (" [TRUNCATED]" if len(doc['text']) > 1200 else "")

            corpus.append({
                'id': doc_id,
                'title': title,
                'text': text,
            })

            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i+1:,} documents...")

    print(f"✓ Loaded {len(corpus):,} documents")

    # Save pickle cache for future runs (only when loading full corpus)
    if limit is None:
        print(f"Saving corpus cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Corpus cache saved")

    return corpus


def load_questions(data_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load questions from JSON/JSONL file."""
    questions = []

    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'data' in data:
                questions = data['data']
            else:
                questions = [data]

    if limit:
        questions = questions[:limit]

    print(f"Loaded {len(questions)} questions from {data_path}")
    return questions


def save_trajectories(trajectories, output_path: str):
    """Save trajectories to JSONL file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        for traj in trajectories:
            record = {
                'trajectory_id': traj.trajectory_id,
                'question': traj.question,
                'gold_answer': traj.gold_answer,
                'final_answer': traj.final_answer,
                'is_correct': traj.is_correct,
                'supporting_facts': traj.supporting_facts,
                'metadata': traj.metadata,
                'steps': [
                    {
                        'step_id': step.step_id,
                        'step_type': step.step_type.value,
                        'text': step.text,
                        'content': step.content,
                        'used_passages': step.used_passages,
                        'metadata': step.metadata,
                    }
                    for step in traj.steps
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Trajectories (Stage 1)")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to questions JSON/JSONL file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSONL file')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of trajectories per question (default: 16)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of questions per batch (default: 50)')
    parser.add_argument('--limit', type=int, default=500,
                        help='Limit number of questions (default: 500)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file (skip already processed questions)')

    # Model settings
    parser.add_argument('--policy_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Policy model name')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--max_steps', type=int, default=10,
                        help='Maximum steps per trajectory')
    parser.add_argument('--max_tokens_per_step', type=int, default=4096,
                        help='Maximum tokens to generate per step (default: 4096)')

    # Retriever settings
    parser.add_argument('--corpus_limit', type=int, default=None,
                        help='Limit corpus size for testing (default: all)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of passages to retrieve per query')
    parser.add_argument('--rerank_top_n', type=int, default=20,
                        help='Rerank top-N candidates from fusion (default: 20)')
    parser.add_argument('--use_hotpotqa_corpus', action='store_true',
                        help='Use HotpotQA context as corpus instead of KILT')
    parser.add_argument('--no_rerank', action='store_true',
                        help='Disable reranking (use BM25 only for HotpotQA)')
    parser.add_argument('--retriever', type=str, default='bm25',
                        choices=['bm25', 'bge', 'e5'],
                        help='Retriever type for HotpotQA corpus mode (default: bm25)')

    # LoRA adapter
    parser.add_argument('--lora_adapter', type=str, default=None,
                        help='Path to LoRA adapter (e.g., outputs/kto_policy_v1/final_model)')

    # GPU settings
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.88,
                        help='GPU memory utilization for policy model')
    parser.add_argument('--max_model_len', type=int, default=16384,
                        help='Max model context length')

    args = parser.parse_args()

    print("=" * 70)
    print("Stage 1: Trajectory Generation")
    print("=" * 70)
    print(f"Policy model: {args.policy_model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    if args.use_hotpotqa_corpus:
        print(f"Retriever: {args.retriever.upper()} (HotpotQA corpus)")
        if args.retriever == 'bm25' and not args.no_rerank:
            print(f"Reranking: Enabled")
        else:
            print(f"Reranking: Disabled")
    else:
        if args.no_rerank:
            print(f"Retriever: Dense (BGE-M3 only) - No Reranker")
        else:
            print(f"Retriever: Dense (BGE-M3 + Reranker) - Fully GPU accelerated")
    print("=" * 70)

    # Load questions
    questions = load_questions(args.data_path, args.limit)
    total_trajectories = len(questions) * args.num_samples
    print(f"Will generate {total_trajectories} trajectories")

    # Data directory
    data_dir = Path(__file__).parent.parent / "data"

    # [1] Load corpus (KILT or HotpotQA)
    if args.use_hotpotqa_corpus:
        print(f"\n[1/4] Building corpus from HotpotQA context...")
        corpus = build_hotpotqa_corpus(questions)
    else:
        print(f"\n[1/4] Loading KILT corpus...")
        corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
        corpus = load_kilt_corpus(corpus_file, limit=args.corpus_limit)

    # [2] Initialize Retriever
    if args.use_hotpotqa_corpus:
        retriever_type = args.retriever
        print(f"\n[2/4] Initializing {retriever_type.upper()} retriever (HotpotQA mode)...")

        if retriever_type == 'e5':
            retriever = E5Retriever(corpus)
            print(f"  ✓ E5 retriever initialized")
        elif retriever_type == 'bge':
            retriever = BGERetriever(corpus)
            print(f"  ✓ BGE retriever initialized")
        else:
            # BM25 (default)
            retriever = BM25Retriever(corpus)
            print(f"  ✓ BM25 retriever initialized")

        if retriever_type == 'bm25' and not args.no_rerank:
            try:
                print(f"  Initializing reranker...")
                reranker = BGEReranker(device="cuda", batch_size=64)
                retriever = DenseRetrieverWithReranker(
                    bge_retriever=retriever,  # Using BM25 as first stage
                    reranker=reranker,
                    top_k_candidates=args.rerank_top_n,
                )
                print(f"  ✓ Reranker enabled")
            except Exception as e:
                print(f"  ⚠ Reranker not available: {e}")
        elif retriever_type != 'bm25':
            print(f"  ✓ Dense retriever (no reranker)")
    else:
        # KILT mode: Dense (BGE, optionally with Reranker)
        if args.no_rerank:
            print(f"\n[2/4] Initializing Dense retriever (BGE-M3 only)...")
        else:
            print(f"\n[2/4] Initializing Dense retriever (BGE-M3 + Reranker)...")

        # BGE retriever
        embedding_cache = data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"
        print(f"  [2.1] Initializing BGE-M3 retriever...")
        bge_device = None  # None = auto-detect (cuda)
        faiss_index_cache = data_dir / "indexes" / "kilt_wikipedia_bge_m3.faiss"
        bge_retriever = BGERetriever(
            corpus=corpus,
            batch_size=64,
            embedding_cache_path=str(embedding_cache),
            device=bge_device,
            faiss_index_path=str(faiss_index_cache),
        )

        if args.no_rerank:
            # Use BGE retriever directly without reranker
            retriever = bge_retriever
            print(f"  ✓ Reranking disabled (BGE-M3 on CPU, no CUDA conflict)")
        else:
            # BGE Reranker
            print(f"  [2.2] Initializing BGE Reranker...")
            reranker = BGEReranker(device="cuda", batch_size=64)

            # Combine BGE + Reranker (fully GPU accelerated)
            print(f"  [2.3] Combining BGE + Reranker...")
            retriever = DenseRetrieverWithReranker(
                bge_retriever=bge_retriever,
                reranker=reranker,
                top_k_candidates=args.rerank_top_n,
            )

    # [3] Initialize policy model
    # Clean up CUDA context before vLLM fork to avoid deadlock
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  ✓ CUDA context cleaned before vLLM init")

    print(f"\n[3/4] Loading policy model...")
    policy_config = {
        'model_name': args.policy_model,
        'temperature': args.temperature,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
    }
    if args.lora_adapter:
        policy_config['lora_adapter'] = args.lora_adapter
        print(f"  LoRA adapter: {args.lora_adapter}")
    policy_model = load_policy_model(policy_config)
    print("✓ Model loaded")

    # Initialize generator
    generator_config = {
        'max_steps': args.max_steps,
        'top_k_passages': args.top_k,
        'temperature': args.temperature,
        'max_tokens_per_step': args.max_tokens_per_step,
    }
    generator = SimpleTrajectoryGenerator(policy_model, retriever, generator_config)

    # [4] Process in batches with incremental saving
    print(f"\n[4/4] Generating trajectories...")
    total_generated = 0
    total_correct = 0

    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Resume: load already processed question IDs
    processed_question_ids = set()
    if args.resume and os.path.exists(args.output_path):
        print(f"Resume mode: loading existing results from {args.output_path}")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Extract question_id from trajectory_id (format: "{qid}_sample_{n}")
                        traj_id = data.get('trajectory_id', '')
                        if '_sample_' in traj_id:
                            qid = traj_id.rsplit('_sample_', 1)[0]
                            processed_question_ids.add(qid)
                        total_generated += 1
                        if data.get('is_correct'):
                            total_correct += 1
                    except json.JSONDecodeError:
                        continue
        print(f"  Found {len(processed_question_ids)} questions already processed ({total_generated} trajectories)")

    # Filter out already processed questions
    if args.resume:
        questions_to_process = [
            q for q in questions
            if q.get('_id', q.get('id', str(hash(q['question'])))) not in processed_question_ids
        ]
        print(f"  Remaining questions: {len(questions_to_process)}")
    else:
        questions_to_process = questions

    if not questions_to_process:
        print("All questions already processed!")
        return

    num_batches = (len(questions_to_process) + args.batch_size - 1) // args.batch_size

    # Open file in append mode for incremental saving
    file_mode = 'a' if args.resume else 'w'
    with open(args.output_path, file_mode, encoding='utf-8') as f:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(questions_to_process))
            batch_questions = questions_to_process[start_idx:end_idx]

            print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
            print(f"  Questions: {start_idx + 1} ~ {end_idx}")
            print(f"  Trajectories to generate: {len(batch_questions) * args.num_samples}")

            # Generate trajectories
            trajectories = generator.generate_batch(
                batch_questions,
                num_samples=args.num_samples,
                show_progress=True,
            )

            # Save each trajectory immediately
            for traj in trajectories:
                # Parse steps into trajectories_xml.jsonl format
                steps_formatted = []
                for step in traj.steps:
                    step_data = {
                        'step_id': step.step_id,
                        'step_type': step.metadata.get('action', 'unknown'),  # search, answer, reason
                    }

                    # Parse text to extract think, search/answer, documents
                    content = step.text or ''

                    # Extract <think>
                    import re
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    if think_match:
                        step_data['think'] = think_match.group(1).strip()

                    # Extract <search> or <answer>
                    search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
                    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)

                    if search_match:
                        step_data['search'] = search_match.group(1).strip()
                    if answer_match:
                        step_data['answer'] = answer_match.group(1).strip()

                    # Extract <documents>
                    docs_match = re.search(r'<documents>(.*?)</documents>', content, re.DOTALL)
                    if docs_match:
                        step_data['documents'] = docs_match.group(1).strip()

                    steps_formatted.append(step_data)

                record = {
                    'trajectory_id': traj.trajectory_id,
                    'question': traj.question,
                    'gold_answer': traj.gold_answer,
                    'final_answer': traj.final_answer,
                    'is_correct': traj.is_correct,
                    'metadata': traj.metadata,
                    'steps': steps_formatted,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

                if traj.is_correct:
                    total_correct += 1

            # Flush and sync to disk after each batch
            f.flush()
            os.fsync(f.fileno())

            total_generated += len(trajectories)
            accuracy = total_correct / total_generated * 100 if total_generated > 0 else 0
            print(f"  ✓ Saved {len(trajectories)} trajectories (Total: {total_generated}, Accuracy: {accuracy:.1f}%)")

    # Print summary
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Total questions processed: {len(questions_to_process)} (of {len(questions)})")
    print(f"Total trajectories: {total_generated}")
    if total_generated > 0:
        print(f"Correct answers: {total_correct}/{total_generated} ({total_correct/total_generated*100:.1f}%)")
    print(f"\nOutput: {args.output_path}")


if __name__ == '__main__':
    main()
