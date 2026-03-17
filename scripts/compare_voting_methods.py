#!/usr/bin/env python3
"""Compare three voting methods on pre-generated trajectories:
1. Our Generative Model - step-level evaluation, select best trajectory
2. Majority Voting - most common final answer
3. VersaPRM - process reward model scoring

Note: This script only compares voting methods on existing trajectories.
      Use generate_trajectories.py to generate trajectories first.

Usage:
    # Generate trajectories first (with BM25 only, no rerank)
    python scripts/generate_trajectories.py \
        --data_path data/hotpot_dev_distractor_v1.json \
        --output_path outputs/trajectories_hotpot.jsonl \
        --num_samples 8 \
        --use_hotpotqa_corpus \
        --no_rerank

    # Then compare voting methods
    python scripts/compare_voting_methods.py \
        --trajectories outputs/trajectories_hotpot.jsonl \
        --generative-model outputs/generative_model_v1/final_model \
        --output outputs/voting_comparison.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================
# Method 4: Qwen2.5-Math-PRM
# ============================================================

def load_mathprm(model_id: str = "Qwen/Qwen2.5-Math-PRM-7B"):
    """Load Qwen2.5-Math-PRM-7B model."""
    print(f"Loading MathPRM from {model_id}...")

    download_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=download_dir
    )
    model = AutoModel.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=download_dir,
    ).eval()

    print("✓ MathPRM loaded")
    return model, tokenizer


def make_step_rewards(logits, token_masks):
    """Extract per-step reward scores from MathPRM logits."""
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)

    all_scores = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        all_scores.append(positive_probs.cpu().tolist())
    return all_scores


def batch_get_mathprm_scores(
    model, tokenizer, trajectories: List[Dict], batch_size: int = 4
) -> List[List[float]]:
    """Batch get MathPRM step-level scores for multiple trajectories.

    Uses <extra_0> token as step separator. Each step's positive probability
    is extracted at <extra_0> positions.
    """
    step_sep = "<extra_0>"
    step_sep_id = tokenizer.encode(step_sep, add_special_tokens=False)[0]

    # Format all inputs
    all_messages = []
    for traj in trajectories:
        question = traj['question']
        steps = traj.get('steps', [])

        # Build step texts with <extra_0> separator
        step_texts = []
        for i, step in enumerate(steps):
            parts = []
            if step.get('think'):
                parts.append(f"<think>{step['think'][:500]}</think>")
            if step.get('search'):
                parts.append(f"<search>{step['search']}</search>")
            elif step.get('answer'):
                parts.append(f"<answer>{step['answer']}</answer>")
            if step.get('documents'):
                parts.append(f"<documents>{step['documents'][:300]}</documents>")
            step_texts.append(f"Step {i+1}: " + " ".join(parts))

        # Join with <extra_0> (MathPRM step separator)
        response_text = (step_sep).join(step_texts) + step_sep

        messages = [
            {"role": "system", "content": "You are evaluating a multi-step reasoning process."},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": response_text},
        ]
        all_messages.append(messages)

    # Batch process
    all_step_scores = []
    num_batches = (len(all_messages) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(all_messages), batch_size), total=num_batches, desc="MathPRM eval"):
        batch_messages = all_messages[i:i+batch_size]

        # Apply chat template
        batch_texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch_messages
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            logits = outputs[0]

            token_masks = (inputs['input_ids'] == step_sep_id).to(logits.device)
            step_rewards = make_step_rewards(logits, token_masks)

            for j in range(len(batch_texts)):
                scores = step_rewards[j] if j < len(step_rewards) else []
                if not scores:
                    # Fallback: use 0.5 for each step
                    num_steps = len(all_messages[i + j][2]['content'].split(step_sep)) - 1
                    scores = [0.5] * max(num_steps, 1)
                all_step_scores.append(scores)

    return all_step_scores


# ============================================================
# Method 1: Our Generative Model (vLLM) - XML format + binary labels
# ============================================================

def load_generative_model_vllm(generative_path: str, base_model: str, gpu_memory_utilization: float = 0.9):
    """Load our trained generative model with vLLM for fast inference."""
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    print(f"Loading Generative model with vLLM...")
    print(f"  Base model: {base_model}")
    print(f"  LoRA adapter: {generative_path}")

    # Download dir for model cache
    download_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=download_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load vLLM with LoRA support
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=download_dir,
    )

    # Create LoRA request
    lora_request = LoRARequest("generative", 1, generative_path)

    print("✓ Generative model loaded with vLLM")
    return llm, tokenizer, lora_request


def parse_step_content_xml(step: dict) -> dict:
    """Parse step content from XML format."""
    return {
        'think': step.get('think', ''),
        'search': step.get('search', ''),
        'answer': step.get('answer', ''),
        'documents': step.get('documents', ''),
        'step_type': step.get('step_type', 'unknown'),
    }


def build_generative_prompt_xml(tokenizer, question: str, steps: List[Dict], step_idx: int) -> str:
    """Build prompt for generative evaluation using XML format and binary labels (1/0)."""
    system_content = (
        "You are a step-level generative for evaluating reasoning quality in multi-hop question answering. "
        "The trajectory uses XML tags: <think> for reasoning, <search> for queries, <answer> for final answers, <documents> for retrieved passages. "
        "Analyze each step's logical soundness and evidence grounding. "
        "First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."
    )

    input_parts = [f"Question: {question}", ""]

    # Previous steps (NO truncation — match training format)
    if step_idx > 0:
        input_parts.append("Previous Steps:")
        for j, prev in enumerate(steps[:step_idx]):
            parsed = parse_step_content_xml(prev)
            input_parts.append(f"## Step {j+1}")
            if parsed['think']:
                input_parts.append(f"<think>{parsed['think']}</think>")
            if parsed['search']:
                input_parts.append(f"<search>{parsed['search']}</search>")
            elif parsed['answer']:
                input_parts.append(f"<answer>{parsed['answer']}</answer>")
            if parsed['documents']:
                input_parts.append(f"<documents>{parsed['documents']}</documents>")
            input_parts.append("")

    # Current step (NO truncation — match training format)
    parsed = parse_step_content_xml(steps[step_idx])
    input_parts.append("Current Step to Evaluate:")
    input_parts.append(f"## Step {step_idx + 1}")
    if parsed['think']:
        input_parts.append(f"<think>{parsed['think']}</think>")
    if parsed['search']:
        input_parts.append(f"<search>{parsed['search']}</search>")
    elif parsed['answer']:
        input_parts.append(f"<answer>{parsed['answer']}</answer>")
    if parsed['documents']:
        input_parts.append(f"<documents>{parsed['documents']}</documents>")
    input_parts.append("")
    input_parts.append("Task: Evaluate the quality of the Current Step. Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")

    user_content = "\n".join(input_parts)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_soft_score(logprobs_list, token_id_1: int, token_id_0: int, binary_label: int) -> float:
    """Extract soft score from logprobs at the label token position.

    Searches from the end of logprobs for the position containing "1" or "0" token,
    then computes P(good) = P("1") / (P("1") + P("0")) using softmax.
    """
    import math

    # Search from the end for the position with label tokens
    search_start = len(logprobs_list) - 1
    search_end = max(len(logprobs_list) - 5, -1)

    for i in range(search_start, search_end, -1):
        if i < 0:
            break
        token_logprobs = logprobs_list[i]
        if token_logprobs is None:
            continue

        logp_1 = None
        logp_0 = None

        for tid, lp in token_logprobs.items():
            if tid == token_id_1:
                logp_1 = lp.logprob
            elif tid == token_id_0:
                logp_0 = lp.logprob

        if logp_1 is not None or logp_0 is not None:
            if logp_1 is not None and logp_0 is not None:
                # Both available - normalize between "1" and "0"
                p_1 = math.exp(logp_1)
                p_0 = math.exp(logp_0)
                return p_1 / (p_1 + p_0)
            elif logp_1 is not None:
                # "0" not in top logprobs → very high confidence good
                return 1.0 - 1e-6
            elif logp_0 is not None:
                # "1" not in top logprobs → very high confidence bad
                return 1e-6

    # Fallback to binary
    return 1.0 if binary_label == 1 else (0.0 if binary_label == 0 else 0.5)


def aggregate_step_scores(step_scores: List[float], method: str) -> float:
    """Aggregate step-level scores into a trajectory-level score."""
    if not step_scores:
        return 0.0
    if method == "min":
        return min(step_scores)
    elif method == "product":
        score = 1.0
        for s in step_scores:
            score *= s
        return score
    else:  # "avg"
        return sum(step_scores) / len(step_scores)


def batch_evaluate_with_generative_vllm(
    llm, tokenizer, lora_request, trajectories: List[Dict],
    use_soft_scores: bool = True,
    batch_size: int = 500,
    output_jsonl_path: str = None,
) -> List[Tuple[List[int], List[float], List[str]]]:
    """Batch evaluate multiple trajectories with generative using vLLM in chunks.
    
    If output_jsonl_path is provided, it will resume from existing results and save 
    new results incrementally.
    """
    from vllm import SamplingParams

    # Get token IDs for soft scoring
    if use_soft_scores:
        token_id_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
        token_id_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
        print(f"  Soft scoring: token_id('1')={token_id_1}, token_id('0')={token_id_0}")

    # Load existing results for resuming
    existing_results = {}
    if output_jsonl_path and Path(output_jsonl_path).exists():
        print(f"  Resuming from existing results: {output_jsonl_path}")
        with open(output_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_results[rec['trajectory_id']] = rec

    # Final results storage
    final_results = [None] * len(trajectories)
    
    # Collect indices of trajectories that need processing
    indices_to_process = []
    for idx, traj in enumerate(trajectories):
        tid = traj.get('trajectory_id', '')
        if tid in existing_results:
            rec = existing_results[tid]
            final_results[idx] = (rec['generative_step_labels'], rec['generative_step_scores'], [s.get('generative_reasoning', '') for s in rec.get('steps', [])])
        else:
            indices_to_process.append(idx)

    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Already processed: {len(existing_results)}")
    print(f"  Remaining to process: {len(indices_to_process)}")

    if not indices_to_process:
        return final_results

    # vLLM batch generate
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        logprobs=20 if use_soft_scores else None,
        stop=["Label: 1", "Label: 0", "Label:1", "Label:0"],
        include_stop_str_in_output=True,
    )

    # Process in chunks
    pbar = tqdm(total=len(indices_to_process), desc="Generative eval")
    
    # Open file for incremental writing
    jsonl_f = open(output_jsonl_path, 'a' if existing_results else 'w')

    for i in range(0, len(indices_to_process), batch_size):
        chunk_indices = indices_to_process[i:i + batch_size]
        chunk_prompts = []
        chunk_info = [] # (idx, step_idx)

        for idx in chunk_indices:
            traj = trajectories[idx]
            question = traj['question']
            steps = traj.get('steps', [])
            for step_idx in range(len(steps)):
                prompt = build_generative_prompt_xml(tokenizer, question, steps, step_idx)
                chunk_prompts.append(prompt)
                chunk_info.append((idx, step_idx))

        if not chunk_prompts:
            pbar.update(len(chunk_indices))
            continue

        outputs = llm.generate(
            chunk_prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False
        )

        # Temporary storage for this chunk's step-level results
        chunk_results = {idx: {'labels': [], 'scores': [], 'reasonings': []} for idx in chunk_indices}
        
        for (idx, step_idx), output_obj in zip(chunk_info, outputs):
            response = output_obj.outputs[0]
            text = response.text

            label = -1
            if "Label: 1" in text or "Label:1" in text or text.strip().endswith("1"):
                label = 1
            elif "Label: 0" in text or "Label:0" in text or text.strip().endswith("0"):
                label = 0

            if use_soft_scores and response.logprobs:
                score = _extract_soft_score(response.logprobs, token_id_1, token_id_0, label)
            else:
                score = 1.0 if label == 1 else (0.0 if label == 0 else 0.5)

            clean_reasoning = text
            if "[REASONING]" in clean_reasoning and "[/REASONING]" in clean_reasoning:
                clean_reasoning = clean_reasoning.split("[REASONING]")[1].split("[/REASONING]")[0]
            else:
                clean_reasoning = clean_reasoning.replace("[REASONING]", "").replace("[/REASONING]", "")
                if "Label:" in clean_reasoning:
                    clean_reasoning = clean_reasoning.split("Label:")[0]
            clean_reasoning = clean_reasoning.strip()

            chunk_results[idx]['labels'].append(label)
            chunk_results[idx]['scores'].append(score)
            chunk_results[idx]['reasonings'].append(clean_reasoning)

        # Save and update final_results
        for idx in chunk_indices:
            traj = trajectories[idx]
            labels = chunk_results[idx]['labels']
            scores = chunk_results[idx]['scores']
            reasonings = chunk_results[idx]['reasonings']
            
            final_results[idx] = (labels, scores, reasonings)
            
            # Save incremental JSONL
            merged_steps = []
            for j, (l, s, r) in enumerate(zip(labels, scores, reasonings)):
                if j < len(traj.get('steps', [])):
                    new_step = traj['steps'][j].copy()
                    new_step.update({'generative_label': l, 'generative_score': s, 'generative_reasoning': r})
                    merged_steps.append(new_step)
            
            jsonl_rec = {
                'trajectory_id': traj.get('trajectory_id', ''),
                'question': traj.get('question', ''),
                'gold_answer': traj.get('gold_answer', ''),
                'predicted_answer': traj.get('final_answer', traj.get('predicted_answer', '')),
                'is_correct': traj.get('is_correct', False),
                'generative_step_labels': labels,
                'generative_step_scores': scores,
                'generative_min': min(scores) if scores else 0.0,
                'steps': merged_steps,
            }
            jsonl_f.write(json.dumps(jsonl_rec, ensure_ascii=False) + '\n')
            jsonl_f.flush() # Ensure it's saved to disk
            
        pbar.update(len(chunk_indices))

    jsonl_f.close()
    pbar.close()
    return final_results


# ============================================================
# Method 2: Majority Voting
# ============================================================

def majority_vote(trajectories: List[Dict]) -> Tuple[str, int]:
    """Simple majority voting on final answers."""
    answers = []
    for traj in trajectories:
        answer = traj.get('final_answer', traj.get('predicted_answer', ''))
        if answer:
            answers.append(answer.strip().lower())

    if not answers:
        return "", 0

    counter = Counter(answers)
    best_answer, count = counter.most_common(1)[0]

    # Return original case
    for traj in trajectories:
        ans = traj.get('final_answer', traj.get('predicted_answer', ''))
        if ans and ans.strip().lower() == best_answer:
            return ans, count

    return best_answer, count


# ============================================================
# Method 3: VersaPRM
# ============================================================

def load_versaprm(model_id: str = "UW-Madison-Lee-Lab/VersaPRM-Base-8B"):
    """Load VersaPRM model."""
    print(f"Loading VersaPRM from {model_id}...")

    download_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=download_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=download_dir,
    )
    model.eval()

    print("✓ VersaPRM loaded")
    return model, tokenizer


def batch_get_versaprm_scores(
    model, tokenizer, trajectories: List[Dict], device, batch_size: int = 8
) -> List[List[float]]:
    """Batch get VersaPRM step-level scores for multiple trajectories.

    Returns:
        List of step_scores lists, one per trajectory.
        Caller should use aggregate_step_scores() to get trajectory-level scores.
    """
    candidate_tokens = [12, 10]
    step_token_id = 23535

    # Format all inputs (XML format)
    all_texts = []
    for traj in trajectories:
        question = traj['question']
        steps = traj.get('steps', [])

        step_texts = []
        for i, step in enumerate(steps):
            # Build step text from XML fields
            parts = []
            if step.get('think'):
                parts.append(f"<think>{step['think'][:300]}</think>")
            if step.get('search'):
                parts.append(f"<search>{step['search']}</search>")
            elif step.get('answer'):
                parts.append(f"<answer>{step['answer']}</answer>")
            if step.get('documents'):
                parts.append(f"<documents>{step['documents'][:200]}</documents>")
            step_texts.append(f"Step {i+1}: " + " ".join(parts))

        step_separator = ' \n\n\n\n'
        input_text = f"Question: {question} \n\n" + step_separator.join(step_texts) + step_separator
        all_texts.append(input_text)

    # Batch process
    all_step_scores = []
    num_batches = (len(all_texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_texts), batch_size), total=num_batches, desc="VersaPRM eval"):
        batch_texts = all_texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(device)

        with torch.no_grad():
            logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 1]

            for j in range(len(batch_texts)):
                input_ids = inputs['input_ids'][j]
                item_scores = scores[j]

                step_mask = (input_ids == step_token_id)
                if step_mask.sum() > 0:
                    step_scores = item_scores[step_mask].tolist()
                else:
                    valid_len = inputs['attention_mask'][j].sum().item()
                    step_scores = [item_scores[valid_len - 1].item()]

                all_step_scores.append(step_scores)

    return all_step_scores


# ============================================================
# Main Comparison
# ============================================================

def group_trajectories_by_question(trajectories: List[Dict]) -> Dict[str, List[Dict]]:
    """Group trajectories by question."""
    groups = {}
    for traj in trajectories:
        traj_id = traj.get('trajectory_id', '')
        if '_sample_' in traj_id:
            qid = traj_id.rsplit('_sample_', 1)[0]
        else:
            qid = traj_id or traj.get('question', '')[:50]

        if qid not in groups:
            groups[qid] = []
        groups[qid].append(traj)

    return groups


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold (Cover Exact Match).

    Cover EM: ground truth answer is contained in the predicted answer.
    """
    if not predicted or not gold:
        return False
    pred_norm = predicted.strip().lower()
    gold_norm = gold.strip().lower()
    return gold_norm in pred_norm


def normalize_answer(s: str) -> str:
    """Normalize answer for token-level F1 (standard QA evaluation)."""
    import re
    import string
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


def compute_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold answer."""
    if not predicted or not gold:
        return 0.0
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(description="Compare voting methods on pre-generated trajectories")

    # Input (required)
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to trajectories JSONL file (use generate_trajectories.py first)")

    # Models
    parser.add_argument("--generative-model", type=str, default="outputs/generative_model_v1/final_model",
                        help="Path to our generative model")
    parser.add_argument("--generative-base", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                        help="Base model for generative")
    parser.add_argument("--versaprm", type=str, default="UW-Madison-Lee-Lab/VersaPRM-Base-8B",
                        help="VersaPRM model ID")
    parser.add_argument("--mathprm", type=str, default="Qwen/Qwen2.5-Math-PRM-7B",
                        help="MathPRM model ID")

    # Output
    parser.add_argument("--output", type=str, default="outputs/voting_comparison.json",
                        help="Output path for results")

    # Skip options
    parser.add_argument("--skip-generative", action="store_true",
                        help="Skip generative evaluation")
    parser.add_argument("--skip-versaprm", action="store_true",
                        help="Skip VersaPRM evaluation")
    parser.add_argument("--use-mathprm", action="store_true",
                        help="Enable MathPRM evaluation")
    parser.add_argument("--skip-mathprm", action="store_true",
                        help="Skip MathPRM evaluation (when --use-mathprm is set)")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")

    # Soft scoring
    parser.add_argument("--soft-scores", action="store_true", default=True,
                        help="Use logprob soft scores instead of binary (default: True)")
    parser.add_argument("--no-soft-scores", dest="soft_scores", action="store_false",
                        help="Use binary scores (0 or 1)")

    args = parser.parse_args()

    # Load trajectories from file
    print(f"Loading trajectories from {args.trajectories}...")
    trajectories = []
    with open(args.trajectories) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    print(f"✓ Loaded {len(trajectories)} trajectories")

    # Group by question
    groups = group_trajectories_by_question(trajectories)
    print(f"✓ {len(groups)} unique questions")

    # Load models
    generative_llm, generative_tokenizer, generative_lora_request = None, None, None
    versaprm_model, versaprm_tokenizer, versaprm_device = None, None, None

    if not args.skip_generative:
        generative_llm, generative_tokenizer, generative_lora_request = load_generative_model_vllm(
            args.generative_model, args.generative_base, gpu_memory_utilization=args.gpu_memory
        )

    # Scoring methods to evaluate
    scoring_methods = ['avg', 'min']

    # Results
    results = {
        'config': {
            'trajectories_file': args.trajectories,
            'num_questions': len(groups),
            'total_trajectories': len(trajectories),
            'scoring_methods': scoring_methods,
            'soft_scores': args.soft_scores,
        },
        'majority_voting': {'correct': 0, 'total': 0, 'f1_sum': 0.0},
        'details': [],
    }

    # Initialize result slots for all method x scoring combinations
    for scoring in scoring_methods:
        results[f'generative_bon_{scoring}'] = {'correct': 0, 'total': 0, 'f1_sum': 0.0}
        results[f'generative_wmv_{scoring}'] = {'correct': 0, 'total': 0, 'f1_sum': 0.0}
        results[f'versaprm_bon_{scoring}'] = {'correct': 0, 'total': 0, 'f1_sum': 0.0}
        results[f'versaprm_wmv_{scoring}'] = {'correct': 0, 'total': 0, 'f1_sum': 0.0}
        results[f'mathprm_bon_{scoring}'] = {'correct': 0, 'total': 0, 'f1_sum': 0.0}
        results[f'mathprm_wmv_{scoring}'] = {'correct': 0, 'total': 0, 'f1_sum': 0.0}

    print(f"\n{'='*70}")
    print("COMPARING VOTING METHODS")
    print(f"{'='*70}\n")

    # Flatten all trajectories
    all_trajs_flat = []
    qid_list = list(groups.keys())
    for qid in qid_list:
        for traj in groups[qid]:
            all_trajs_flat.append(traj)

    # Method 1: Majority Voting
    print("Computing Majority Voting...")
    majority_results = {}
    for qid, trajs in groups.items():
        majority_answer, vote_count = majority_vote(trajs)
        gold_answer = trajs[0].get('gold_answer')
        majority_correct = check_answer(majority_answer, gold_answer)
        majority_f1 = compute_f1(majority_answer, gold_answer)
        majority_results[qid] = {
            'answer': majority_answer,
            'votes': vote_count,
            'correct': majority_correct,
            'f1': majority_f1,
        }
        results['majority_voting']['total'] += 1
        results['majority_voting']['f1_sum'] += majority_f1
        if majority_correct:
            results['majority_voting']['correct'] += 1

    # Method 2: Our Generative (inference once, aggregate both ways)
    # Per-question step scores for reuse across scoring methods
    generative_per_question = {}  # qid -> [(traj, step_scores), ...]

    if generative_llm:
        print(f"Evaluating with Generative (vLLM)... {len(all_trajs_flat)} trajectories")
        # Define output path for incremental saving
        jsonl_path = args.output.replace('.json', '_per_trajectory.jsonl')
        
        generative_step_results = batch_evaluate_with_generative_vllm(
            generative_llm, generative_tokenizer, generative_lora_request, all_trajs_flat,
            use_soft_scores=args.soft_scores,
            output_jsonl_path=jsonl_path
        )

        # Group results by question
        for idx, (qid, trajs) in enumerate(groups.items()):
            # Find trajectories for this question in the flattened list
            # We need to map them correctly based on the original order in all_trajs_flat
            pass # We'll re-calculate mapping below
        
        # Simplified mapping back to groups
        flat_idx = 0
        for qid in qid_list:
            trajs = groups[qid]
            generative_per_question[qid] = []
            for traj in trajs:
                res = generative_step_results[flat_idx]
                if res:
                    step_labels, step_scores, step_reasonings = res
                    generative_per_question[qid].append((traj, step_scores))
                flat_idx += 1
        
        print(f"  ✓ Processed generative results for {len(generative_per_question)} questions")

        # Compute all scoring method combinations
        for scoring in scoring_methods:
            for qid in qid_list:
                trajs_and_scores = generative_per_question[qid]
                gold_answer = groups[qid][0].get('gold_answer')

                # Aggregate step scores → trajectory scores
                traj_scores = [aggregate_step_scores(ss, scoring) for _, ss in trajs_and_scores]

                # BoN: select trajectory with highest score
                best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
                best_traj = trajs_and_scores[best_idx][0]
                bon_answer = best_traj.get('final_answer', best_traj.get('predicted_answer', ''))
                bon_correct = check_answer(bon_answer, gold_answer)
                bon_f1 = compute_f1(bon_answer, gold_answer)
                results[f'generative_bon_{scoring}']['total'] += 1
                results[f'generative_bon_{scoring}']['f1_sum'] += bon_f1
                if bon_correct:
                    results[f'generative_bon_{scoring}']['correct'] += 1

                # Weighted Majority Voting
                weighted_votes = {}
                original_answers = {}
                for (traj, _), sc in zip(trajs_and_scores, traj_scores):
                    ans = traj.get('final_answer', traj.get('predicted_answer', ''))
                    if ans:
                        key = ans.strip().lower()
                        weighted_votes[key] = weighted_votes.get(key, 0.0) + sc
                        if key not in original_answers:
                            original_answers[key] = ans

                if weighted_votes:
                    best_key = max(weighted_votes, key=weighted_votes.get)
                    wmv_answer = original_answers.get(best_key, best_key)
                else:
                    wmv_answer = ""

                wmv_correct = check_answer(wmv_answer, gold_answer)
                wmv_f1 = compute_f1(wmv_answer, gold_answer)
                results[f'generative_wmv_{scoring}']['total'] += 1
                results[f'generative_wmv_{scoring}']['f1_sum'] += wmv_f1
                if wmv_correct:
                    results[f'generative_wmv_{scoring}']['correct'] += 1

    # Free GPU memory before loading next model
    need_next_model = (not args.skip_versaprm) or (args.use_mathprm and not args.skip_mathprm)
    if generative_llm and need_next_model:
        print("Unloading Generative model...")
        del generative_llm
        torch.cuda.empty_cache()

    # Load VersaPRM
    if not args.skip_versaprm and versaprm_model is None:
        versaprm_model, versaprm_tokenizer = load_versaprm(args.versaprm)
        versaprm_device = next(versaprm_model.parameters()).device

    # Method 3: VersaPRM (inference once, aggregate both ways)
    versaprm_per_question = {}  # qid -> [(traj, step_scores), ...]

    if versaprm_model:
        print(f"Evaluating with VersaPRM... {len(all_trajs_flat)} trajectories")
        versaprm_step_scores = batch_get_versaprm_scores(
            versaprm_model, versaprm_tokenizer, all_trajs_flat, versaprm_device, batch_size=8
        )

        # Group step scores by question
        score_idx = 0
        for qid in qid_list:
            trajs = groups[qid]
            versaprm_per_question[qid] = []
            for traj in trajs:
                step_scores = versaprm_step_scores[score_idx]
                versaprm_per_question[qid].append((traj, step_scores))
                score_idx += 1

        # Compute all scoring method combinations
        for scoring in scoring_methods:
            for qid in qid_list:
                trajs_and_scores = versaprm_per_question[qid]
                gold_answer = groups[qid][0].get('gold_answer')

                # Aggregate step scores → trajectory scores
                traj_scores = [aggregate_step_scores(ss, scoring) for _, ss in trajs_and_scores]

                # BoN
                best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
                best_traj = trajs_and_scores[best_idx][0]
                bon_answer = best_traj.get('final_answer', best_traj.get('predicted_answer', ''))
                bon_correct = check_answer(bon_answer, gold_answer)
                bon_f1 = compute_f1(bon_answer, gold_answer)
                results[f'versaprm_bon_{scoring}']['total'] += 1
                results[f'versaprm_bon_{scoring}']['f1_sum'] += bon_f1
                if bon_correct:
                    results[f'versaprm_bon_{scoring}']['correct'] += 1

                # Weighted Majority Voting
                weighted_votes = {}
                original_answers = {}
                for (traj, _), sc in zip(trajs_and_scores, traj_scores):
                    ans = traj.get('final_answer', traj.get('predicted_answer', ''))
                    if ans:
                        key = ans.strip().lower()
                        weighted_votes[key] = weighted_votes.get(key, 0.0) + sc
                        if key not in original_answers:
                            original_answers[key] = ans

                if weighted_votes:
                    best_key = max(weighted_votes, key=weighted_votes.get)
                    wmv_answer = original_answers.get(best_key, best_key)
                else:
                    wmv_answer = ""

                wmv_correct = check_answer(wmv_answer, gold_answer)
                wmv_f1 = compute_f1(wmv_answer, gold_answer)
                results[f'versaprm_wmv_{scoring}']['total'] += 1
                results[f'versaprm_wmv_{scoring}']['f1_sum'] += wmv_f1
                if wmv_correct:
                    results[f'versaprm_wmv_{scoring}']['correct'] += 1

    # Save VersaPRM per-trajectory JSONL
    if versaprm_per_question:
        versaprm_jsonl_path = args.output.replace('.json', '_per_trajectory.jsonl')
        print(f"Saving VersaPRM per-trajectory scores to {versaprm_jsonl_path}")
        with open(versaprm_jsonl_path, 'w') as vf:
            for qid in qid_list:
                if qid not in versaprm_per_question:
                    continue
                for traj, step_scores in versaprm_per_question[qid]:
                    rec = {
                        'trajectory_id': traj.get('trajectory_id', ''),
                        'question': traj.get('question', ''),
                        'gold_answer': traj.get('gold_answer', ''),
                        'predicted_answer': traj.get('final_answer', traj.get('predicted_answer', '')),
                        'is_correct': traj.get('is_correct', False),
                        'versaprm_step_scores': step_scores,
                        'versaprm_min': min(step_scores) if step_scores else 0.0,
                    }
                    vf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  ✓ Saved {sum(len(v) for v in versaprm_per_question.values())} trajectories")

    # Free VersaPRM GPU memory before MathPRM
    if versaprm_model and args.use_mathprm and not args.skip_mathprm:
        print("Unloading VersaPRM model...")
        del versaprm_model, versaprm_tokenizer
        torch.cuda.empty_cache()

    # Method 4: MathPRM (Qwen2.5-Math-PRM-7B)
    mathprm_per_question = {}  # qid -> [(traj, step_scores), ...]

    if args.use_mathprm and not args.skip_mathprm:
        mathprm_model, mathprm_tokenizer = load_mathprm(args.mathprm)

        print(f"Evaluating with MathPRM... {len(all_trajs_flat)} trajectories")
        mathprm_step_scores = batch_get_mathprm_scores(
            mathprm_model, mathprm_tokenizer, all_trajs_flat, batch_size=4
        )

        # Group step scores by question
        score_idx = 0
        for qid in qid_list:
            trajs = groups[qid]
            mathprm_per_question[qid] = []
            for traj in trajs:
                step_scores = mathprm_step_scores[score_idx]
                mathprm_per_question[qid].append((traj, step_scores))
                score_idx += 1

        # Compute all scoring method combinations
        for scoring in scoring_methods:
            for qid in qid_list:
                trajs_and_scores = mathprm_per_question[qid]
                gold_answer = groups[qid][0].get('gold_answer')

                # Aggregate step scores -> trajectory scores
                traj_scores = [aggregate_step_scores(ss, scoring) for _, ss in trajs_and_scores]

                # BoN
                best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
                best_traj = trajs_and_scores[best_idx][0]
                bon_answer = best_traj.get('final_answer', best_traj.get('predicted_answer', ''))
                bon_correct = check_answer(bon_answer, gold_answer)
                bon_f1 = compute_f1(bon_answer, gold_answer)
                results[f'mathprm_bon_{scoring}']['total'] += 1
                results[f'mathprm_bon_{scoring}']['f1_sum'] += bon_f1
                if bon_correct:
                    results[f'mathprm_bon_{scoring}']['correct'] += 1

                # Weighted Majority Voting
                weighted_votes = {}
                original_answers = {}
                for (traj, _), sc in zip(trajs_and_scores, traj_scores):
                    ans = traj.get('final_answer', traj.get('predicted_answer', ''))
                    if ans:
                        key = ans.strip().lower()
                        weighted_votes[key] = weighted_votes.get(key, 0.0) + sc
                        if key not in original_answers:
                            original_answers[key] = ans

                if weighted_votes:
                    best_key = max(weighted_votes, key=weighted_votes.get)
                    wmv_answer = original_answers.get(best_key, best_key)
                else:
                    wmv_answer = ""

                wmv_correct = check_answer(wmv_answer, gold_answer)
                wmv_f1 = compute_f1(wmv_answer, gold_answer)
                results[f'mathprm_wmv_{scoring}']['total'] += 1
                results[f'mathprm_wmv_{scoring}']['f1_sum'] += wmv_f1
                if wmv_correct:
                    results[f'mathprm_wmv_{scoring}']['correct'] += 1

        # Save MathPRM per-trajectory JSONL
        mathprm_jsonl_path = args.output.replace('.json', '_per_trajectory.jsonl')
        print(f"Saving MathPRM per-trajectory scores to {mathprm_jsonl_path}")
        with open(mathprm_jsonl_path, 'w') as mf:
            for qid in qid_list:
                if qid not in mathprm_per_question:
                    continue
                for traj, step_scores in mathprm_per_question[qid]:
                    rec = {
                        'trajectory_id': traj.get('trajectory_id', ''),
                        'question': traj.get('question', ''),
                        'gold_answer': traj.get('gold_answer', ''),
                        'predicted_answer': traj.get('final_answer', traj.get('predicted_answer', '')),
                        'is_correct': traj.get('is_correct', False),
                        'mathprm_step_scores': step_scores,
                        'mathprm_min': min(step_scores) if step_scores else 0.0,
                    }
                    mf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  ✓ Saved {sum(len(v) for v in mathprm_per_question.values())} trajectories")

        # Free MathPRM
        del mathprm_model, mathprm_tokenizer
        torch.cuda.empty_cache()

    # Build details
    for qid in qid_list:
        trajs = groups[qid]
        detail = {
            'question_id': qid,
            'question': trajs[0]['question'][:100],
            'gold_answer': trajs[0].get('gold_answer'),
            'num_trajectories': len(trajs),
            'majority': majority_results.get(qid, {}),
        }
        # Add per-question generative step scores for analysis
        if qid in generative_per_question:
            for scoring in scoring_methods:
                traj_scores = [aggregate_step_scores(ss, scoring) for _, ss in generative_per_question[qid]]
                best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
                best_traj = generative_per_question[qid][best_idx][0]
                detail[f'generative_{scoring}'] = {
                    'answer': best_traj.get('final_answer', best_traj.get('predicted_answer', '')),
                    'score': traj_scores[best_idx],
                    'correct': check_answer(
                        best_traj.get('final_answer', best_traj.get('predicted_answer', '')),
                        trajs[0].get('gold_answer')
                    ),
                }
        if qid in versaprm_per_question:
            for scoring in scoring_methods:
                traj_scores = [aggregate_step_scores(ss, scoring) for _, ss in versaprm_per_question[qid]]
                best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
                best_traj = versaprm_per_question[qid][best_idx][0]
                detail[f'versaprm_{scoring}'] = {
                    'answer': best_traj.get('final_answer', best_traj.get('predicted_answer', '')),
                    'score': traj_scores[best_idx],
                    'correct': check_answer(
                        best_traj.get('final_answer', best_traj.get('predicted_answer', '')),
                        trajs[0].get('gold_answer')
                    ),
                }
        if qid in mathprm_per_question:
            for scoring in scoring_methods:
                traj_scores = [aggregate_step_scores(ss, scoring) for _, ss in mathprm_per_question[qid]]
                best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
                best_traj = mathprm_per_question[qid][best_idx][0]
                detail[f'mathprm_{scoring}'] = {
                    'answer': best_traj.get('final_answer', best_traj.get('predicted_answer', '')),
                    'score': traj_scores[best_idx],
                    'correct': check_answer(
                        best_traj.get('final_answer', best_traj.get('predicted_answer', '')),
                        trajs[0].get('gold_answer')
                    ),
                }
        results['details'].append(detail)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # Print majority first
    data = results['majority_voting']
    acc = 100 * data['correct'] / data['total']
    f1 = 100 * data['f1_sum'] / data['total'] if data['total'] > 0 else 0
    print(f"{'Majority Voting':30s}: EM={acc:.1f}%  F1={f1:.1f}%")
    print()

    # Print all methods with both EM and F1
    for scoring in scoring_methods:
        for method_prefix, label_prefix in [('generative', 'Generative'), ('versaprm', 'VersaPRM'), ('mathprm', 'MathPRM')]:
            for strategy, strategy_label in [('bon', 'BoN'), ('wmv', 'Weighted MV')]:
                key = f'{method_prefix}_{strategy}_{scoring}'
                data = results[key]
                if data['total'] > 0:
                    acc = 100 * data['correct'] / data['total']
                    f1 = 100 * data['f1_sum'] / data['total']
                    label = f"{label_prefix} ({strategy_label}, {scoring})"
                    print(f"{label:30s}: EM={acc:.1f}%  F1={f1:.1f}%")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    print(">>> SCRIPT STARTING...")
    main()
