#!/usr/bin/env python3
"""Judge labeling using QwQ-32B model.

Optimizations:
1. Trajectory 단위 배치: 한 번의 LLM 호출로 모든 step 평가
2. Incremental save: JSONL로 저장하여 중간에 끊겨도 복구 가능
3. Resume 지원: 이미 처리된 question_id 스킵
"""

import json
import sys
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prmrag.labeling.judge_labeler import JudgeLabeler
from prmrag.data.schemas import Trajectory, TrajectoryStep


def parse_content(content: str) -> dict:
    """Parse content field to extract XML tag contents.

    XML format:
        <think>reasoning</think>
        <search>query</search> or <answer>answer</answer>
        <documents>retrieved passages</documents>

    Returns dict with:
        - think: reasoning content
        - search: search query (if search step)
        - answer: final answer (if answer step)
        - documents: retrieved passages
        - action_type: 'search', 'answer', or 'reason'
    """
    think = ""
    search = ""
    answer = ""
    documents = ""
    action_type = "reason"

    # Extract XML tags
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    docs_match = re.search(r'<documents>(.*?)</documents>', content, re.DOTALL)

    if think_match:
        think = think_match.group(1).strip()

    if search_match:
        search = search_match.group(1).strip()
        action_type = "search"
    elif answer_match:
        answer = answer_match.group(1).strip()
        action_type = "answer"

    if docs_match:
        documents = docs_match.group(1).strip()

    return {
        'think': think,
        'search': search,
        'answer': answer,
        'documents': documents,
        'action_type': action_type,
    }


def load_trajectories(filepath: Path):
    """Load trajectories from JSONL file.

    Handles XML format: <think>, <search>, <answer>, <documents>
    """
    trajectories = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Convert to Trajectory schema
            steps = []
            for step_data in data['steps']:
                # Get text content - try 'text'/'content' first, then reconstruct from parsed fields
                content = step_data.get('text', step_data.get('content', ''))

                if not content:
                    # Reconstruct XML from parsed fields (think, search, answer, documents)
                    parts = []
                    if step_data.get('think'):
                        parts.append(f"<think>{step_data['think']}</think>")
                    if step_data.get('search'):
                        parts.append(f"<search>{step_data['search']}</search>")
                    if step_data.get('answer'):
                        parts.append(f"<answer>{step_data['answer']}</answer>")
                    if step_data.get('documents'):
                        parts.append(f"<documents>{step_data['documents']}</documents>")
                    content = "\n".join(parts)

                parsed = parse_content(content)

                # Map action_type: search -> search, answer -> finish
                action_type = parsed['action_type']
                if action_type == 'answer':
                    action_type = 'finish'

                step = TrajectoryStep(
                    step_id=step_data.get('step_id', 1) - 1,
                    action_type=action_type,
                    action=content,  # XML content (original or reconstructed)
                    observation=parsed['documents'],
                    passages=[parsed['documents']] if parsed['documents'] else [],
                )
                steps.append(step)

            trajectory = Trajectory(
                trajectory_id=data.get('trajectory_id', data.get('question_id', '')),
                question=data['question'],
                gold_answer=data['gold_answer'],
                final_answer=data.get('final_answer', ''),
                steps=steps,
                supporting_facts=[],
            )
            trajectories.append((trajectory, data))

    return trajectories


def load_processed_ids(output_file: Path) -> set:
    """Load already processed trajectory IDs from output file."""
    processed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Support both old (question_id) and new (trajectory_id) format
                        tid = data.get('trajectory_id', data.get('question_id'))
                        if tid:
                            processed.add(tid)
                    except:
                        pass
    return processed


def main():
    parser = argparse.ArgumentParser(description="Judge labeling with QwQ-32B")
    parser.add_argument("--input", type=str,
                        default="./outputs/trajectories_xml.jsonl")
    parser.add_argument("--output", type=str,
                        default="./outputs/judge_labels_500q_16s.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of trajectories (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for vLLM batch processing (default: 50)")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    print("=" * 70)
    print("JUDGE LABELING with QwQ-32B (Optimized)")
    print("=" * 70)
    print()
    print("Optimizations:")
    print("  ✓ vLLM batch processing: 여러 trajectory를 한 번에 처리")
    print("  ✓ Trajectory 단위 배치: 한 번의 LLM 호출로 모든 step 평가")
    print("  ✓ Incremental save: JSONL로 저장 (중간 중단 시 복구 가능)")
    print("  ✓ Resume 지원: --resume로 이어서 처리")
    print()

    # Load trajectories
    print(f"Loading trajectories from: {input_file}")
    trajectory_pairs = load_trajectories(input_file)
    print(f"✓ Loaded {len(trajectory_pairs)} trajectories")

    # Apply limit if specified
    if args.limit:
        trajectory_pairs = trajectory_pairs[:args.limit]
        print(f"✓ Limited to first {len(trajectory_pairs)} trajectories")

    # Load processed IDs for resume
    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(output_file)
        print(f"✓ Resume mode: {len(processed_ids)} already processed")

    # Filter out already processed
    remaining = [(t, d) for t, d in trajectory_pairs if t.trajectory_id not in processed_ids]
    print(f"✓ Remaining to process: {len(remaining)} trajectories")
    print()

    # Initialize Judge labeler
    config = {
        'model_name': 'Qwen/QwQ-32B',
        'temperature': 0.3,
        'max_tokens': 8192,
        'max_model_len': 32768,
        'gpu_memory_utilization': 0.95,
        'tensor_parallel_size': 2,
        'prompt_style': 'versaprm',
        'use_gold_answer': True,
        'use_supporting_facts': False,
    }

    print("Initializing QwQ-32B (one-time)...")
    judge_labeler = JudgeLabeler(config)
    print("✓ Model loaded\n")

    # Open output file for incremental writing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_fp = open(output_file, 'a' if args.resume else 'w', encoding='utf-8')

    # Statistics
    total_steps = 0
    good_count = 0
    bad_count = 0
    error_count = 0

    # Process in batches using vLLM batch processing
    batch_size = args.batch_size
    num_batches = (len(remaining) + batch_size - 1) // batch_size

    print(f"Processing {len(remaining)} trajectories in {num_batches} batches (batch_size={batch_size})")
    print()

    for batch_idx in tqdm(range(num_batches), desc="Batch processing"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(remaining))
        batch = remaining[start_idx:end_idx]

        # Extract trajectories for batch processing
        trajectories = [t for t, d in batch]
        original_datas = [d for t, d in batch]

        try:
            # Use vLLM batch processing for much faster inference
            all_judge_labels = judge_labeler.label_batch(trajectories, show_progress=False)
        except Exception as e:
            tqdm.write(f"  BATCH ERROR: {e}")
            error_count += len(batch)
            from prmrag.data.schemas import JudgeLabel
            all_judge_labels = [
                [
                    JudgeLabel(
                        step_id=step_idx,
                        label='UNKNOWN',
                        reasoning=f'Error: {str(e)}',
                        confidence=0.0,
                    )
                    for step_idx in range(len(t.steps))
                ]
                for t in trajectories
            ]

        # Process each result in the batch
        for trajectory, original_data, judge_labels in zip(trajectories, original_datas, all_judge_labels):
            # Prepare output
            output_data = {
                'trajectory_id': trajectory.trajectory_id,
                'question': trajectory.question,
                'gold_answer': trajectory.gold_answer,
                'predicted_answer': trajectory.final_answer,
                'is_correct': original_data.get('is_correct', False),
                'steps': []
            }

            for step_idx, (step, judge_label) in enumerate(zip(original_data['steps'], judge_labels)):
                # Parse content to extract XML fields
                content = step.get('text', step.get('content', ''))

                if not content:
                    # Reconstruct XML from parsed fields
                    parts = []
                    if step.get('think'): parts.append(f"<think>{step['think']}</think>")
                    if step.get('search'): parts.append(f"<search>{step['search']}</search>")
                    if step.get('answer'): parts.append(f"<answer>{step['answer']}</answer>")
                    if step.get('documents'): parts.append(f"<documents>{step['documents']}</documents>")
                    content = "\n".join(parts)

                parsed = parse_content(content)

                step_output = {
                    'step_id': step.get('step_id', step_idx + 1),
                    'step_type': parsed['action_type'],
                    # XML fields
                    'think': parsed['think'],
                    'search': parsed['search'] if parsed['action_type'] == 'search' else None,
                    'answer': parsed['answer'] if parsed['action_type'] == 'answer' else None,
                    'documents': parsed['documents'][:2000] if parsed['documents'] else None,
                    # Judge labels
                    'judge_label': judge_label.label,
                    'judge_reasoning': judge_label.reasoning,
                }
                # Remove None fields
                step_output = {k: v for k, v in step_output.items() if v is not None}
                output_data['steps'].append(step_output)

                # Update statistics
                total_steps += 1
                if judge_label.label == 'GOOD':
                    good_count += 1
                elif judge_label.label == 'BAD':
                    bad_count += 1

            # Write immediately (incremental save)
            out_fp.write(json.dumps(output_data, ensure_ascii=False) + '\n')

        # Flush after each batch
        out_fp.flush()

    out_fp.close()

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Total trajectories processed: {len(remaining)}")
    print(f"Total steps evaluated: {total_steps}")
    print(f"Errors: {error_count}")
    if total_steps > 0:
        print(f"GOOD labels: {good_count}/{total_steps} ({good_count/total_steps*100:.1f}%)")
        print(f"BAD labels: {bad_count}/{total_steps} ({bad_count/total_steps*100:.1f}%)")
    print()
    print(f"✓ Output saved to: {output_file}")
    print()


if __name__ == '__main__':
    main()
