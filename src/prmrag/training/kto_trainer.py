"""Step-level KTO Trainer for Policy Model.

Implements KTO (Kahneman-Tversky Optimization) with step-level PRM loss.
Key features:
1. Step-level loss: Each step in a trajectory gets its own KTO loss
2. Document masking: <documents>...</documents> tokens excluded from loss
3. Fixed lambda_U/lambda_D for GOOD/BAD ratio balancing (λD*nD ≈ λU*nU)
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


SYSTEM_PROMPT = """You are a helpful assistant who is good at answering questions with multi-turn search engine calling. To answer questions, you must first reason through the available information using <think> and </think>. If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with relevant documents enclosed in <documents> and </documents>. You can search as many times as you want. Once you have sufficient information or if you find no further external knowledge is needed, directly provide a concise final answer using <answer> and </answer> without detailed illustrations."""


@dataclass
class StepAnnotation:
    """Annotation for a single step within a trajectory."""
    step_start_token: int   # step start token idx in completion
    step_end_token: int     # step end token idx in completion
    doc_mask: List[bool]    # True = mask out (documents region)
    label: bool             # True=GOOD, False=BAD
    generative_score: float     # generative model score (used for label thresholding)


class KTODataPreparer:
    """Trajectory + generative scores -> KTO training data."""

    def __init__(self, tokenizer: AutoTokenizer, system_prompt: str = SYSTEM_PROMPT):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

        # Document boundary markers (text-based matching)
        self._doc_start_tag = "<documents>"
        self._doc_end_tag = "</documents>"

    def prepare_dataset(
        self,
        trajectories_path: str,
        generative_scores_path: str,
        truncate_after_bad: bool = False,
        best_of_n: bool = False,
        limit: Optional[int] = None,
    ) -> Dataset:
        """Convert trajectories + generative scores into KTO training dataset.

        Uses two separate files (old format: judge_labels + generative_scores).

        Args:
            trajectories_path: Path to judge_labels JSONL (step content)
            generative_scores_path: Path to generative_scores per_trajectory JSONL
            truncate_after_bad: If True, remove steps after first BAD step
            best_of_n: If True, pick only the best trajectory per question
                       (highest generative_min)
            limit: Limit number of trajectories for debugging

        Returns:
            Dataset with prompt, completion, step_annotations (serialized)
        """
        from prmrag.regeneration.classifier import QuestionClassifier

        scored_trajs = QuestionClassifier.load_and_merge(
            trajectories_path, generative_scores_path
        )

        if best_of_n:
            scored_trajs = self._select_best_per_question(scored_trajs)

        if limit:
            scored_trajs = scored_trajs[:limit]

        # Convert ScoredTrajectory to generic dicts for shared processing
        traj_dicts = []
        for t in scored_trajs:
            traj_dicts.append({
                'trajectory_id': t.trajectory_id,
                'question': t.question,
                'is_correct': t.is_correct,
                'generative_step_scores': t.generative_step_scores,
                'generative_step_labels': t.generative_step_labels,
                'steps': t.steps,
            })

        return self._process_trajectories(traj_dicts, truncate_after_bad)

    def prepare_dataset_combined(
        self,
        input_paths: List[str],
        truncate_after_bad: bool = False,
        filter_no_gold_in_docs: bool = False,
        limit: Optional[int] = None,
    ) -> Dataset:
        """Convert combined-format JSONL files into KTO training dataset.

        Reads from files where each trajectory already has generative_step_labels,
        generative_step_scores, and per-step generative_label/generative_score.

        This supports the new pipeline where:
        - Original data: hotpotqa_generative_results_v8_per_trajectory.jsonl
        - Regenerated data: regenerated_generative_scored.jsonl

        Args:
            input_paths: List of JSONL file paths to combine
            truncate_after_bad: If True, remove steps after first BAD step
            limit: Limit total number of trajectories for debugging

        Returns:
            Dataset with prompt, completion, step_annotations (serialized)
        """
        traj_dicts = []
        filtered_count = 0

        for path in input_paths:
            count = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)

                    # Filter: skip trajectories where gold answer not in any document
                    if filter_no_gold_in_docs:
                        gold = record.get('gold_answer', '')
                        if gold:
                            gold_norm = re.sub(r'[^a-z0-9\s]', ' ',
                                               gold.lower().strip())
                            gold_norm = re.sub(r'\s+', ' ', gold_norm).strip()
                            has_gold = False
                            for step in record.get('steps', []):
                                docs = step.get('documents', '')
                                if docs:
                                    docs_norm = re.sub(r'[^a-z0-9\s]', ' ',
                                                       docs.lower().strip())
                                    docs_norm = re.sub(r'\s+', ' ', docs_norm).strip()
                                    if gold_norm in docs_norm:
                                        has_gold = True
                                        break
                            if not has_gold:
                                filtered_count += 1
                                continue

                    steps = record.get('steps', [])
                    # Get trajectory-level labels/scores (preferred)
                    # or derive from per-step fields
                    if 'generative_step_labels' in record:
                        generative_labels = record['generative_step_labels']
                        generative_scores = record['generative_step_scores']
                    else:
                        generative_labels = [s.get('generative_label', 1) for s in steps]
                        generative_scores = [s.get('generative_score', 1.0) for s in steps]

                    traj_dicts.append({
                        'trajectory_id': record['trajectory_id'],
                        'question': record['question'],
                        'is_correct': record.get('is_correct', False),
                        'generative_step_scores': generative_scores,
                        'generative_step_labels': generative_labels,
                        'steps': steps,
                        'source': path,
                    })
                    count += 1
            print(f"  Loaded {count} trajectories from {path}")

        if filter_no_gold_in_docs:
            print(f"  Filtered out {filtered_count} trajectories (gold not in docs)")
        print(f"  Total: {len(traj_dicts)} trajectories from {len(input_paths)} files")

        if limit:
            traj_dicts = traj_dicts[:limit]
            print(f"  Limited to {len(traj_dicts)} trajectories")

        return self._process_trajectories(traj_dicts, truncate_after_bad)

    def _process_trajectories(
        self,
        traj_dicts: List[Dict[str, Any]],
        truncate_after_bad: bool = False,
    ) -> Dataset:
        """Shared processing: convert trajectory dicts into KTO Dataset.

        Each dict must have: trajectory_id, question, is_correct,
        generative_step_scores, generative_step_labels, steps.
        """
        samples = []
        stats = {"total": 0, "good_steps": 0, "bad_steps": 0, "skipped": 0,
                 "doc_tokens_masked": 0, "total_completion_tokens": 0}

        for traj in traj_dicts:
            steps = traj['steps']
            generative_scores = traj['generative_step_scores']
            generative_labels = traj['generative_step_labels']

            # Ensure same number of steps and scores
            if len(steps) != len(generative_scores):
                stats["skipped"] += 1
                continue

            # Optionally truncate after first BAD step
            if truncate_after_bad:
                truncated_steps = []
                truncated_scores = []
                truncated_labels = []
                for i, (step, score, label) in enumerate(zip(steps, generative_scores, generative_labels)):
                    truncated_steps.append(step)
                    truncated_scores.append(score)
                    truncated_labels.append(label)
                    if label == 0:  # BAD: include this step, then stop
                        break
                steps = truncated_steps
                generative_scores = truncated_scores
                generative_labels = truncated_labels

            # Build prompt and completion
            prompt = self._build_prompt(traj['question'])
            completion = self._build_trajectory_text(steps)

            if not completion.strip():
                stats["skipped"] += 1
                continue

            # Tokenize completion to find step boundaries and document spans
            completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

            # Find step boundaries in token space
            step_boundaries = self._find_step_boundaries(completion, completion_ids, steps)

            if not step_boundaries:
                stats["skipped"] += 1
                continue

            # Find document spans in token space
            doc_spans = self._find_document_spans(completion, completion_ids)

            # Build step annotations
            step_annotations = []
            for i, (start, end) in enumerate(step_boundaries):
                if i >= len(generative_scores):
                    break

                # Build doc mask for this step
                doc_mask = [False] * (end - start)
                for doc_start, doc_end in doc_spans:
                    # Overlap with this step
                    overlap_start = max(start, doc_start) - start
                    overlap_end = min(end, doc_end) - start
                    if overlap_start < overlap_end:
                        for j in range(overlap_start, overlap_end):
                            doc_mask[j] = True

                label = generative_labels[i] == 1  # 1=GOOD -> True
                score = generative_scores[i]

                step_annotations.append(StepAnnotation(
                    step_start_token=start,
                    step_end_token=end,
                    doc_mask=doc_mask,
                    label=label,
                    generative_score=score,
                ))

                if label:
                    stats["good_steps"] += 1
                else:
                    stats["bad_steps"] += 1

            if not step_annotations:
                stats["skipped"] += 1
                continue

            # Serialize annotations for Dataset storage
            serialized_annotations = self._serialize_annotations(step_annotations)

            stats["total"] += 1
            stats["doc_tokens_masked"] += sum(
                sum(ann.doc_mask) for ann in step_annotations
            )
            stats["total_completion_tokens"] += len(completion_ids)

            samples.append({
                "prompt": prompt,
                "completion": completion,
                "step_annotations_json": json.dumps(serialized_annotations),
                "trajectory_id": traj['trajectory_id'],
                "question": traj['question'],
                "is_correct": traj['is_correct'],
            })

        # Print stats
        print(f"\n{'='*60}")
        print(f"KTO Data Preparation Stats:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  GOOD steps: {stats['good_steps']}")
        print(f"  BAD steps: {stats['bad_steps']}")
        if stats['total_completion_tokens'] > 0:
            mask_pct = stats['doc_tokens_masked'] / stats['total_completion_tokens'] * 100
            print(f"  Document tokens masked: {stats['doc_tokens_masked']} ({mask_pct:.1f}%)")
        print(f"{'='*60}\n")

        return Dataset.from_list(samples)

    @staticmethod
    def _select_best_per_question(scored_trajs) -> list:
        """Pick best trajectory per question (highest generative_min)."""
        from collections import defaultdict
        by_question = defaultdict(list)
        for traj in scored_trajs:
            by_question[traj.question_id].append(traj)

        best_trajs = []
        for qid, trajs in sorted(by_question.items()):
            best = max(trajs, key=lambda t: t.generative_min)
            best_trajs.append(best)

        print(f"  Best-of-N: {len(scored_trajs)} trajectories -> {len(best_trajs)} (1 per question)")
        return best_trajs

    def _build_prompt(self, question: str) -> str:
        """Build the prompt (system + question) using chat template."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]
        # Apply chat template without generation prompt (we add completion separately)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def _build_trajectory_text(self, steps: List[Dict]) -> str:
        """Convert steps to XML format trajectory text (the completion)."""
        parts = []
        for step in steps:
            # Think
            think = step.get("think", "")
            if think:
                parts.append(f"<think>{think}</think>")

            # Search or Answer
            search = step.get("search", "")
            answer = step.get("answer", "")
            if search:
                parts.append(f"<search>{search}</search>")
            elif answer:
                parts.append(f"<answer>{answer}</answer>")

            # Documents (only for search steps)
            docs = step.get("documents", "")
            if docs:
                parts.append(f"<documents>{docs}</documents>")

        return "\n".join(parts)

    def _find_step_boundaries(
        self,
        completion_text: str,
        completion_ids: List[int],
        steps: List[Dict],
    ) -> List[Tuple[int, int]]:
        """Find each step's start/end token positions in the completion.

        Each step starts with <think> and ends before the next <think> or at completion end.
        """
        # Find <think> positions in the text to identify step boundaries
        think_pattern = re.compile(r"<think>")
        text_positions = [m.start() for m in think_pattern.finditer(completion_text)]

        if not text_positions:
            return []

        # Convert text positions to approximate token positions
        # by encoding prefixes
        boundaries = []
        for i, text_pos in enumerate(text_positions):
            # Token position: encode text up to this point
            prefix = completion_text[:text_pos]
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            start_token = len(prefix_ids)

            if i + 1 < len(text_positions):
                # End at next step's start
                next_prefix = completion_text[:text_positions[i + 1]]
                next_prefix_ids = self.tokenizer.encode(next_prefix, add_special_tokens=False)
                end_token = len(next_prefix_ids)
            else:
                # Last step: end at completion end
                end_token = len(completion_ids)

            if start_token < end_token:
                boundaries.append((start_token, end_token))

        return boundaries

    def _find_document_spans(self, completion_text: str, completion_ids: List[int]) -> List[Tuple[int, int]]:
        """Find <documents>...</documents> spans in token space.

        Uses text-based matching then converts to token positions,
        because tokenizer may merge tag characters with adjacent content.

        Returns list of (start, end) token positions to mask.
        """
        spans = []
        search_start = 0

        while True:
            doc_start = completion_text.find(self._doc_start_tag, search_start)
            if doc_start == -1:
                break

            doc_end_pos = completion_text.find(self._doc_end_tag, doc_start)
            if doc_end_pos == -1:
                # No closing tag: mask to end
                prefix = completion_text[:doc_start]
                start_token = len(self.tokenizer.encode(prefix, add_special_tokens=False))
                spans.append((start_token, len(completion_ids)))
                break

            doc_end = doc_end_pos + len(self._doc_end_tag)

            # Convert text positions to token positions
            prefix_before = completion_text[:doc_start]
            prefix_after = completion_text[:doc_end]
            start_token = len(self.tokenizer.encode(prefix_before, add_special_tokens=False))
            end_token = len(self.tokenizer.encode(prefix_after, add_special_tokens=False))

            if start_token < end_token:
                spans.append((start_token, end_token))

            search_start = doc_end

        return spans

    @staticmethod
    def _serialize_annotations(annotations: List[StepAnnotation]) -> List[Dict]:
        """Serialize StepAnnotation list for JSON storage."""
        return [
            {
                "start": ann.step_start_token,
                "end": ann.step_end_token,
                "doc_mask": ann.doc_mask,
                "label": ann.label,
                "generative_score": ann.generative_score,
            }
            for ann in annotations
        ]

    @staticmethod
    def deserialize_annotations(json_str: str) -> List[StepAnnotation]:
        """Deserialize step annotations from JSON string."""
        data = json.loads(json_str)
        return [
            StepAnnotation(
                step_start_token=d["start"],
                step_end_token=d["end"],
                doc_mask=d["doc_mask"],
                label=d["label"],
                generative_score=d["generative_score"],
            )
            for d in data
        ]


class EarlyCollapseCallback(TrainerCallback):
    """Collapse detection callback — stops training early if KTO diverges.

    Collapse criteria (checked after warmup_steps):
    - bad_loss < bad_loss_threshold  : BAD steps get no gradient (sigmoid saturated)
    - kl < kl_threshold              : model worse than ref across all steps
    """

    def __init__(
        self,
        bad_loss_threshold: float = 0.05,
        kl_threshold: float = -5.0,
        warmup_steps: int = 30,
    ):
        self.bad_loss_threshold = bad_loss_threshold
        self.kl_threshold = kl_threshold
        self.warmup_steps = warmup_steps

        self.collapsed = False
        self.collapse_step: Optional[int] = None
        self.collapse_reason: Optional[str] = None
        self.metric_history: List[Dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        bad_loss = logs.get("kto/bad_loss")
        kl = logs.get("kto/kl")
        good_loss = logs.get("kto/good_loss")

        if bad_loss is not None and kl is not None:
            self.metric_history.append({
                "step": state.global_step,
                "bad_loss": bad_loss,
                "kl": kl,
                "good_loss": good_loss,
            })

        if state.global_step < self.warmup_steps:
            return

        if bad_loss is not None and bad_loss < self.bad_loss_threshold:
            self.collapsed = True
            self.collapse_step = state.global_step
            self.collapse_reason = f"bad_loss={bad_loss:.4f} < {self.bad_loss_threshold}"
            print(f"\n{'!'*60}")
            print(f"[COLLAPSE @ step {state.global_step}] {self.collapse_reason}")
            print(f"{'!'*60}")
            control.should_training_stop = True

        elif kl is not None and kl < self.kl_threshold:
            self.collapsed = True
            self.collapse_step = state.global_step
            self.collapse_reason = f"KL={kl:.4f} < {self.kl_threshold}"
            print(f"\n{'!'*60}")
            print(f"[COLLAPSE @ step {state.global_step}] {self.collapse_reason}")
            print(f"{'!'*60}")
            control.should_training_stop = True

    def summary(self) -> Dict:
        """Return summary dict for sweep result file."""
        last = self.metric_history[-1] if self.metric_history else {}
        return {
            "collapsed": self.collapsed,
            "collapse_step": self.collapse_step,
            "collapse_reason": self.collapse_reason,
            "final_step": last.get("step"),
            "final_bad_loss": last.get("bad_loss"),
            "final_good_loss": last.get("good_loss"),
            "final_kl": last.get("kl"),
            "metric_history": self.metric_history,
        }


class KTODebugCallback(TrainerCallback):
    """Callback for monitoring KTO training metrics."""

    def __init__(self):
        self.loss_history = []
        self.good_loss_history = []
        self.bad_loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            if loss is not None:
                self.loss_history.append(loss)

            # Custom metrics from compute_loss
            good_loss = logs.get("kto/good_loss")
            bad_loss = logs.get("kto/bad_loss")
            if good_loss is not None:
                self.good_loss_history.append(good_loss)
            if bad_loss is not None:
                self.bad_loss_history.append(bad_loss)

            if torch.cuda.is_available() and args.local_rank in (-1, 0):
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3

                print(f"\n{'='*60}")
                print(f"[Step {state.global_step}] Loss: {loss:.4f}" if loss else f"[Step {state.global_step}]")
                if good_loss is not None:
                    print(f"  GOOD loss: {good_loss:.4f}")
                if bad_loss is not None:
                    print(f"  BAD loss: {bad_loss:.4f}")
                kl = logs.get("kto/kl")
                if kl is not None:
                    print(f"  KL: {kl:.4f}")
                z0 = logs.get("kto/z0")
                if z0 is not None:
                    print(f"  z0: {z0:.4f}")
                print(f"  GPU: {allocated:.2f}GB / {reserved:.2f}GB")
                print(f"{'='*60}")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*70}")
        print("KTO TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total steps: {state.global_step}")
        if self.loss_history:
            print(f"Loss: {self.loss_history[0]:.4f} -> {self.loss_history[-1]:.4f}")
            print(f"Min: {min(self.loss_history):.4f}, Max: {max(self.loss_history):.4f}")
        if self.good_loss_history:
            print(f"GOOD loss avg: {sum(self.good_loss_history)/len(self.good_loss_history):.4f}")
        if self.bad_loss_history:
            print(f"BAD loss avg: {sum(self.bad_loss_history)/len(self.bad_loss_history):.4f}")
        print(f"{'='*70}")


def _get_per_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probabilities.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) - token IDs

    Returns:
        (batch, seq_len-1) - log probs for each predicted token
    """
    # Shift: logits[t] predicts labels[t+1]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    # Gather the log prob of the actual next token
    per_token_logps = log_probs.gather(
        dim=-1, index=labels[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return per_token_logps


class StepLevelKTOTrainer(Trainer):
    """Custom Trainer implementing step-level KTO loss.

    Unlike trl KTOTrainer which treats entire completion as chosen/rejected,
    this trainer computes KTO loss at the step level within a single trajectory.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        beta: float = 0.1,
        lambda0: float = 1.0,
        desirable_weight: float = 1.0,
        max_length: int = 4096,
        **kwargs,
    ):
        super().__init__(model=model, processing_class=tokenizer, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.lambda0 = lambda0
        self.desirable_weight = desirable_weight
        self.max_length = max_length
        self._tokenizer = tokenizer

        # Move ref_model to same device and freeze
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # Running KL estimate (EMA across batches) for logging
        self._running_kl = 0.0
        self._running_kl_alpha = 0.1  # EMA decay
        self._current_z0 = 0.0        # z0 used in last compute_loss call (for logging)
        self._last_logged_step = -1   # Prevent duplicate logs per optimizer step

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute step-level KTO loss.

        Steps:
        1. Forward pass through model and ref_model
        2. Compute token-level log probs
        3. Pre-compute batch-level KL (reference point)
        4. Apply KTO loss per step with document masking
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        step_annotations_json = inputs["step_annotations_json"]
        prompt_lengths = inputs["prompt_length"]

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Forward pass: model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        token_logps = _get_per_token_logps(outputs.logits, input_ids)

        # Forward pass: ref_model (no grad)
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_token_logps = _get_per_token_logps(ref_outputs.logits, input_ids)

        # --- Phase 1: Compute batch-level z0 (KTO paper equation) ---
        # Key fix: compute z0 from lagged EMA, not current batch.
        #
        # Previous issue: z0 = max(0, batch_kl_of_current_batch)
        #   → In GOOD-only batches, z0 ≈ mean(logratio) ≈ logratio
        #   → logratio - z0 ≈ 0 → sigmoid(0) = 0.5 → loss = 0.5 forever
        #   → z0 rises with model, loss stays constant (self-reference loop)
        #
        # Fix: z0 = self._running_kl (EMA of previous batches, independent of current)
        #   → Current batch KL used only for EMA update (becomes next batch's z0)
        #   → logratio - z0 has meaningful values → loss responds to training
        #
        # Relation to paper: paper computes z0 from mismatched pairs (extra forward pass).
        # Lagged EMA approximates matched pairs but correctly removes self-reference.
        all_step_logratios = []

        with torch.no_grad():
            for b in range(batch_size):
                prompt_len = prompt_lengths[b].item()
                annotations = KTODataPreparer.deserialize_annotations(
                    step_annotations_json[b]
                )
                logps_offset = prompt_len - 1

                for ann in annotations:
                    start = max(logps_offset + ann.step_start_token, 0)
                    end = min(logps_offset + ann.step_end_token, token_logps.shape[1])
                    if start >= end:
                        continue

                    doc_mask_tensor = torch.tensor(
                        ann.doc_mask[:end - start], dtype=torch.bool, device=device
                    )
                    keep_mask = ~doc_mask_tensor
                    n_tokens = keep_mask.sum().item()
                    if n_tokens == 0:
                        continue

                    # Per-token mean logratio (scale-invariant)
                    kl = (token_logps[b, start:end][keep_mask] -
                          ref_token_logps[b, start:end][keep_mask]).sum().item() / n_tokens
                    all_step_logratios.append(kl)

        # z0 = lagged EMA (running KL from previous batches) — removes self-reference
        z0 = max(0.0, self._running_kl)

        # EMA update: update running_kl with current batch KL (reflected in next batch's z0)
        if all_step_logratios:
            batch_kl = sum(all_step_logratios) / len(all_step_logratios)
            self._running_kl = (1 - self._running_kl_alpha) * self._running_kl + \
                                self._running_kl_alpha * batch_kl

        self._current_z0 = z0

        # --- Phase 2: Compute KTO loss per step ---
        total_loss = 0.0  # plain float; becomes grad tensor on first addition
        total_steps = 0
        good_loss_sum = 0.0
        bad_loss_sum = 0.0
        good_count = 0
        bad_count = 0

        for b in range(batch_size):
            prompt_len = prompt_lengths[b].item()
            annotations = KTODataPreparer.deserialize_annotations(
                step_annotations_json[b]
            )
            logps_offset = prompt_len - 1

            for ann in annotations:
                start = max(logps_offset + ann.step_start_token, 0)
                end = min(logps_offset + ann.step_end_token, token_logps.shape[1])
                if start >= end:
                    continue

                doc_mask_tensor = torch.tensor(
                    ann.doc_mask[:end - start], dtype=torch.bool, device=device
                )
                keep_mask = ~doc_mask_tensor
                if keep_mask.sum() == 0:
                    continue

                n_tokens = keep_mask.sum()
                # Per-token mean logratio (matches z0 scale)
                step_logps_masked = token_logps[b, start:end][keep_mask].sum() / n_tokens
                step_ref_logps_masked = ref_token_logps[b, start:end][keep_mask].sum() / n_tokens
                logratio = step_logps_masked - step_ref_logps_masked
                # Clip to prevent over-optimization (runaway logratio → collapse)
                logratio = logratio.clamp(-10.0, 10.0)

                total_steps += 1

                if ann.label:
                    # GOOD step: chosen loss
                    step_loss = 1.0 - torch.sigmoid(self.beta * (logratio - z0))
                    total_loss = total_loss + self.desirable_weight * step_loss
                    good_loss_sum += step_loss.item()
                    good_count += 1
                else:
                    # BAD step: rejected loss
                    step_loss = 1.0 - torch.sigmoid(self.beta * (z0 - logratio))
                    total_loss = total_loss + self.lambda0 * step_loss
                    bad_loss_sum += step_loss.item()
                    bad_count += 1

        if total_steps > 0:
            total_loss = total_loss / total_steps
        else:
            # No valid steps in this batch — return dummy zero loss
            total_loss = (outputs.logits * 0).sum()

        # Log custom metrics — only once per optimizer step (not per micro-batch)
        if (self.state.global_step % self.args.logging_steps == 0
                and self.state.global_step != self._last_logged_step):
            self._last_logged_step = self.state.global_step
            self._log_custom_metrics(good_loss_sum, bad_loss_sum, good_count, bad_count)

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _log_custom_metrics(self, good_loss, bad_loss, good_count, bad_count):
        """Log KTO-specific metrics (rank 0 only in DDP)."""
        if self.args.local_rank not in (-1, 0):
            return
        metrics = {}
        if good_count > 0:
            metrics["kto/good_loss"] = good_loss / good_count
        if bad_count > 0:
            metrics["kto/bad_loss"] = bad_loss / bad_count
        metrics["kto/kl"] = self._running_kl
        metrics["kto/z0"] = self._current_z0
        metrics["kto/good_count"] = good_count
        metrics["kto/bad_count"] = bad_count
        metrics["kto/lambda_u_base"] = self.lambda0

        if metrics:
            self.log(metrics)

    def get_train_dataloader(self):
        """Override to use custom collator with DDP-aware DistributedSampler."""
        if self.args.local_rank != -1:
            # DDP mode: each process sees a disjoint subset of data
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    def _collate_fn(self, examples: List[Dict]) -> Dict[str, Any]:
        """Custom collator: combine token IDs directly to preserve exact boundaries.

        Tokenizes prompt and completion separately, then concatenates IDs
        to avoid tokenizer merging across the boundary (safe for any tokenizer).
        """
        prompts = [ex["prompt"] for ex in examples]
        completions = [ex["completion"] for ex in examples]
        step_annotations_json = [ex["step_annotations_json"] for ex in examples]

        prompt_ids_list = [
            self._tokenizer.encode(p, add_special_tokens=False) for p in prompts
        ]
        completion_ids_list = [
            self._tokenizer.encode(c, add_special_tokens=False) for c in completions
        ]
        prompt_lengths = [len(ids) for ids in prompt_ids_list]

        # Concatenate IDs directly (no re-tokenization)
        input_ids_list = [p + c for p, c in zip(prompt_ids_list, completion_ids_list)]

        # Manual padding and truncation
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_length)
        pad_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for p_len, ids in zip(prompt_lengths, input_ids_list):
            ids = ids[:max_len]
            pad_len = max_len - len(ids)

            padded_ids = ids + [pad_id] * pad_len
            attn_mask = [1] * len(ids) + [0] * pad_len

            labels = list(padded_ids)
            labels[:p_len] = [-100] * min(p_len, max_len)
            if pad_len > 0:
                labels[-pad_len:] = [-100] * pad_len

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(attn_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "step_annotations_json": step_annotations_json,
            "prompt_length": torch.tensor(prompt_lengths, dtype=torch.long),
        }
