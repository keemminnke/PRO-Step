"""Step-level DPO Trainer for Policy Model.

Trajectory-level input (chosen/rejected pairs) with document masking in loss.
Key feature: <documents>...</documents> tokens excluded from log probability
computation, so the model isn't trained on retrieved passages.

Pairing strategies for data preparation:
- best-vs-all: best chosen (highest critic_min) vs all rejected
- best-vs-worst: best chosen vs worst rejected (1 pair per question)
- all-vs-all: all chosen vs all rejected (cartesian product)
"""

import json
from collections import defaultdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
)

from prmrag.training.kto_trainer import SYSTEM_PROMPT


def _get_per_token_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probabilities.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) - token IDs

    Returns:
        (batch, seq_len-1) - log probs for each predicted token
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    per_token_logps = log_probs.gather(
        dim=-1, index=labels[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return per_token_logps


class DPODataPreparer:
    """Trajectory + critic scores -> DPO training pairs."""

    def __init__(self, tokenizer, system_prompt: str = SYSTEM_PROMPT):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def prepare_dataset(
        self,
        trajectories_path: str,
        critic_scores_path: str,
        pairing: str = "best-vs-all",
        limit: Optional[int] = None,
    ) -> Dataset:
        """Convert trajectories + critic scores into DPO training dataset.

        Args:
            trajectories_path: Path to judge_labels JSONL (step content)
            critic_scores_path: Path to critic_scores per_trajectory JSONL
            pairing: Pairing strategy (best-vs-all | best-vs-worst | all-vs-all)
            limit: Limit number of pairs for debugging

        Returns:
            Dataset with prompt, chosen, rejected columns
        """
        from prmrag.regeneration.classifier import QuestionClassifier

        scored_trajs = QuestionClassifier.load_and_merge(
            trajectories_path, critic_scores_path
        )

        # Group by question_id
        by_question: Dict[str, List] = defaultdict(list)
        for traj in scored_trajs:
            by_question[traj.question_id].append(traj)

        pairs = []
        stats = {
            "total_questions": len(by_question),
            "questions_with_pairs": 0,
            "total_pairs": 0,
            "skipped_no_correct": 0,
            "skipped_no_incorrect": 0,
        }

        for qid, trajs in sorted(by_question.items()):
            correct = [t for t in trajs if t.is_correct]
            incorrect = [t for t in trajs if not t.is_correct]

            if not correct:
                stats["skipped_no_correct"] += 1
                continue
            if not incorrect:
                stats["skipped_no_incorrect"] += 1
                continue

            stats["questions_with_pairs"] += 1

            # Sort by critic_min for selection
            correct.sort(key=lambda t: t.critic_min, reverse=True)
            incorrect.sort(key=lambda t: t.critic_min)  # worst first

            question = trajs[0].question
            prompt = self._build_prompt(question)

            if pairing == "best-vs-all":
                best = correct[0]
                chosen_text = self._build_completion(best.steps)
                for rej in incorrect:
                    rejected_text = self._build_completion(rej.steps)
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                    })

            elif pairing == "best-vs-worst":
                best = correct[0]
                worst = incorrect[0]
                chosen_text = self._build_completion(best.steps)
                rejected_text = self._build_completion(worst.steps)
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                })

            elif pairing == "all-vs-all":
                for cho in correct:
                    chosen_text = self._build_completion(cho.steps)
                    for rej in incorrect:
                        rejected_text = self._build_completion(rej.steps)
                        pairs.append({
                            "prompt": prompt,
                            "chosen": chosen_text,
                            "rejected": rejected_text,
                        })
            else:
                raise ValueError(f"Unknown pairing strategy: {pairing}")

        stats["total_pairs"] = len(pairs)

        if limit and len(pairs) > limit:
            pairs = pairs[:limit]

        # Print stats
        print(f"\n{'='*60}")
        print(f"DPO Data Preparation Stats:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Questions with pairs: {stats['questions_with_pairs']}")
        print(f"  Skipped (no correct): {stats['skipped_no_correct']}")
        print(f"  Skipped (no incorrect): {stats['skipped_no_incorrect']}")
        print(f"  Total pairs generated: {stats['total_pairs']}")
        print(f"  Pairing strategy: {pairing}")
        if limit and stats['total_pairs'] > limit:
            print(f"  Limited to: {limit}")
        print(f"{'='*60}\n")

        return Dataset.from_list(pairs)

    def _build_prompt(self, question: str) -> str:
        """Build the prompt (system + question) using chat template."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def _build_completion(self, steps: List[Dict]) -> str:
        """Convert steps to XML format trajectory text."""
        parts = []
        for step in steps:
            think = step.get("think", "")
            if think:
                parts.append(f"<think>{think}</think>")

            search = step.get("search", "")
            answer = step.get("answer", "")
            if search:
                parts.append(f"<search>{search}</search>")
            elif answer:
                parts.append(f"<answer>{answer}</answer>")

            docs = step.get("documents", "")
            if docs:
                parts.append(f"<documents>{docs}</documents>")

        return "\n".join(parts)


class DPODebugCallback(TrainerCallback):
    """Callback for monitoring DPO training metrics."""

    def __init__(self):
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            if loss is not None:
                self.loss_history.append(loss)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3

                print(f"\n{'='*60}")
                print(f"[Step {state.global_step}] Loss: {loss:.4f}" if loss else f"[Step {state.global_step}]")
                cho_reward = logs.get("dpo/chosen_reward")
                rej_reward = logs.get("dpo/rejected_reward")
                margin = logs.get("dpo/reward_margin")
                if cho_reward is not None:
                    print(f"  Chosen reward:   {cho_reward:.4f}")
                if rej_reward is not None:
                    print(f"  Rejected reward: {rej_reward:.4f}")
                if margin is not None:
                    print(f"  Margin:          {margin:.4f}")
                doc_pct = logs.get("dpo/doc_tokens_masked_pct")
                if doc_pct is not None:
                    print(f"  Doc masked:      {doc_pct:.1f}%")
                print(f"  GPU: {allocated:.2f}GB / {reserved:.2f}GB")
                print(f"{'='*60}")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*70}")
        print("DPO TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total steps: {state.global_step}")
        if self.loss_history:
            print(f"Loss: {self.loss_history[0]:.4f} -> {self.loss_history[-1]:.4f}")
            print(f"Min: {min(self.loss_history):.4f}, Max: {max(self.loss_history):.4f}")
        print(f"{'='*70}")


class StepLevelDPOTrainer(Trainer):
    """Custom Trainer implementing DPO loss with document masking.

    Unlike trl DPOTrainer which computes loss on all completion tokens,
    this trainer masks <documents>...</documents> spans so the model is only
    trained on reasoning (think, search, answer) tokens.

    Loss: -log sigmoid(beta * (sum_masked_logratios_chosen - sum_masked_logratios_rejected))
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        beta: float = 0.1,
        max_length: int = 4096,
        **kwargs,
    ):
        super().__init__(model=model, processing_class=tokenizer, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.max_length = max_length
        self._tokenizer = tokenizer
        self._doc_start = "<documents>"
        self._doc_end = "</documents>"

        # Freeze ref model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def _find_doc_mask(self, completion_text: str, completion_ids: List[int]) -> List[bool]:
        """Find <documents>...</documents> spans in completion and return boolean mask.

        Returns:
            List[bool] of length len(completion_ids). True = document token (mask out).
        """
        n = len(completion_ids)
        mask = [False] * n
        search_start = 0

        while True:
            start = completion_text.find(self._doc_start, search_start)
            if start == -1:
                break
            end_pos = completion_text.find(self._doc_end, start)
            if end_pos == -1:
                # No closing tag: mask to end
                prefix = completion_text[:start]
                start_tok = len(self._tokenizer.encode(prefix, add_special_tokens=False))
                for j in range(start_tok, n):
                    mask[j] = True
                break

            end = end_pos + len(self._doc_end)
            prefix_before = completion_text[:start]
            prefix_after = completion_text[:end]
            start_tok = len(self._tokenizer.encode(prefix_before, add_special_tokens=False))
            end_tok = len(self._tokenizer.encode(prefix_after, add_special_tokens=False))
            for j in range(start_tok, min(end_tok, n)):
                mask[j] = True
            search_start = end

        return mask

    def get_train_dataloader(self):
        """Override to use custom collator."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    def _collate_fn(self, examples: List[Dict]) -> Dict[str, Any]:
        """Tokenize chosen/rejected pairs and create document masks.

        For each pair, creates:
        - input_ids, attention_mask (for model forward)
        - doc_mask: True for prompt + document + padding tokens (excluded from loss)
        """
        pad_id = self._tokenizer.pad_token_id or 0

        all_cho_ids, all_cho_attn, all_cho_doc = [], [], []
        all_rej_ids, all_rej_attn, all_rej_doc = [], [], []

        for ex in examples:
            prompt = ex["prompt"]
            chosen = ex["chosen"]
            rejected = ex["rejected"]

            prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            plen = len(prompt_ids)

            cho_ids = self._tokenizer.encode(
                prompt + chosen, add_special_tokens=False
            )[:self.max_length]
            rej_ids = self._tokenizer.encode(
                prompt + rejected, add_special_tokens=False
            )[:self.max_length]

            # Doc mask on completion part
            cho_comp_ids = cho_ids[plen:]
            rej_comp_ids = rej_ids[plen:]
            cho_doc = self._find_doc_mask(chosen, cho_comp_ids)
            rej_doc = self._find_doc_mask(rejected, rej_comp_ids)

            # Full mask: prompt=True, completion=doc_mask, (pad will be added below)
            cho_full_doc = [True] * plen + cho_doc
            rej_full_doc = [True] * plen + rej_doc

            all_cho_ids.append(cho_ids)
            all_cho_attn.append([1] * len(cho_ids))
            all_cho_doc.append(cho_full_doc)
            all_rej_ids.append(rej_ids)
            all_rej_attn.append([1] * len(rej_ids))
            all_rej_doc.append(rej_full_doc)

        # Pad to max length in batch
        max_cho = max(len(ids) for ids in all_cho_ids)
        max_rej = max(len(ids) for ids in all_rej_ids)

        for i in range(len(all_cho_ids)):
            pad = max_cho - len(all_cho_ids[i])
            all_cho_ids[i] += [pad_id] * pad
            all_cho_attn[i] += [0] * pad
            all_cho_doc[i] += [True] * pad  # pad tokens masked

            pad = max_rej - len(all_rej_ids[i])
            all_rej_ids[i] += [pad_id] * pad
            all_rej_attn[i] += [0] * pad
            all_rej_doc[i] += [True] * pad

        return {
            "chosen_input_ids": torch.tensor(all_cho_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(all_cho_attn, dtype=torch.long),
            "chosen_doc_mask": torch.tensor(all_cho_doc, dtype=torch.bool),
            "rejected_input_ids": torch.tensor(all_rej_ids, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(all_rej_attn, dtype=torch.long),
            "rejected_doc_mask": torch.tensor(all_rej_doc, dtype=torch.bool),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute DPO loss with document masking.

        1. Forward chosen/rejected through model and ref_model
        2. Compute per-token log probs
        3. Mask prompt + document + padding tokens
        4. Sum log ratios (model vs ref) for chosen and rejected
        5. DPO loss: -log sigmoid(beta * (chosen_logratio - rejected_logratio))
        """
        cho_ids = inputs["chosen_input_ids"]
        cho_attn = inputs["chosen_attention_mask"]
        cho_doc = inputs["chosen_doc_mask"]  # True = mask out

        rej_ids = inputs["rejected_input_ids"]
        rej_attn = inputs["rejected_attention_mask"]
        rej_doc = inputs["rejected_doc_mask"]

        device = cho_ids.device
        batch_size = cho_ids.shape[0]

        # Forward: model
        cho_logits = model(input_ids=cho_ids, attention_mask=cho_attn).logits
        rej_logits = model(input_ids=rej_ids, attention_mask=rej_attn).logits

        # Forward: ref model (no grad)
        # In DDP, ref_model is on same device as inputs (local_rank GPU)
        ref_device = next(self.ref_model.parameters()).device
        if ref_device != device:
            self.ref_model = self.ref_model.to(device)
            ref_device = device
        with torch.no_grad():
            cho_ref_logits = self.ref_model(
                input_ids=cho_ids,
                attention_mask=cho_attn,
            ).logits
            rej_ref_logits = self.ref_model(
                input_ids=rej_ids,
                attention_mask=rej_attn,
            ).logits

        # Per-token log probs (shifted: logps[i] predicts token at position i+1)
        cho_logps = _get_per_token_logps(cho_logits, cho_ids)
        cho_ref_logps = _get_per_token_logps(cho_ref_logits, cho_ids)
        rej_logps = _get_per_token_logps(rej_logits, rej_ids)
        rej_ref_logps = _get_per_token_logps(rej_ref_logits, rej_ids)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        chosen_rewards_sum = 0.0
        rejected_rewards_sum = 0.0
        doc_masked_count = 0
        completion_token_count = 0

        for b in range(batch_size):
            cho_len = cho_logps.shape[1]
            rej_len = rej_logps.shape[1]

            # Keep mask for logps: logps[i] predicts token[i+1]
            # Keep logps[i] if token[i+1] is NOT masked (not prompt/doc/pad)
            cho_keep = ~cho_doc[b, 1:cho_len + 1]
            cho_keep = cho_keep & cho_attn[b, 1:cho_len + 1].bool()

            rej_keep = ~rej_doc[b, 1:rej_len + 1]
            rej_keep = rej_keep & rej_attn[b, 1:rej_len + 1].bool()

            if cho_keep.sum() == 0 or rej_keep.sum() == 0:
                continue

            # Masked log ratios
            cho_logratio = (cho_logps[b][cho_keep] - cho_ref_logps[b][cho_keep]).sum()
            rej_logratio = (rej_logps[b][rej_keep] - rej_ref_logps[b][rej_keep]).sum()

            # DPO loss
            pair_loss = -F.logsigmoid(self.beta * (cho_logratio - rej_logratio))
            total_loss = total_loss + pair_loss

            # Track metrics
            with torch.no_grad():
                chosen_rewards_sum += cho_logratio.item()
                rejected_rewards_sum += rej_logratio.item()

                # Doc masking stats (how many completion tokens are masked)
                cho_in_seq = cho_attn[b, 1:cho_len + 1].bool()
                doc_masked_count += int((cho_doc[b, 1:cho_len + 1] & cho_in_seq).sum().item())
                completion_token_count += int(cho_in_seq.sum().item())

        total_loss = total_loss / max(batch_size, 1)

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "dpo/chosen_reward": chosen_rewards_sum / max(batch_size, 1),
                "dpo/rejected_reward": rejected_rewards_sum / max(batch_size, 1),
                "dpo/reward_margin": (chosen_rewards_sum - rejected_rewards_sum) / max(batch_size, 1),
            }
            if completion_token_count > 0:
                metrics["dpo/doc_tokens_masked_pct"] = doc_masked_count / completion_token_count * 100
            self.log(metrics)

        if return_outputs:
            return total_loss, cho_logits
        return total_loss
