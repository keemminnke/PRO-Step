"""SFT Trainer for Policy Model — train on all-GOOD trajectories only.

Loss is computed only on GOOD step tokens (documents masked out).
No ref_model needed — standard cross-entropy SFT.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer

# Reuse system prompt from kto_trainer
from prmrag.training.kto_trainer import SYSTEM_PROMPT


class SFTDataPreparer:
    """Load trajectories → filter all-GOOD → build SFT dataset."""

    def __init__(self, tokenizer: AutoTokenizer, system_prompt: str = SYSTEM_PROMPT):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def prepare_dataset(
        self,
        input_paths: List[str],
        filter_all_good: bool = True,
        limit: Optional[int] = None,
    ) -> Dataset:
        samples = []
        skipped = 0
        filtered = 0

        for path in input_paths:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    steps = record.get("steps", [])

                    # Resolve labels
                    if "generative_step_labels" in record:
                        labels = record["generative_step_labels"]
                    else:
                        labels = [s.get("generative_label", 1) for s in steps]

                    # Filter: only all-GOOD trajectories
                    if filter_all_good and not all(l == 1 for l in labels):
                        filtered += 1
                        continue

                    if not steps:
                        skipped += 1
                        continue

                    prompt = self._build_prompt(record["question"])
                    completion = self._build_completion(steps)

                    if not completion.strip():
                        skipped += 1
                        continue

                    # Find document spans in completion token space
                    completion_ids = self.tokenizer.encode(
                        completion, add_special_tokens=False
                    )
                    doc_spans = self._find_document_spans(completion, completion_ids)

                    samples.append({
                        "prompt": prompt,
                        "completion": completion,
                        "doc_spans_json": json.dumps(doc_spans),
                        "trajectory_id": record.get("trajectory_id", ""),
                        "question": record.get("question", ""),
                    })
                    count += 1

            print(f"  {path.split('/')[-1]}: {count} trajectories loaded")

        print(f"\n{'='*60}")
        print(f"SFT Data Preparation Stats:")
        print(f"  All-GOOD trajectories : {len(samples)}")
        print(f"  Filtered (has BAD)    : {filtered}")
        print(f"  Skipped (empty/error) : {skipped}")
        print(f"{'='*60}\n")

        if limit:
            samples = samples[:limit]
            print(f"  Limited to {len(samples)} samples")

        return Dataset.from_list(samples)

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_completion(self, steps: List[Dict]) -> str:
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

    def _find_document_spans(
        self, completion_text: str, completion_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """Find <documents>...</documents> token spans to mask from loss."""
        spans = []
        search_start = 0
        while True:
            doc_start = completion_text.find("<documents>", search_start)
            if doc_start == -1:
                break
            doc_end_pos = completion_text.find("</documents>", doc_start)
            if doc_end_pos == -1:
                prefix = completion_text[:doc_start]
                start_token = len(
                    self.tokenizer.encode(prefix, add_special_tokens=False)
                )
                spans.append((start_token, len(completion_ids)))
                break
            doc_end = doc_end_pos + len("</documents>")
            start_token = len(
                self.tokenizer.encode(
                    completion_text[:doc_start], add_special_tokens=False
                )
            )
            end_token = len(
                self.tokenizer.encode(
                    completion_text[:doc_end], add_special_tokens=False
                )
            )
            if start_token < end_token:
                spans.append((start_token, end_token))
            search_start = doc_end
        return spans


class SFTDataCollator:
    """Collator: mask prompt + document tokens, supervise GOOD step tokens only."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, examples: List[Dict]) -> Dict[str, Any]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ex in examples:
            prompt_ids = self.tokenizer.encode(
                ex["prompt"], add_special_tokens=False
            )
            completion_ids = self.tokenizer.encode(
                ex["completion"], add_special_tokens=False
            )
            doc_spans = json.loads(ex["doc_spans_json"])

            # Concatenate and truncate
            input_ids = (prompt_ids + completion_ids)[: self.max_length]
            prompt_len = min(len(prompt_ids), self.max_length)
            pad_len = self.max_length - len(input_ids)

            # Build labels: -100 for prompt & doc spans, token id otherwise
            labels = [-100] * len(input_ids)
            for i in range(prompt_len, len(input_ids)):
                comp_idx = i - prompt_len
                in_doc = any(s <= comp_idx < e for s, e in doc_spans)
                if not in_doc:
                    labels[i] = input_ids[i]

            batch_input_ids.append(input_ids + [self.pad_id] * pad_len)
            batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
