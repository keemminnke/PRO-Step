"""
DeepSeek-Critic Trainer (Final Optimized Version)

Key Features:
1. DeepSeek-R1 Style: Uses <think> tags to encourage CoT reasoning.
2. Loss Masking: Uses DataCollatorForCompletionOnlyLM to train ONLY on the output (Assistant), ignoring the input (User).
3. Dynamic Template: Automatically detects if the model uses <|im_start|> or <｜Assistant｜>.
4. LoRA/QLoRA: Efficient fine-tuning on GH200.
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


class DataCollatorForCompletionOnlyLM:
    """Custom Data Collator that masks input tokens and only computes loss on completion (assistant response).

    This implementation compares tokenization with/without assistant response to find
    the exact boundary, which is more reliable than searching for template tokens.
    """

    def __init__(self, response_template: str, tokenizer, mlm: bool = False, debug: bool = True):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.debug = debug
        self._debug_count = 0

        print(f"[DataCollator] Initialized (boundary detection mode)")

    def __call__(self, examples):
        import torch

        # Batch tokenization
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_tensors="pt",
        )

        # Create labels (copy of input_ids)
        labels = batch["input_ids"].clone()

        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # We need the original messages to compute the boundary
        # The examples should have 'messages' field from SFTTrainer
        for i in range(len(labels)):
            input_ids = batch["input_ids"][i]

            # Default: mask everything (will be updated if we find boundary)
            # We need to find where assistant content starts
            # Look for the boundary by finding EOS token position and working backwards
            # Or use the last segment of non-padding tokens

            # Simple approach: find the assistant response boundary by looking at token patterns
            # For DeepSeek: <｜Assistant｜> token marks the start
            # We want to supervise everything AFTER the LAST such token

            boundary = self._find_assistant_boundary(input_ids.tolist())

            if boundary > 0:
                labels[i, :boundary] = -100

                if self.debug and self._debug_count < 3:
                    self._debug_count += 1
                    non_masked = (labels[i] != -100).sum().item()
                    total = (input_ids != self.tokenizer.pad_token_id).sum().item() if self.tokenizer.pad_token_id else len(input_ids)

                    supervised_ids = input_ids[labels[i] != -100]
                    supervised_text = self.tokenizer.decode(supervised_ids)

                    print(f"\n[DataCollator Debug #{self._debug_count}]")
                    print(f"  Boundary at token: {boundary}")
                    print(f"  Supervised tokens: {non_masked}/{total}")
                    print(f"  Supervised text: {supervised_text[:300]}...")
            else:
                # Fallback: mask everything
                labels[i, :] = -100
                if self.debug and self._debug_count < 3:
                    print(f"\n[DataCollator WARNING] Could not find boundary, masking all")

        batch["labels"] = labels
        return batch

    def _find_assistant_boundary(self, input_ids: list) -> int:
        """Find where the assistant's response content starts.

        Strategy: Find the LAST occurrence of the assistant marker token,
        then return the position right after it.

        For DeepSeek models: <｜Assistant｜> is token 151670
        """
        # Known assistant marker tokens for different models
        assistant_markers = {
            151670,  # DeepSeek <｜Assistant｜>
            151644,  # Qwen <|im_start|> (need to check for 'assistant' after)
        }

        # Find the LAST occurrence of any assistant marker
        last_marker_pos = -1

        for pos, token_id in enumerate(input_ids):
            if token_id in assistant_markers:
                last_marker_pos = pos

        if last_marker_pos != -1:
            # Return position AFTER the marker
            return last_marker_pos + 1

        # Fallback: try to find by decoding and searching for patterns
        # This handles cases where the marker might be tokenized differently
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        # Look for common assistant patterns
        patterns = ['<｜Assistant｜>', '<|im_start|>assistant', 'assistant\n']

        best_pos = -1
        for pattern in patterns:
            # Find LAST occurrence
            pos = full_text.rfind(pattern)
            if pos != -1:
                # Convert text position to token position
                text_before = full_text[:pos + len(pattern)]
                tokens_before = self.tokenizer.encode(text_before, add_special_tokens=False)
                token_pos = len(tokens_before)
                if token_pos > best_pos:
                    best_pos = token_pos

        return best_pos


class DebugCallback(TrainerCallback):
    """실시간 학습 디버깅을 위한 Callback.

    Features:
    1. Loss 추이 모니터링 (매 step)
    2. 실제 예측 결과 샘플링 (매 N step)
    3. GPU 메모리 사용량 출력
    4. Gradient norm 추적
    """

    def __init__(self, tokenizer, eval_samples, eval_interval=100, num_samples=3):
        """
        Args:
            tokenizer: 토크나이저
            eval_samples: 평가용 샘플 리스트 (messages 포함)
            eval_interval: 몇 step마다 샘플 예측할지
            num_samples: 예측할 샘플 수
        """
        self.tokenizer = tokenizer
        self.eval_samples = eval_samples[:num_samples] if eval_samples else []
        self.eval_interval = eval_interval
        self.loss_history = []
        self.grad_norm_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """매 로깅 step마다 호출."""
        if logs:
            loss = logs.get('loss')
            grad_norm = logs.get('grad_norm')

            if loss is not None:
                self.loss_history.append(loss)

            if grad_norm is not None:
                self.grad_norm_history.append(grad_norm)

            # GPU 메모리 사용량
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3

                print(f"\n{'='*60}")
                print(f"[Step {state.global_step}] Loss: {loss:.4f}" if loss else f"[Step {state.global_step}]")
                if grad_norm:
                    print(f"  Grad Norm: {grad_norm:.4f}")
                print(f"  GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")

                # Loss 추이 요약 (최근 10개)
                if len(self.loss_history) >= 10:
                    recent = self.loss_history[-10:]
                    print(f"  Recent Loss (last 10): {sum(recent)/len(recent):.4f} (min: {min(recent):.4f}, max: {max(recent):.4f})")
                print(f"{'='*60}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """매 step 끝에 호출. 주기적으로 샘플 예측 수행."""
        if state.global_step % self.eval_interval == 0 and state.global_step > 0:
            if model and self.eval_samples:
                self._run_sample_predictions(model, state.global_step)

    def _run_sample_predictions(self, model, step):
        """샘플에 대해 실제 예측 수행."""
        print(f"\n{'#'*70}")
        print(f"# SAMPLE PREDICTIONS at Step {step}")
        print(f"{'#'*70}")

        model.eval()

        for i, sample in enumerate(self.eval_samples):
            messages = sample['messages']
            expected_label = sample.get('label', 'UNKNOWN')

            # User message만 추출 (assistant 제외)
            input_messages = [m for m in messages if m['role'] != 'assistant']

            # 입력 생성
            input_text = self.tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)

            # Use autocast for FlashAttention compatibility (requires bf16/fp16)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Label 추출 (1 = GOOD, 0 = BAD)
            predicted_label = -1  # Unknown
            if "Label: 1" in generated or "Label:1" in generated:
                predicted_label = 1
            elif "Label: 0" in generated or "Label:0" in generated:
                predicted_label = 0

            match = "✓" if predicted_label == expected_label else "✗"

            print(f"\n[Sample {i+1}] Expected: {expected_label} | Predicted: {predicted_label} {match}")
            print(f"Generated (truncated):")
            print(f"  {generated[:300]}...")

        model.train()
        print(f"\n{'#'*70}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 최종 통계 출력."""
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total steps: {state.global_step}")
        print(f"Total epochs: {state.epoch:.2f}")

        if self.loss_history:
            print(f"\nLoss Statistics:")
            print(f"  Initial: {self.loss_history[0]:.4f}")
            print(f"  Final: {self.loss_history[-1]:.4f}")
            print(f"  Min: {min(self.loss_history):.4f}")
            print(f"  Max: {max(self.loss_history):.4f}")
            print(f"  Avg: {sum(self.loss_history)/len(self.loss_history):.4f}")

            # Loss 감소율
            if len(self.loss_history) > 1:
                reduction = (self.loss_history[0] - self.loss_history[-1]) / self.loss_history[0] * 100
                print(f"  Reduction: {reduction:.1f}%")

        if self.grad_norm_history:
            print(f"\nGradient Norm Statistics:")
            print(f"  Avg: {sum(self.grad_norm_history)/len(self.grad_norm_history):.4f}")
            print(f"  Max: {max(self.grad_norm_history):.4f}")
        print(f"{'='*70}")


@dataclass
class CriticTrainingConfig:
    """Configuration for critic model training."""

    # Model settings
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # or 8B depending on exact repo

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16  # alpha = r for balanced scaling
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # Training settings
    output_dir: str = "outputs/critic_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # DeepSeek-R1 generates long chains of thought, so we need a long context.
    max_length: int = 4096 

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Optimization
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "prmrag-critic"
    wandb_run_name: Optional[str] = None  # Auto-generated if None


class CriticDataFormatter:
    """Format trajectory data for DeepSeek-R1 critic training."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def format_step_for_training(
        self,
        question: str,
        previous_steps: List[Dict[str, Any]],
        current_step: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Constructs the chat messages using XML format.
        Structure:
        System: Define role.
        User: Context (Question + History + Current Step) in XML tags.
        Assistant: Reasoning + Label.
        """

        # 1. Build User Context (XML format)
        input_parts = [f"Question: {question}", ""]

        if previous_steps:
            input_parts.append("Previous Steps:")
            for i, prev_step in enumerate(previous_steps, 1):
                input_parts.append(f"## Step {i}")

                # <think>
                if prev_step.get('think'):
                    input_parts.append(f"<think>{prev_step['think']}</think>")

                # <search> or <answer>
                if prev_step.get('search'):
                    input_parts.append(f"<search>{prev_step['search']}</search>")
                elif prev_step.get('answer'):
                    input_parts.append(f"<answer>{prev_step['answer']}</answer>")

                # <documents>
                if prev_step.get('documents'):
                    input_parts.append(f"<documents>{prev_step['documents']}</documents>")
                input_parts.append("")

        input_parts.append("Current Step to Evaluate:")
        input_parts.append(f"## Step {len(previous_steps) + 1}")

        # <think>
        if current_step.get('think'):
            input_parts.append(f"<think>{current_step['think']}</think>")

        # <search> or <answer>
        if current_step.get('search'):
            input_parts.append(f"<search>{current_step['search']}</search>")
        elif current_step.get('answer'):
            input_parts.append(f"<answer>{current_step['answer']}</answer>")

        # <documents>
        if current_step.get('documents'):
            input_parts.append(f"<documents>{current_step['documents']}</documents>")

        input_parts.append("")
        input_parts.append("Task: Evaluate the quality of the Current Step. Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")

        user_content = "\n".join(input_parts)

        # 2. Build Assistant Response
        # Loss is computed on BOTH reasoning AND Label
        # NOTE: We use [REASONING] instead of <think> because <think> is a special token
        # in DeepSeek models, which causes tokenization issues
        label = current_step.get('judge_label', 0)
        reasoning = current_step.get('judge_reasoning', '')

        # Use [REASONING] tags to avoid special token conflicts
        assistant_content = f"[REASONING]\n{reasoning}\n[/REASONING]\nLabel: {label}"

        system_content = (
            "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
            "The trajectory uses XML tags: <think> for reasoning, <search> for queries, <answer> for final answers, <documents> for retrieved passages. "
            "Analyze each step's logical soundness and evidence grounding. "
            "First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

    def prepare_training_data(self, data_file: Path, truncate_after_bad: bool = False) -> List[Dict[str, Any]]:
        training_samples = []
        label_counts = {1: 0, 0: 0}  # 1=good, 0=bad
        truncated_steps = 0

        print(f"Reading data from {data_file}...")
        if truncate_after_bad:
            print(f"  [truncate_after_bad=True] Discarding steps after first BAD step")
        with open(data_file) as f:
            for line in f:
                try:
                    trajectory = json.loads(line)
                    question = trajectory['question']
                    steps = trajectory['steps']

                    for idx, step in enumerate(steps):
                        # Convert label: GOOD->1, BAD->0
                        raw_label = step.get('judge_label')
                        if raw_label == 'GOOD' or raw_label == 1:
                            label = 1
                        elif raw_label == 'BAD' or raw_label == 0:
                            label = 0
                        else:
                            continue  # Skip invalid labels

                        # If truncate_after_bad: include the first BAD step, skip everything after
                        if truncate_after_bad and label == 0:
                            # Include this BAD step as training sample
                            step_with_label = {**step, 'judge_label': label}
                            messages = self.format_step_for_training(
                                question=question,
                                previous_steps=steps[:idx],
                                current_step=step_with_label,
                            )
                            training_samples.append({
                                'messages': messages,
                                'question_id': trajectory.get('question_id', trajectory.get('trajectory_id')),
                                'step_num': step.get('step_id', idx + 1),
                                'label': label,
                            })
                            label_counts[label] += 1
                            truncated_steps += len(steps) - idx - 1
                            break  # Stop processing this trajectory

                        # Update step with numeric label for formatting
                        step_with_label = {**step, 'judge_label': label}

                        messages = self.format_step_for_training(
                            question=question,
                            previous_steps=steps[:idx],
                            current_step=step_with_label,
                        )

                        training_samples.append({
                            'messages': messages,  # SFTTrainer expects 'messages' column
                            'question_id': trajectory.get('question_id', trajectory.get('trajectory_id')),
                            'step_num': step.get('step_id', idx + 1),
                            'label': label,  # 1=good, 0=bad
                        })
                        label_counts[label] += 1
                except json.JSONDecodeError:
                    continue

        print(f"  Data Stats: 1(good)={label_counts[1]}, 0(bad)={label_counts[0]}")
        if truncate_after_bad:
            print(f"  Truncated steps (discarded after first BAD): {truncated_steps}")
        return training_samples


class CriticTrainer:
    """Trainer for critic model with auto-masking and LoRA."""

    def __init__(self, config: CriticTrainingConfig):
        self.config = config
        print(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT: Set truncation_side="left" to preserve Assistant output (training target)
        # When sequence exceeds max_length, truncate from the left (input) instead of right (output)
        self.tokenizer.truncation_side = "left"
        # Set padding_side="right" for proper batch padding
        self.tokenizer.padding_side = "right"

        self.formatter = CriticDataFormatter(self.tokenizer)
        self.model = None

    def _get_response_template(self) -> str:
        """
        Detect the correct response template based on model's chat template.

        Different models use different assistant markers:
        - Qwen: <|im_start|>assistant
        - DeepSeek: <｜Assistant｜> or similar

        We test by applying the chat template and looking for the pattern.
        """
        # Test with a simple message to detect the format
        test_messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "ASSISTANT_CONTENT_START"}
        ]

        try:
            formatted = self.tokenizer.apply_chat_template(
                test_messages, tokenize=False, add_generation_prompt=False
            )

            # Find where assistant content starts
            marker_pos = formatted.find("ASSISTANT_CONTENT_START")
            if marker_pos == -1:
                print("[WARNING] Could not detect assistant marker, using fallback")
                return "<|im_start|>assistant\n"

            # Extract the template (text just before the content)
            # Look backwards from marker to find the assistant start marker
            search_region = formatted[max(0, marker_pos - 100):marker_pos]

            # Common patterns to look for
            patterns = [
                "<|im_start|>assistant\n",  # Qwen style
                "<｜Assistant｜>",           # DeepSeek style (full-width)
                "<|Assistant|>",             # DeepSeek style (ASCII)
                "assistant\n",               # Simple
            ]

            for pattern in patterns:
                if pattern in search_region:
                    print(f"[Response Template] Detected: {repr(pattern)}")
                    return pattern

            # Fallback: use the last 30 chars before content as template
            fallback = search_region[-30:] if len(search_region) >= 30 else search_region
            print(f"[Response Template] Using fallback: {repr(fallback)}")
            return fallback

        except Exception as e:
            print(f"[WARNING] Template detection failed: {e}, using default")
            return "<|im_start|>assistant\n"

    def prepare_data(self, train_file: Path, eval_file: Optional[Path] = None, truncate_after_bad: bool = False) -> tuple:
        print(f"\nPreparing training data from: {train_file}")
        train_samples = self.formatter.prepare_training_data(train_file, truncate_after_bad=truncate_after_bad)
        print(f"✓ Prepared {len(train_samples)} training samples")
        train_dataset = Dataset.from_list(train_samples)

        eval_dataset = None
        if eval_file and eval_file.exists():
            print(f"\nPreparing eval data from: {eval_file}")
            eval_samples = self.formatter.prepare_training_data(eval_file, truncate_after_bad=truncate_after_bad)
            print(f"✓ Prepared {len(eval_samples)} eval samples")
            eval_dataset = Dataset.from_list(eval_samples)

        return train_dataset, eval_dataset

    def load_model(self):
        print(f"\nLoading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" # Highly recommended for GH200
        )

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora_target_modules,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        print("✓ Model loaded with LoRA")

    def _verify_masking(self, dataset: Dataset, collator: DataCollatorForCompletionOnlyLM, num_samples: int = 5):
        """Verify that masking is working correctly before training."""
        print(f"Checking {num_samples} samples for proper masking...")

        total_tokens = 0
        total_trained = 0
        all_masked_count = 0

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            messages = sample['messages']

            # Tokenize the sample
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )

            # Apply collator to get labels
            batch = collator([{
                'input_ids': tokenized['input_ids'][0].tolist(),
                'attention_mask': tokenized['attention_mask'][0].tolist(),
            }])

            labels = batch['labels'][0]
            num_tokens = len(labels)
            num_trained = (labels != -100).sum().item()

            total_tokens += num_tokens
            total_trained += num_trained

            if num_trained == 0:
                all_masked_count += 1
                print(f"  Sample {i+1}: ALL MASKED! ({num_tokens} tokens)")
                # Show what we're looking for
                print(f"    Text preview: {text[:200]}...")
            else:
                print(f"  Sample {i+1}: {num_trained}/{num_tokens} tokens trained ({100*num_trained/num_tokens:.1f}%)")

        print(f"\nMasking Summary:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Trained tokens: {total_trained} ({100*total_trained/total_tokens:.1f}%)")
        print(f"  Fully masked samples: {all_masked_count}/{num_samples}")

        if all_masked_count == num_samples:
            print("\n" + "!" * 70)
            print("CRITICAL ERROR: ALL SAMPLES ARE FULLY MASKED!")
            print("This means the response template is not being found.")
            print("Training will result in 0 loss (nothing to learn).")
            print("!" * 70)

            # Show what the template looks like in actual data
            sample_text = self.tokenizer.apply_chat_template(
                dataset[0]['messages'], tokenize=False, add_generation_prompt=False
            )
            print(f"\nActual text structure:")
            print(sample_text[:1000])
            raise ValueError("Masking verification failed - response template not found in any sample")

        if total_trained / total_tokens < 0.05:
            print("\n[WARNING] Less than 5% of tokens are being trained. Check if this is expected.")

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, debug_interval: int = 100):
        if self.model is None:
            self.load_model()

        # [CRITICAL] Setup Data Collator for Loss Masking
        # This ensures we only calculate loss on the Assistant's response (Reasoning + Label),
        # not on the User's prompt.
        response_template = self._get_response_template()

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

        # Setup wandb logging
        report_to = []
        if self.config.use_wandb and WANDB_AVAILABLE:
            report_to.append("wandb")
            run_name = self.config.wandb_run_name or f"critic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "model_name": self.config.model_name,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "warmup_ratio": self.config.warmup_ratio,
                    "lr_scheduler": "cosine",
                    "batch_size": self.config.per_device_train_batch_size,
                    "gradient_accumulation": self.config.gradient_accumulation_steps,
                    "effective_batch_size": self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps,
                    "max_length": self.config.max_length,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "num_epochs": self.config.num_train_epochs,
                    "num_train_samples": len(train_dataset),
                },
                tags=["critic", "prmrag", self.config.model_name.split("/")[-1]],
            )
            print(f"✓ Wandb initialized: {self.config.wandb_project}/{run_name}")
        else:
            report_to.append("tensorboard")
            if self.config.use_wandb and not WANDB_AVAILABLE:
                print("Warning: wandb requested but not available, using tensorboard")

        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",  # Cosine decay for smoother training
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_total_limit=2,
            max_length=self.config.max_length,
            report_to=report_to,
            remove_unused_columns=True,
        )

        # 디버깅용 샘플 준비 (1=GOOD / 0=BAD 각각 포함)
        debug_samples = []
        good_sample = None
        bad_sample = None
        for sample in train_dataset:
            if sample.get('label') == 1 and good_sample is None:
                good_sample = sample
            elif sample.get('label') == 0 and bad_sample is None:
                bad_sample = sample
            if good_sample and bad_sample:
                break
        if good_sample:
            debug_samples.append(good_sample)
        if bad_sample:
            debug_samples.append(bad_sample)

        # DebugCallback 생성
        debug_callback = DebugCallback(
            tokenizer=self.tokenizer,
            eval_samples=debug_samples,
            eval_interval=debug_interval,
            num_samples=2,
        )

        print("\n" + "=" * 70)
        print(f"STARTING TRAINING with template: {repr(response_template)}")
        print(f"Debug interval: {debug_interval} steps")
        print("=" * 70)

        # CRITICAL: Verify masking is working correctly before training
        print("\n[MASKING VERIFICATION]")
        self._verify_masking(train_dataset, collator)

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=collator, # Apply masking
            callbacks=[debug_callback],  # 디버깅 callback 추가
        )

        # Check for resume from checkpoint
        resume_checkpoint = None
        checkpoint_dir = Path(self.config.output_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"\n✓ Resuming from checkpoint: {resume_checkpoint}")

        trainer.train(resume_from_checkpoint=resume_checkpoint)

        # Save Final Model
        final_path = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        print(f"\n✓ Final model saved to: {final_path}")
        
        # Save info
        info = {
            'config': {
                'model_name': self.config.model_name,
                'lora_r': self.config.lora_r,
                'max_length': self.config.max_length,
                'learning_rate': self.config.learning_rate,
            },
            'training': {
                'completed_at': datetime.now().isoformat(),
                'samples': len(train_dataset)
            }
        }
        with open(Path(self.config.output_dir) / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        # Finish wandb run
        if self.config.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
            print("✓ Wandb run finished")


def create_critic_trainer(model_name: str, output_dir: str, **kwargs) -> CriticTrainer:
    config = CriticTrainingConfig(model_name=model_name, output_dir=output_dir, **kwargs)
    return CriticTrainer(config)

if __name__ == "__main__":
    # Example usage
    trainer = create_critic_trainer(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        output_dir="outputs/critic"
    )
    
    # Assuming data exists
    train_file = Path("data/merged_consensus_data.jsonl")
    if train_file.exists():
        train_ds, _ = trainer.prepare_data(train_file)
        trainer.train(train_ds)
