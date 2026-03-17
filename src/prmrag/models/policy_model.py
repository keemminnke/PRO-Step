"""Policy model wrapper for trajectory generation."""

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PolicyModel:
    """Wrapper for policy model (e.g., Qwen2.5-7B-Instruct) for MC-CoT generation."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype=torch.float16,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        """Initialize policy model.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
            device: Device to load model on
            torch_dtype: Data type for model weights
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        print(f"Loading policy model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"✓ Model loaded on {device}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            top_p: Nucleus sampling (overrides default)
            stop_sequences: List of strings to stop generation

        Returns:
            Generated text (without prompt)
        """
        # Use defaults if not specified
        max_tokens = max_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part (excluding prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]

        return generated_text.strip()

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: List of strings to stop generation

        Returns:
            List of generated texts
        """
        # For simplicity, generate one by one
        # Can be optimized with batch processing later
        return [
            self.generate(prompt, max_tokens, temperature, top_p, stop_sequences)
            for prompt in prompts
        ]

    def format_prompt_for_qwen(self, user_message: str) -> str:
        """Format prompt for Qwen2.5 chat template.

        Qwen2.5-Instruct uses a specific chat format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant

        Args:
            user_message: User's message/prompt

        Returns:
            Formatted prompt
        """
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves problems step by step."},
                {"role": "user", "content": user_message}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to manual format
            return f"""<|im_start|>system
You are a helpful assistant that solves problems step by step.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

    def generate_with_chat_template(
        self,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        allow_observation: bool = False,
    ) -> str:
        """Generate using Qwen chat template.

        Args:
            user_message: User's prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: Stop sequences
            allow_observation: Unused for HF backend; kept for API parity.

        Returns:
            Generated text
        """
        prompt = self.format_prompt_for_qwen(user_message)
        return self.generate(prompt, max_tokens, temperature, top_p, stop_sequences)


def load_policy_model(config: dict) -> PolicyModel:
    """Load policy model from config.

    Args:
        config: Model configuration dict with keys:
            - model_name: HuggingFace model name
            - device: Device to use (default: "cuda")
            - temperature: Sampling temperature (default: 0.8)
            - top_p: Nucleus sampling (default: 0.95)
            - max_tokens: Max tokens per generation (default: 200)

    Returns:
        PolicyModel instance
    """
    return PolicyModel(
        model_name=config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct'),
        device=config.get('device', 'cuda'),
        torch_dtype=torch.float16,
        max_new_tokens=config.get('max_tokens', 200),
        temperature=config.get('temperature', 0.8),
        top_p=config.get('top_p', 0.95),
    )
