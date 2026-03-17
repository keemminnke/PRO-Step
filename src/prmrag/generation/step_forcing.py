"""Step forcing utilities for structured solution generation.

This module implements the step forcing approach where we explicitly
guide the model to generate solutions in a structured format:
Step 1: {content}
Step 2: {content}
...
Step T: {content}

This ensures semantic step boundaries instead of arbitrary "\n\n" splits.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ForcedStep:
    """A step generated with step forcing."""
    step_number: int
    content: str

    def __str__(self) -> str:
        return f"Step {self.step_number}: {self.content}"


class StepForcingPrompt:
    """Build prompts that force structured step-by-step generation."""

    @staticmethod
    def build_initial_prompt(question: str, context: str = "") -> str:
        """Build prompt for generating the first step.

        Args:
            question: The question to solve
            context: Optional context (e.g., retrieved passages)

        Returns:
            Prompt that forces "Step 1:" generation
        """
        prompt = f"""Question: {question}

Please solve this problem step by step. Start with:

Step 1:"""

        if context:
            prompt = f"""Question: {question}

Context:
{context}

Please solve this problem step by step using the provided context. Start with:

Step 1:"""

        return prompt

    @staticmethod
    def build_continuation_prompt(
        question: str,
        previous_steps: List[ForcedStep],
        context: str = ""
    ) -> str:
        """Build prompt for continuing with next step.

        Args:
            question: The question being solved
            previous_steps: Steps generated so far
            context: Optional new context (for RAG)

        Returns:
            Prompt that forces "Step N:" generation
        """
        next_step_num = len(previous_steps) + 1

        # Format previous steps
        steps_text = "\n".join(str(step) for step in previous_steps)

        prompt = f"""Question: {question}

{steps_text}

Step {next_step_num}:"""

        if context:
            prompt = f"""Question: {question}

{steps_text}

Retrieved Information:
{context}

Step {next_step_num}:"""

        return prompt

    @staticmethod
    def build_final_answer_prompt(
        question: str,
        previous_steps: List[ForcedStep]
    ) -> str:
        """Build prompt for generating final answer.

        Args:
            question: The question being solved
            previous_steps: All reasoning steps so far

        Returns:
            Prompt that forces final answer
        """
        steps_text = "\n".join(str(step) for step in previous_steps)

        prompt = f"""Question: {question}

{steps_text}

Therefore, the final answer is:"""

        return prompt


class StepParser:
    """Parse responses generated with step forcing."""

    # Regex patterns for step detection
    STEP_PATTERN = re.compile(r'^Step\s+(\d+):\s*(.+?)(?=\n(?:Step\s+\d+:|$))',
                               re.MULTILINE | re.DOTALL)
    SINGLE_STEP_PATTERN = re.compile(r'^Step\s+(\d+):\s*(.+)$',
                                       re.MULTILINE | re.DOTALL)

    @staticmethod
    def parse_steps(text: str) -> List[ForcedStep]:
        """Parse text containing multiple steps.

        Args:
            text: Generated text with "Step N:" format

        Returns:
            List of parsed ForcedStep objects
        """
        steps = []

        # Try to find all steps
        matches = StepParser.STEP_PATTERN.findall(text)

        if not matches:
            # Try single step pattern
            single_match = StepParser.SINGLE_STEP_PATTERN.search(text)
            if single_match:
                step_num = int(single_match.group(1))
                content = single_match.group(2).strip()
                return [ForcedStep(step_number=step_num, content=content)]
            else:
                # No step markers found, treat as single step 1
                if text.strip():
                    return [ForcedStep(step_number=1, content=text.strip())]
                return []

        for step_num_str, content in matches:
            step_num = int(step_num_str)
            content = content.strip()
            steps.append(ForcedStep(step_number=step_num, content=content))

        return steps

    @staticmethod
    def parse_single_step_response(text: str, expected_step_num: int) -> Optional[str]:
        """Parse response for a single step generation.

        When we prompt with "Step N:", the model should continue with the content.
        This extracts just the content (without the "Step N:" prefix).

        Args:
            text: Generated text
            expected_step_num: The step number we prompted for

        Returns:
            Step content or None if parsing failed
        """
        text = text.strip()

        # If text starts with "Step N:", extract content after it
        pattern = re.compile(rf'^Step\s+{expected_step_num}:\s*(.+)', re.DOTALL)
        match = pattern.search(text)

        if match:
            return match.group(1).strip()

        # Otherwise, assume the entire text is the step content
        # (model continued directly from "Step N:" prompt)
        return text if text else None

    @staticmethod
    def extract_final_answer(text: str) -> Optional[str]:
        """Extract final answer from response.

        Args:
            text: Generated text

        Returns:
            Extracted answer or None
        """
        # Common answer patterns
        patterns = [
            r'(?:the\s+)?final\s+answer\s+is:?\s*(.+)',
            r'(?:therefore|thus|hence),?\s+(?:the\s+answer\s+is:?\s*)?(.+)',
            r'answer:\s*(.+)',
        ]

        text_lower = text.lower()

        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return last sentence
        sentences = text.strip().split('.')
        if sentences:
            return sentences[-1].strip()

        return text.strip()


class MultiPathSampler:
    """Sample multiple solution paths with correct/incorrect balancing."""

    def __init__(
        self,
        max_paths: int = 2048,
        min_correct_paths: int = 1,
        min_incorrect_paths: int = 1,
    ):
        """Initialize sampler.

        Args:
            max_paths: Maximum number of paths to sample per problem
            min_correct_paths: Minimum correct paths required
            min_incorrect_paths: Minimum incorrect paths required
        """
        self.max_paths = max_paths
        self.min_correct_paths = min_correct_paths
        self.min_incorrect_paths = min_incorrect_paths

    def should_continue_sampling(
        self,
        num_sampled: int,
        num_correct: int,
        num_incorrect: int,
    ) -> bool:
        """Check if we should continue sampling.

        Args:
            num_sampled: Total number of paths sampled
            num_correct: Number of correct paths found
            num_incorrect: Number of incorrect paths found

        Returns:
            True if should continue sampling
        """
        # Stop if reached max
        if num_sampled >= self.max_paths:
            return False

        # Continue if we don't have enough correct or incorrect paths
        if num_correct < self.min_correct_paths:
            return True
        if num_incorrect < self.min_incorrect_paths:
            return True

        # Stop if we have sufficient paths of both types
        return False

    def should_discard_problem(
        self,
        num_sampled: int,
        num_correct: int,
        num_incorrect: int,
    ) -> bool:
        """Check if problem should be discarded.

        A problem is discarded if we sampled max_paths but still don't have
        both correct and incorrect paths.

        Args:
            num_sampled: Total number of paths sampled
            num_correct: Number of correct paths found
            num_incorrect: Number of incorrect paths found

        Returns:
            True if problem should be discarded
        """
        if num_sampled < self.max_paths:
            return False

        # Discard if missing correct or incorrect paths
        if num_correct < self.min_correct_paths:
            return True
        if num_incorrect < self.min_incorrect_paths:
            return True

        return False


def format_trajectory_with_steps(steps: List[ForcedStep], final_answer: str) -> str:
    """Format a complete trajectory with forced steps.

    Args:
        steps: List of forced steps
        final_answer: Final answer

    Returns:
        Formatted trajectory string
    """
    lines = [str(step) for step in steps]
    lines.append(f"\nTherefore, the final answer is: {final_answer}")
    return "\n".join(lines)
