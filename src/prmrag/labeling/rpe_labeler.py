"""MC-based RPE labeler using vLLM."""

from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np

from .base_labeler import BaseLabeler
from ..data.schemas import Trajectory, RPELabel, LabelType
from ..utils.answer_utils import (
    extract_answer_from_text,
    normalize_answer,
    check_answer_match,
)


class RPELabeler(BaseLabeler):
    """RPE (Relative Policy Evaluation) labeler using Monte Carlo estimation.

    This labeler:
    1. For each step t, fixes the prefix up to t
    2. Runs K rollouts from that prefix (MC estimation)
    3. Computes MC(s_t) and MC(s_t, a_t)
    4. Calculates RPE = MC(s_t, a_t) / MC(s_t)
    5. Thresholds RPE to assign GOOD/BORDERLINE/BAD labels

    Now uses vLLM backend for fast inference.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model=None,
    ):
        """Initialize RPE labeler.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Model name for vLLM
                - num_rollouts: Number of MC rollouts
                - max_rollout_steps: Max steps per rollout
                - threshold: RPE threshold (>= threshold: GOOD, < threshold: BAD)
                - temperature: Sampling temperature
                - gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.7)
                - tensor_parallel_size: Number of GPUs for vLLM (default: 1)
            model: Pre-loaded vLLM PolicyModel (optional)
        """
        super().__init__(config)

        self.model_name = config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self.num_rollouts = config.get("num_rollouts", 5)
        self.max_rollout_steps = config.get("max_rollout_steps", 20)
        self.threshold = config.get("threshold", 0.5)
        self.temperature = config.get("temperature", 0.8)
        self.max_tokens = config.get("max_tokens", 512)

        # vLLM-specific settings
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.8)
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)

        self.model = model

        # Initialize vLLM model if not provided
        if self.model is None:
            self._initialize_vllm_model()

    def _initialize_vllm_model(self):
        """Initialize vLLM model for RPE labeling."""
        from ..models import load_policy_model

        model_config = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'tensor_parallel_size': self.tensor_parallel_size,
        }

        print(f"Initializing RPE model with vLLM: {self.model_name}")
        self.model = load_policy_model(model_config)

    def label_trajectory(self, trajectory: Trajectory) -> List[RPELabel]:
        """Label a trajectory using MC-based RPE.

        Args:
            trajectory: Trajectory to label

        Returns:
            List of RPELabel objects, one per step
        """
        labels = []

        for step_idx in range(len(trajectory.steps)):
            # Compute RPE for this step
            rpe_label = self._compute_step_rpe(trajectory, step_idx)
            labels.append(rpe_label)

        return labels

    def _compute_step_rpe(
        self,
        trajectory: Trajectory,
        step_idx: int,
    ) -> RPELabel:
        """Compute RPE for a specific step.

        Args:
            trajectory: Trajectory containing the step
            step_idx: Index of the step to evaluate

        Returns:
            RPELabel for this step
        """
        # Get prefix (steps before this one)
        prefix = trajectory.get_prefix(step_idx - 1) if step_idx > 0 else []

        # MC(s_t): Success rate starting from prefix (without this step)
        mc_s_t = self._monte_carlo_estimate(
            question=trajectory.question,
            prefix_steps=prefix,
            gold_answer=trajectory.gold_answer,
        )

        # MC(s_t, a_t): Success rate including this step
        prefix_with_step = trajectory.get_prefix(step_idx)
        mc_s_t_a_t = self._monte_carlo_estimate(
            question=trajectory.question,
            prefix_steps=prefix_with_step,
            gold_answer=trajectory.gold_answer,
        )

        # Compute RPE
        # Add small epsilon to avoid division by zero
        rpe = mc_s_t_a_t / (mc_s_t + 1e-8)

        # Assign binary label based on threshold
        label_type = LabelType.GOOD if rpe >= self.threshold else LabelType.BAD

        return RPELabel(
            step_id=step_idx,
            mc_s_t=mc_s_t,
            mc_s_t_a_t=mc_s_t_a_t,
            rpe=rpe,
            label=label_type,
            confidence=1.0,  # Can be refined based on variance
            metadata={
                "num_rollouts": self.num_rollouts,
                "threshold": self.threshold,
            },
        )

    def _monte_carlo_estimate(
        self,
        question: str,
        prefix_steps: List,
        gold_answer: Optional[str] = None,
    ) -> float:
        """Estimate success probability via Monte Carlo rollouts.

        Args:
            question: The question
            prefix_steps: Prefix trajectory steps
            gold_answer: Ground truth answer for evaluation

        Returns:
            Estimated success probability (0-1)
        """
        if self.model is None:
            # Fallback: random estimate (for testing)
            return np.random.uniform(0.3, 0.9)

        successes = 0

        for _ in range(self.num_rollouts):
            # Run rollout from this prefix
            final_answer = self._rollout(question, prefix_steps)

            # Check if answer is correct
            if gold_answer and self._check_answer(final_answer, gold_answer):
                successes += 1

        return successes / self.num_rollouts

    def _rollout(
        self,
        question: str,
        prefix_steps: List,
    ) -> str:
        """Perform a single rollout from a prefix using vLLM.

        Args:
            question: The question
            prefix_steps: Starting trajectory prefix

        Returns:
            Final answer from the rollout
        """
        if self.model is None:
            return "placeholder_answer"

        # Format prefix as prompt
        prompt = self._format_prompt(question, prefix_steps)

        # Generate with vLLM
        # Keep rollout short using step-based token budget
        max_tokens = min(self.max_tokens, int(self.max_rollout_steps * 64))
        response = self.model.generate_with_chat_template(
            user_message=prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )

        # Extract answer from response
        return self._extract_answer(response)

    def _format_prompt(
        self,
        question: str,
        prefix_steps: List,
    ) -> str:
        """Format question and prefix as prompt.

        Args:
            question: The question
            prefix_steps: Prefix trajectory steps

        Returns:
            Formatted prompt string
        """
        lines = [f"Question: {question}\n"]

        for i, step in enumerate(prefix_steps):
            lines.append(f"Step {i + 1}: {step.action}")
            # Include retrieval evidence to stabilize MC estimates
            if step.passages:
                lines.append("Passages:")
                for j, passage in enumerate(step.passages[:5]):
                    lines.append(f"  [{j + 1}] {passage}")
            lines.append(f"Result: {step.observation}\n")

        lines.append(
            f"Continue reasoning to answer the question in at most {self.max_rollout_steps} short steps."
        )
        lines.append("At the end, provide your final answer in this format:")
        lines.append('"Therefore, the answer is [your answer]."')

        return "\n".join(lines)

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from model response.

        Args:
            response: Model response text

        Returns:
            Extracted answer
        """
        return extract_answer_from_text(response)

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold answer.

        Args:
            predicted: Predicted answer
            gold: Gold answer

        Returns:
            True if match (>=50% token overlap)
        """
        pred_extracted = extract_answer_from_text(predicted)
        gold_extracted = extract_answer_from_text(gold)

        pred_norm = normalize_answer(pred_extracted)
        gold_norm = normalize_answer(gold_extracted)

        # Check match using cover exact match
        return check_answer_match(pred_norm, gold_norm)

    def label_batch(
        self,
        trajectories: List[Trajectory],
        show_progress: bool = True,
    ) -> List[List[RPELabel]]:
        """Label a batch of trajectories with progress bar.

        Args:
            trajectories: List of trajectories
            show_progress: Whether to show progress bar

        Returns:
            List of label lists
        """
        labels = []

        iterator = tqdm(trajectories, desc="RPE labeling") if show_progress else trajectories

        for traj in iterator:
            traj_labels = self.label_trajectory(traj)
            labels.append(traj_labels)

        return labels


def load_rpe_model(config: Dict[str, Any]):
    """Load vLLM model for RPE labeling.

    Args:
        config: Configuration with model_name and vLLM settings

    Returns:
        vLLM PolicyModel instance
    """
    from ..models import load_policy_model

    return load_policy_model(config)


def create_rpe_labeler(config: Dict[str, Any]) -> RPELabeler:
    """Create RPELabeler with vLLM backend.

    Args:
        config: Configuration with model settings

    Returns:
        RPELabeler instance with vLLM model
    """
    return RPELabeler(config)
