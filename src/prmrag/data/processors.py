"""Data preprocessing and processing utilities."""

from typing import List, Dict, Any
from .schemas import Trajectory, TrajectoryStep


class TrajectoryProcessor:
    """Processor for RAG-CoT trajectories."""

    def __init__(self, max_length: int = 50):
        """Initialize processor.

        Args:
            max_length: Maximum trajectory length
        """
        self.max_length = max_length

    def process(self, trajectory: Trajectory) -> Trajectory:
        """Process a single trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            Processed trajectory
        """
        # Truncate if too long
        if len(trajectory.steps) > self.max_length:
            trajectory.steps = trajectory.steps[:self.max_length]

        # Re-index steps
        for i, step in enumerate(trajectory.steps):
            step.step_id = i

        return trajectory

    def batch_process(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Process a batch of trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            List of processed trajectories
        """
        return [self.process(traj) for traj in trajectories]

    @staticmethod
    def format_trajectory_for_prompt(
        trajectory: Trajectory,
        up_to_step: int = None,
    ) -> str:
        """Format trajectory as text for prompting.

        Args:
            trajectory: Trajectory to format
            up_to_step: Only include steps up to this index (inclusive)

        Returns:
            Formatted string
        """
        steps = trajectory.steps if up_to_step is None else trajectory.steps[:up_to_step + 1]

        lines = [f"Question: {trajectory.question}\n"]

        for i, step in enumerate(steps):
            lines.append(f"\nStep {i + 1}:")
            lines.append(f"Action: {step.action}")
            if step.passages:
                lines.append(f"Retrieved passages: {len(step.passages)}")
                for j, passage in enumerate(step.passages[:3]):  # Show first 3
                    lines.append(f"  [{j+1}] {passage[:100]}...")
            lines.append(f"Observation: {step.observation}")

        return "\n".join(lines)

    @staticmethod
    def format_step_for_prompt(
        trajectory: Trajectory,
        step_idx: int,
        include_prefix: bool = True,
    ) -> str:
        """Format a specific step with context for prompting.

        Args:
            trajectory: Trajectory containing the step
            step_idx: Index of the step
            include_prefix: Whether to include previous steps

        Returns:
            Formatted string
        """
        lines = [f"Question: {trajectory.question}\n"]

        if include_prefix and step_idx > 0:
            lines.append("Previous steps:")
            for i in range(step_idx):
                step = trajectory.steps[i]
                lines.append(f"  {i + 1}. {step.action} → {step.observation[:50]}...")
            lines.append("")

        step = trajectory.steps[step_idx]
        lines.append(f"Current step {step_idx + 1}:")
        lines.append(f"Action: {step.action}")

        if step.passages:
            lines.append(f"Retrieved passages:")
            for j, passage in enumerate(step.passages):
                lines.append(f"  [{j+1}] {passage}")

        lines.append(f"Observation: {step.observation}")

        return "\n".join(lines)
