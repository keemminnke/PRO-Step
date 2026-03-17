"""Consensus module for combining RPE and Judge labels.

This is the core of the auto-labeling pipeline:
- Combines MC-based RPE labels with LLM Judge labels
- Filters out disagreements and uncertain cases
- Produces high-confidence 0/1 labels automatically
- The filtering process IS the labeling process
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..data.schemas import (
    Trajectory,
    RPELabel,
    JudgeLabel,
    ConsensusLabel,
    LabeledTrajectory,
    LabelType,
)


@dataclass
class ConsensusStats:
    """Statistics from consensus filtering."""
    total_steps: int
    positive_consensus: int  # Both agree: GOOD
    negative_consensus: int  # Both agree: BAD
    disagreements: int  # Conflicting labels
    filtered_steps: int  # Steps removed
    agreement_rate: float  # % of steps with consensus
    label_distribution: Dict[str, int]  # Distribution of final labels


class ConsensusModule:
    """Module for consensus-based filtering and labeling.

    This module implements the core logic:
    1. Takes RPE labels (GOOD/BORDERLINE/BAD) and Judge labels (GOOD/BAD)
    2. Applies consensus rules to determine final labels
    3. Filters out steps where labels conflict
    4. Optionally filters entire trajectories if any step conflicts

    The key insight: **filtering = labeling**
    Steps that pass consensus automatically have reliable 0/1 labels.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize consensus module.

        Args:
            config: Configuration dictionary with keys:
                - strategy: "strict", "lenient", or "balanced"
                - rules: Custom consensus rules
                - trajectory_level: Whether to filter entire trajectories
                - min_agreement: Minimum agreement threshold
        """
        self.config = config
        self.strategy = config.get("strategy", "strict")
        self.trajectory_level = config.get("trajectory_level", True)
        self.min_agreement = config.get("min_agreement", 0.8)

        # Load consensus rules
        self.rules = config.get("rules", self._get_default_rules())

    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default consensus rules based on strategy (binary labels only)."""
        # All strategies are the same now with binary labels
        # Only keep consensus when both agree (strict)
        return {
            "positive_consensus": {
                "rpe": ["GOOD"],
                "judge": ["GOOD"],
                "label": 1,
            },
            "negative_consensus": {
                "rpe": ["BAD"],
                "judge": ["BAD"],
                "label": 0,
            },
            "filter_disagreement": True,
        }

    def create_consensus_labels(
        self,
        trajectory: Trajectory,
        rpe_labels: List[RPELabel],
        judge_labels: List[JudgeLabel],
    ) -> LabeledTrajectory:
        """Create consensus labels for a trajectory.

        Args:
            trajectory: The trajectory
            rpe_labels: RPE labels for each step
            judge_labels: Judge labels for each step

        Returns:
            LabeledTrajectory with consensus labels and filtering decision
        """
        assert len(rpe_labels) == len(judge_labels) == len(trajectory.steps), \
            "Label lists must match trajectory length"

        consensus_labels = []
        has_conflict = False

        for rpe_label, judge_label in zip(rpe_labels, judge_labels):
            consensus = self._compute_step_consensus(rpe_label, judge_label)
            consensus_labels.append(consensus)

            if not consensus.is_consensus:
                has_conflict = True

        # Determine if trajectory should be filtered
        is_filtered = False
        filter_reason = None

        if self.trajectory_level and has_conflict:
            is_filtered = True
            filter_reason = "trajectory_level_conflict"

        # Compute agreement rate
        agreement_rate = sum(c.is_consensus for c in consensus_labels) / len(consensus_labels)

        if agreement_rate < self.min_agreement:
            is_filtered = True
            filter_reason = f"agreement_rate_too_low_{agreement_rate:.2f}"

        return LabeledTrajectory(
            trajectory=trajectory,
            rpe_labels=rpe_labels,
            judge_labels=judge_labels,
            consensus_labels=consensus_labels,
            is_filtered=is_filtered,
            filter_reason=filter_reason,
        )

    def _compute_step_consensus(
        self,
        rpe_label: RPELabel,
        judge_label: JudgeLabel,
    ) -> ConsensusLabel:
        """Compute consensus for a single step.

        Args:
            rpe_label: RPE label
            judge_label: Judge label

        Returns:
            ConsensusLabel indicating consensus result
        """
        rpe_type = rpe_label.label.value
        judge_type = judge_label.label

        # Check positive consensus
        if (
            rpe_type in self.rules["positive_consensus"]["rpe"]
            and judge_type in self.rules["positive_consensus"]["judge"]
        ):
            return ConsensusLabel(
                step_id=rpe_label.step_id,
                rpe_label=rpe_label.label,
                judge_label=judge_label.label,
                consensus_label=1,
                is_consensus=True,
                confidence=min(rpe_label.confidence, judge_label.confidence),
            )

        # Check negative consensus
        if (
            rpe_type in self.rules["negative_consensus"]["rpe"]
            and judge_type in self.rules["negative_consensus"]["judge"]
        ):
            return ConsensusLabel(
                step_id=rpe_label.step_id,
                rpe_label=rpe_label.label,
                judge_label=judge_label.label,
                consensus_label=0,
                is_consensus=True,
                confidence=min(rpe_label.confidence, judge_label.confidence),
            )

        # No consensus - conflict or uncertain
        conflict_type = f"rpe_{rpe_type}_judge_{judge_type}"

        return ConsensusLabel(
            step_id=rpe_label.step_id,
            rpe_label=rpe_label.label,
            judge_label=judge_label.label,
            consensus_label=None,  # Filtered
            is_consensus=False,
            confidence=0.0,
            filtered_reason=conflict_type,
        )

    def process_batch(
        self,
        trajectories: List[Trajectory],
        rpe_labels_batch: List[List[RPELabel]],
        judge_labels_batch: List[List[JudgeLabel]],
    ) -> List[LabeledTrajectory]:
        """Process a batch of trajectories.

        Args:
            trajectories: List of trajectories
            rpe_labels_batch: List of RPE label lists
            judge_labels_batch: List of Judge label lists

        Returns:
            List of LabeledTrajectory objects
        """
        labeled_trajectories = []

        for traj, rpe_labels, judge_labels in zip(
            trajectories, rpe_labels_batch, judge_labels_batch
        ):
            labeled_traj = self.create_consensus_labels(traj, rpe_labels, judge_labels)
            labeled_trajectories.append(labeled_traj)

        return labeled_trajectories

    def compute_statistics(
        self,
        labeled_trajectories: List[LabeledTrajectory],
    ) -> ConsensusStats:
        """Compute statistics over labeled trajectories.

        Args:
            labeled_trajectories: List of labeled trajectories

        Returns:
            ConsensusStats object
        """
        total_steps = 0
        positive_consensus = 0
        negative_consensus = 0
        disagreements = 0
        filtered_steps = 0

        label_counts = {0: 0, 1: 0, None: 0}

        for labeled_traj in labeled_trajectories:
            for consensus in labeled_traj.consensus_labels:
                total_steps += 1

                if consensus.consensus_label == 1:
                    positive_consensus += 1
                    label_counts[1] += 1
                elif consensus.consensus_label == 0:
                    negative_consensus += 1
                    label_counts[0] += 1
                else:
                    disagreements += 1
                    filtered_steps += 1
                    label_counts[None] += 1

        agreement_rate = (
            (positive_consensus + negative_consensus) / total_steps
            if total_steps > 0
            else 0.0
        )

        return ConsensusStats(
            total_steps=total_steps,
            positive_consensus=positive_consensus,
            negative_consensus=negative_consensus,
            disagreements=disagreements,
            filtered_steps=filtered_steps,
            agreement_rate=agreement_rate,
            label_distribution={
                "positive (1)": label_counts[1],
                "negative (0)": label_counts[0],
                "filtered (None)": label_counts[None],
            },
        )

    def get_high_quality_samples(
        self,
        labeled_trajectories: List[LabeledTrajectory],
    ) -> List[Dict[str, Any]]:
        """Extract high-quality training samples from labeled trajectories.

        Only includes samples that passed consensus filtering.

        Args:
            labeled_trajectories: List of labeled trajectories

        Returns:
            List of training samples
        """
        all_samples = []

        for labeled_traj in labeled_trajectories:
            if not labeled_traj.is_filtered:
                samples = labeled_traj.get_training_samples()
                all_samples.extend(samples)

        return all_samples

    def filter_by_confidence(
        self,
        labeled_trajectories: List[LabeledTrajectory],
        min_confidence: float = 0.7,
    ) -> List[LabeledTrajectory]:
        """Further filter by confidence threshold.

        Args:
            labeled_trajectories: Input labeled trajectories
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list
        """
        filtered = []

        for labeled_traj in labeled_trajectories:
            if labeled_traj.is_filtered:
                continue

            # Check if all consensus labels meet confidence threshold
            confidences = [c.confidence for c in labeled_traj.consensus_labels]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            if avg_confidence >= min_confidence:
                filtered.append(labeled_traj)

        return filtered


def print_consensus_report(stats: ConsensusStats) -> None:
    """Print a formatted consensus statistics report.

    Args:
        stats: ConsensusStats object
    """
    print("\n" + "=" * 60)
    print("CONSENSUS FILTERING REPORT")
    print("=" * 60)
    print(f"\nTotal steps evaluated: {stats.total_steps}")
    print(f"\nConsensus Results:")
    print(f"  ✓ Positive consensus (label=1): {stats.positive_consensus}")
    print(f"  ✓ Negative consensus (label=0): {stats.negative_consensus}")
    print(f"  ✗ Disagreements (filtered):     {stats.disagreements}")
    print(f"\nAgreement rate: {stats.agreement_rate:.1%}")
    print(f"Filtered rate:  {stats.filtered_steps / stats.total_steps:.1%}")
    print(f"\nFinal label distribution:")
    for label, count in stats.label_distribution.items():
        pct = count / stats.total_steps * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    print("=" * 60 + "\n")
