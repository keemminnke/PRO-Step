"""Data sampling by difficulty for scaling experiments.

This module provides:
- Difficulty classification based on MC/RPE signals
- Stratified sampling for balanced datasets
- Incremental dataset construction for checkpoints
"""

import json
import random
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class DifficultyLevel(str, Enum):
    """Difficulty levels for training samples."""
    EASY = "easy"      # High MC, high consensus
    MEDIUM = "medium"  # Medium MC, some uncertainty
    HARD = "hard"      # Low MC, high disagreement


@dataclass
class SamplingConfig:
    """Configuration for difficulty-based sampling."""

    # Difficulty thresholds (based on trajectory-level MC)
    easy_mc_threshold: float = 0.7      # MC >= 0.7 → Easy
    medium_mc_threshold: float = 0.3    # 0.3 <= MC < 0.7 → Medium
    # MC < 0.3 → Hard

    # Additional criteria
    use_consensus_rate: bool = True     # Also consider RPE-Judge consensus
    min_consensus_for_easy: float = 0.9 # Easy requires 90%+ consensus

    # Sampling ratios for each checkpoint
    checkpoint_configs: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "pilot_500": {"hard": 200, "medium": 200, "easy": 100},
        "main_1000": {"hard": 400, "medium": 400, "easy": 200},  # Cumulative
        "full_2000": {"hard": 800, "medium": 800, "easy": 400},  # Cumulative
    })

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class TrainingSample:
    """A single training sample with metadata."""

    trajectory_id: str
    question: str
    gold_answer: str
    steps: List[Dict[str, Any]]
    difficulty: DifficultyLevel

    # Quality metrics
    avg_mc: float = 0.0
    avg_rpe: float = 0.0
    consensus_rate: float = 0.0
    is_correct: bool = False

    # Labels (after consensus)
    step_labels: List[Optional[int]] = field(default_factory=list)  # 0/1/None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trajectory_id": self.trajectory_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "steps": self.steps,
            "difficulty": self.difficulty.value,
            "avg_mc": self.avg_mc,
            "avg_rpe": self.avg_rpe,
            "consensus_rate": self.consensus_rate,
            "is_correct": self.is_correct,
            "step_labels": self.step_labels,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSample":
        """Create from dictionary."""
        return cls(
            trajectory_id=data["trajectory_id"],
            question=data["question"],
            gold_answer=data["gold_answer"],
            steps=data["steps"],
            difficulty=DifficultyLevel(data["difficulty"]),
            avg_mc=data.get("avg_mc", 0.0),
            avg_rpe=data.get("avg_rpe", 0.0),
            consensus_rate=data.get("consensus_rate", 0.0),
            is_correct=data.get("is_correct", False),
            step_labels=data.get("step_labels", []),
        )


class DifficultySampler:
    """Samples training data stratified by difficulty level.

    This class:
    1. Classifies trajectories by difficulty (Hard/Med/Easy)
    2. Samples according to specified ratios
    3. Supports incremental/cumulative checkpoint construction
    """

    def __init__(self, config: Optional[SamplingConfig] = None):
        """Initialize sampler.

        Args:
            config: Sampling configuration
        """
        self.config = config or SamplingConfig()
        random.seed(self.config.seed)

        # Storage for classified samples
        self.samples_by_difficulty: Dict[DifficultyLevel, List[TrainingSample]] = {
            DifficultyLevel.EASY: [],
            DifficultyLevel.MEDIUM: [],
            DifficultyLevel.HARD: [],
        }

        # Track what's been sampled for each checkpoint
        self.sampled_ids: Dict[str, set] = defaultdict(set)

    def classify_difficulty(self, trajectory: Dict[str, Any]) -> DifficultyLevel:
        """Classify a trajectory's difficulty based on MC and consensus.

        Args:
            trajectory: Trajectory data with steps containing MC/RPE

        Returns:
            DifficultyLevel
        """
        steps = trajectory.get("steps", [])
        if not steps:
            return DifficultyLevel.HARD

        # Calculate average MC (use mc_after as the primary signal)
        mc_values = []
        for step in steps:
            mc_after = step.get("mc_after", 0)
            if mc_after > 0:  # Ignore MC=0 steps
                mc_values.append(mc_after)

        avg_mc = sum(mc_values) / len(mc_values) if mc_values else 0.0

        # Calculate consensus rate if available
        consensus_rate = 1.0  # Default: assume consensus
        if self.config.use_consensus_rate:
            consensus_labels = trajectory.get("consensus_labels", [])
            if consensus_labels:
                valid_labels = [l for l in consensus_labels if l is not None]
                consensus_rate = len(valid_labels) / len(consensus_labels)

        # Classification logic
        if avg_mc >= self.config.easy_mc_threshold:
            if consensus_rate >= self.config.min_consensus_for_easy:
                return DifficultyLevel.EASY
            else:
                return DifficultyLevel.MEDIUM
        elif avg_mc >= self.config.medium_mc_threshold:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD

    def load_and_classify(
        self,
        input_file: Path,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """Load trajectories and classify by difficulty.

        Args:
            input_file: Path to JSONL file with trajectories
            limit: Optional limit on number of trajectories

        Returns:
            Dictionary with counts per difficulty level
        """
        print(f"Loading trajectories from {input_file}...")

        count = 0
        with open(input_file, 'r') as f:
            for line in f:
                if limit and count >= limit:
                    break

                trajectory = json.loads(line)

                # Classify difficulty
                difficulty = self.classify_difficulty(trajectory)

                # Calculate metrics
                steps = trajectory.get("steps", [])
                mc_values = [s.get("mc_after", 0) for s in steps]
                rpe_values = [s.get("rpe", 0) for s in steps]

                avg_mc = sum(mc_values) / len(mc_values) if mc_values else 0
                avg_rpe = sum(rpe_values) / len(rpe_values) if rpe_values else 0

                # Get step labels (from RPE for now)
                step_labels = []
                for step in steps:
                    label = step.get("label", "").lower()
                    if label == "good":
                        step_labels.append(1)
                    elif label == "bad":
                        step_labels.append(0)
                    else:
                        step_labels.append(None)

                # Create sample
                sample = TrainingSample(
                    trajectory_id=trajectory.get("question_id", str(count)),
                    question=trajectory.get("question", ""),
                    gold_answer=trajectory.get("gold_answer", ""),
                    steps=steps,
                    difficulty=difficulty,
                    avg_mc=avg_mc,
                    avg_rpe=avg_rpe,
                    is_correct=trajectory.get("is_correct", False),
                    step_labels=step_labels,
                )

                self.samples_by_difficulty[difficulty].append(sample)
                count += 1

        # Print summary
        counts = {d.value: len(samples) for d, samples in self.samples_by_difficulty.items()}
        print(f"Loaded {count} trajectories:")
        for diff, cnt in counts.items():
            print(f"  {diff}: {cnt}")

        return counts

    def sample_for_checkpoint(
        self,
        checkpoint_name: str,
        cumulative: bool = True,
    ) -> List[TrainingSample]:
        """Sample data for a specific checkpoint.

        Args:
            checkpoint_name: Name of checkpoint (e.g., "pilot_500")
            cumulative: If True, include samples from previous checkpoints

        Returns:
            List of TrainingSample objects
        """
        if checkpoint_name not in self.config.checkpoint_configs:
            raise ValueError(f"Unknown checkpoint: {checkpoint_name}")

        target_counts = self.config.checkpoint_configs[checkpoint_name]
        samples = []

        for difficulty in DifficultyLevel:
            diff_key = difficulty.value
            target = target_counts.get(diff_key, 0)
            available = self.samples_by_difficulty[difficulty]

            if cumulative:
                # Exclude already sampled for this checkpoint
                available = [s for s in available
                            if s.trajectory_id not in self.sampled_ids[checkpoint_name]]

            # Sample
            n_sample = min(target, len(available))
            sampled = random.sample(available, n_sample)

            # Track sampled IDs
            for s in sampled:
                self.sampled_ids[checkpoint_name].add(s.trajectory_id)

            samples.extend(sampled)

            print(f"  {diff_key}: sampled {n_sample}/{target} (available: {len(available)})")

        # Shuffle
        random.shuffle(samples)

        return samples

    def create_checkpoint_datasets(
        self,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Create datasets for all checkpoints.

        Args:
            output_dir: Directory to save datasets

        Returns:
            Dictionary mapping checkpoint names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_files = {}
        cumulative_samples = []

        # Process checkpoints in order
        checkpoint_order = ["pilot_500", "main_1000", "full_2000"]

        for checkpoint_name in checkpoint_order:
            if checkpoint_name not in self.config.checkpoint_configs:
                continue

            print(f"\n--- Creating {checkpoint_name} dataset ---")

            # Sample new data for this checkpoint
            new_samples = self.sample_for_checkpoint(checkpoint_name, cumulative=True)
            cumulative_samples.extend(new_samples)

            # Save
            output_file = output_dir / f"{checkpoint_name}.jsonl"
            with open(output_file, 'w') as f:
                for sample in cumulative_samples:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')

            checkpoint_files[checkpoint_name] = output_file

            # Print stats
            diff_counts = defaultdict(int)
            for s in cumulative_samples:
                diff_counts[s.difficulty.value] += 1

            print(f"  Total samples: {len(cumulative_samples)}")
            print(f"  Distribution: {dict(diff_counts)}")
            print(f"  Saved to: {output_file}")

        return checkpoint_files

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": sum(len(s) for s in self.samples_by_difficulty.values()),
            "by_difficulty": {},
        }

        for difficulty, samples in self.samples_by_difficulty.items():
            if not samples:
                continue

            avg_mc = sum(s.avg_mc for s in samples) / len(samples)
            avg_rpe = sum(s.avg_rpe for s in samples) / len(samples)
            correct_rate = sum(s.is_correct for s in samples) / len(samples)

            stats["by_difficulty"][difficulty.value] = {
                "count": len(samples),
                "avg_mc": round(avg_mc, 3),
                "avg_rpe": round(avg_rpe, 3),
                "correct_rate": round(correct_rate, 3),
            }

        return stats


def create_scaling_datasets(
    input_file: Path,
    output_dir: Path,
    config: Optional[SamplingConfig] = None,
) -> Dict[str, Path]:
    """Convenience function to create all checkpoint datasets.

    Args:
        input_file: Path to input trajectories JSONL
        output_dir: Directory to save datasets
        config: Optional sampling configuration

    Returns:
        Dictionary mapping checkpoint names to file paths
    """
    sampler = DifficultySampler(config)
    sampler.load_and_classify(input_file)

    print("\nData Statistics:")
    stats = sampler.get_statistics()
    for diff, info in stats["by_difficulty"].items():
        print(f"  {diff}: {info}")

    return sampler.create_checkpoint_datasets(output_dir)
