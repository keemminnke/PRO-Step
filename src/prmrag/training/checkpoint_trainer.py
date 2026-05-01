"""Checkpoint-based trainer for scaling law experiments.

This module provides:
- Training with periodic checkpoint saving
- Resume from checkpoints
- Evaluation at each checkpoint
- Scaling law analysis support
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .data_sampler import DifficultySampler, SamplingConfig, TrainingSample, DifficultyLevel


class CheckpointStatus(str, Enum):
    """Status of a training checkpoint."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EVALUATED = "evaluated"


@dataclass
class TrainingCheckpoint:
    """Represents a training checkpoint."""

    name: str                          # e.g., "pilot_500"
    num_samples: int                   # Total samples at this checkpoint
    difficulty_distribution: Dict[str, int]  # {hard: N, medium: N, easy: N}

    # Status tracking
    status: CheckpointStatus = CheckpointStatus.PENDING

    # Training metadata
    trained_at: Optional[str] = None
    training_duration_sec: Optional[float] = None
    final_loss: Optional[float] = None

    # Evaluation results
    eval_results: Dict[str, Any] = field(default_factory=dict)

    # File paths
    data_path: Optional[str] = None
    model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingCheckpoint":
        """Create from dictionary."""
        data['status'] = CheckpointStatus(data['status'])
        return cls(**data)


@dataclass
class ScalingExperiment:
    """Configuration and state for a scaling law experiment."""

    name: str
    description: str = ""

    # Checkpoints in order
    checkpoints: List[TrainingCheckpoint] = field(default_factory=list)

    # Experiment config
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    base_learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_epochs: int = 3
    warmup_ratio: float = 0.1

    # Paths
    output_dir: str = "outputs/scaling_experiments"

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'model_name': self.model_name,
            'base_learning_rate': self.base_learning_rate,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_epochs': self.max_epochs,
            'warmup_ratio': self.warmup_ratio,
            'output_dir': self.output_dir,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalingExperiment":
        """Create from dictionary."""
        checkpoints = [TrainingCheckpoint.from_dict(c) for c in data.pop('checkpoints', [])]
        experiment = cls(**data)
        experiment.checkpoints = checkpoints
        return experiment

    def save(self, path: Optional[Path] = None):
        """Save experiment state to JSON."""
        if path is None:
            path = Path(self.output_dir) / self.name / "experiment.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ScalingExperiment":
        """Load experiment state from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class CheckpointTrainer:
    """Trainer that supports checkpoint-based scaling law experiments.

    This trainer:
    1. Creates datasets for each checkpoint (cumulative)
    2. Trains model at each checkpoint
    3. Evaluates and saves results
    4. Supports resuming from any checkpoint
    """

    def __init__(
        self,
        experiment: Optional[ScalingExperiment] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """Initialize trainer.

        Args:
            experiment: Scaling experiment configuration
            sampling_config: Configuration for data sampling
        """
        self.experiment = experiment or self._create_default_experiment()
        self.sampling_config = sampling_config or SamplingConfig()
        self.sampler = DifficultySampler(self.sampling_config)

        # Callbacks
        self._train_fn: Optional[Callable] = None
        self._eval_fn: Optional[Callable] = None

    def _create_default_experiment(self) -> ScalingExperiment:
        """Create default scaling experiment with standard checkpoints."""
        experiment = ScalingExperiment(
            name=f"scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Default scaling law experiment",
        )

        # Create checkpoints based on sampling config
        checkpoint_configs = SamplingConfig().checkpoint_configs

        for name, dist in checkpoint_configs.items():
            total = sum(dist.values())
            checkpoint = TrainingCheckpoint(
                name=name,
                num_samples=total,
                difficulty_distribution=dist,
            )
            experiment.checkpoints.append(checkpoint)

        return experiment

    def register_train_fn(self, fn: Callable[[Path, Path, Dict], Dict]):
        """Register training function.

        Args:
            fn: Function that takes (data_path, output_path, config) and returns metrics
        """
        self._train_fn = fn

    def register_eval_fn(self, fn: Callable[[Path, Path], Dict]):
        """Register evaluation function.

        Args:
            fn: Function that takes (model_path, eval_data_path) and returns metrics
        """
        self._eval_fn = fn

    def prepare_data(self, input_file: Path, limit: Optional[int] = None):
        """Load and classify input data.

        Args:
            input_file: Path to input trajectories JSONL
            limit: Optional limit on trajectories to load
        """
        counts = self.sampler.load_and_classify(input_file, limit)
        print(f"\nData loaded and classified:")
        for diff, count in counts.items():
            print(f"  {diff}: {count}")

        return counts

    def create_checkpoint_datasets(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Create datasets for all checkpoints.

        Args:
            output_dir: Directory to save datasets

        Returns:
            Dictionary mapping checkpoint names to file paths
        """
        if output_dir is None:
            output_dir = Path(self.experiment.output_dir) / self.experiment.name / "data"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use sampler to create datasets
        checkpoint_files = self.sampler.create_checkpoint_datasets(output_dir)

        # Update checkpoint metadata
        for checkpoint in self.experiment.checkpoints:
            if checkpoint.name in checkpoint_files:
                checkpoint.data_path = str(checkpoint_files[checkpoint.name])

        # Save experiment state
        self.experiment.save()

        return checkpoint_files

    def train_checkpoint(
        self,
        checkpoint_name: str,
        resume_from: Optional[str] = None,
    ) -> TrainingCheckpoint:
        """Train a single checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to train
            resume_from: Optional previous checkpoint to resume from

        Returns:
            Updated TrainingCheckpoint
        """
        # Find checkpoint
        checkpoint = None
        for c in self.experiment.checkpoints:
            if c.name == checkpoint_name:
                checkpoint = c
                break

        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint_name}")

        if checkpoint.data_path is None:
            raise ValueError(f"No data path for checkpoint {checkpoint_name}. Run prepare_data first.")

        # Update status
        checkpoint.status = CheckpointStatus.IN_PROGRESS
        self.experiment.save()

        print(f"\n{'='*60}")
        print(f"Training checkpoint: {checkpoint_name}")
        print(f"  Samples: {checkpoint.num_samples}")
        print(f"  Distribution: {checkpoint.difficulty_distribution}")
        print(f"  Data: {checkpoint.data_path}")
        print(f"{'='*60}")

        # Model output path
        model_dir = Path(self.experiment.output_dir) / self.experiment.name / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / checkpoint_name

        # Training config
        train_config = {
            'model_name': self.experiment.model_name,
            'learning_rate': self.experiment.base_learning_rate,
            'batch_size': self.experiment.batch_size,
            'gradient_accumulation_steps': self.experiment.gradient_accumulation_steps,
            'max_epochs': self.experiment.max_epochs,
            'warmup_ratio': self.experiment.warmup_ratio,
            'resume_from': resume_from,
        }

        # Run training if function registered
        if self._train_fn is not None:
            import time
            start_time = time.time()

            metrics = self._train_fn(
                Path(checkpoint.data_path),
                model_path,
                train_config,
            )

            duration = time.time() - start_time

            # Update checkpoint
            checkpoint.trained_at = datetime.now().isoformat()
            checkpoint.training_duration_sec = duration
            checkpoint.final_loss = metrics.get('final_loss')
            checkpoint.model_path = str(model_path)
        else:
            print("  [STUB] No training function registered - skipping actual training")
            checkpoint.model_path = str(model_path)

        # Update status
        checkpoint.status = CheckpointStatus.COMPLETED
        self.experiment.save()

        return checkpoint

    def evaluate_checkpoint(
        self,
        checkpoint_name: str,
        eval_data_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate a trained checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to evaluate
            eval_data_path: Path to evaluation data

        Returns:
            Evaluation metrics
        """
        # Find checkpoint
        checkpoint = None
        for c in self.experiment.checkpoints:
            if c.name == checkpoint_name:
                checkpoint = c
                break

        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint_name}")

        if checkpoint.model_path is None:
            raise ValueError(f"Checkpoint {checkpoint_name} not trained yet")

        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {checkpoint_name}")
        print(f"  Model: {checkpoint.model_path}")
        print(f"{'='*60}")

        # Run evaluation if function registered
        if self._eval_fn is not None and eval_data_path is not None:
            metrics = self._eval_fn(
                Path(checkpoint.model_path),
                eval_data_path,
            )
            checkpoint.eval_results = metrics
        else:
            print("  [STUB] No evaluation function registered - returning placeholder metrics")
            checkpoint.eval_results = {
                'accuracy': 0.0,
                'f1': 0.0,
                'evaluated_at': datetime.now().isoformat(),
                'note': 'placeholder - no eval function registered',
            }

        # Update status
        checkpoint.status = CheckpointStatus.EVALUATED
        self.experiment.save()

        return checkpoint.eval_results

    def run_full_experiment(
        self,
        input_file: Path,
        eval_data_path: Optional[Path] = None,
        start_from: Optional[str] = None,
    ) -> ScalingExperiment:
        """Run full scaling experiment.

        Args:
            input_file: Path to input trajectories
            eval_data_path: Path to evaluation data
            start_from: Optional checkpoint name to start from (for resuming)

        Returns:
            Completed ScalingExperiment
        """
        print(f"\n{'#'*60}")
        print(f"# Scaling Law Experiment: {self.experiment.name}")
        print(f"{'#'*60}")

        # 1. Prepare data
        print("\n[Step 1] Preparing data...")
        self.prepare_data(input_file)

        # 2. Create checkpoint datasets
        print("\n[Step 2] Creating checkpoint datasets...")
        self.create_checkpoint_datasets()

        # 3. Train each checkpoint
        print("\n[Step 3] Training checkpoints...")

        started = start_from is None
        prev_checkpoint = None

        for checkpoint in self.experiment.checkpoints:
            if not started:
                if checkpoint.name == start_from:
                    started = True
                else:
                    prev_checkpoint = checkpoint.name
                    continue

            # Train
            self.train_checkpoint(
                checkpoint.name,
                resume_from=prev_checkpoint,
            )

            # Evaluate
            if eval_data_path is not None:
                self.evaluate_checkpoint(checkpoint.name, eval_data_path)

            prev_checkpoint = checkpoint.name

        # 4. Print summary
        print(f"\n{'#'*60}")
        print(f"# Experiment Complete!")
        print(f"{'#'*60}")

        self._print_summary()

        return self.experiment

    def _print_summary(self):
        """Print experiment summary."""
        print("\n--- Checkpoint Summary ---")

        for checkpoint in self.experiment.checkpoints:
            status_symbol = {
                CheckpointStatus.PENDING: "⏳",
                CheckpointStatus.IN_PROGRESS: "🔄",
                CheckpointStatus.COMPLETED: "✅",
                CheckpointStatus.EVALUATED: "📊",
            }.get(checkpoint.status, "❓")

            print(f"\n{status_symbol} {checkpoint.name}")
            print(f"   Samples: {checkpoint.num_samples}")
            print(f"   Distribution: {checkpoint.difficulty_distribution}")

            if checkpoint.final_loss is not None:
                print(f"   Final Loss: {checkpoint.final_loss:.4f}")

            if checkpoint.eval_results:
                acc = checkpoint.eval_results.get('accuracy', 'N/A')
                f1 = checkpoint.eval_results.get('f1', 'N/A')
                print(f"   Eval - Accuracy: {acc}, F1: {f1}")

    def get_scaling_data(self) -> List[Dict[str, Any]]:
        """Get data for scaling curve plotting.

        Returns:
            List of dicts with (num_samples, metrics) for each checkpoint
        """
        data = []

        for checkpoint in self.experiment.checkpoints:
            if checkpoint.status in [CheckpointStatus.COMPLETED, CheckpointStatus.EVALUATED]:
                entry = {
                    'name': checkpoint.name,
                    'num_samples': checkpoint.num_samples,
                    'difficulty_distribution': checkpoint.difficulty_distribution,
                    'final_loss': checkpoint.final_loss,
                    **checkpoint.eval_results,
                }
                data.append(entry)

        return data
