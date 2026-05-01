"""Data schemas for RAG-CoT trajectories and labels."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class ActionType(str, Enum):
    """Type of action in a trajectory step."""
    RETRIEVE = "retrieve"
    REASON = "reason"
    ANSWER = "answer"


class LabelType(str, Enum):
    """Label types for RPE labeling (binary)."""
    GOOD = "GOOD"
    BAD = "BAD"


@dataclass
class TrajectoryStep:
    """A single step in a RAG-CoT trajectory.

    Attributes:
        step_id: Unique identifier for this step
        action_type: Type of action (retrieve, reason, answer)
        action: The actual action text/query
        observation: Result/response from the action
        passages: Retrieved passages (if retrieve action)
        metadata: Additional metadata
    """
    step_id: int
    action_type: ActionType
    action: str
    observation: str
    passages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "action_type": self.action_type.value,
            "action": self.action,
            "observation": self.observation,
            "passages": self.passages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            action_type=ActionType(data["action_type"]),
            action=data["action"],
            observation=data["observation"],
            passages=data.get("passages", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Trajectory:
    """A complete RAG-CoT trajectory for a question.

    Attributes:
        trajectory_id: Unique identifier
        question: The question being answered
        gold_answer: Ground truth answer (optional)
        supporting_facts: Gold supporting facts (optional)
        steps: List of trajectory steps
        final_answer: The final answer produced
        is_correct: Whether final answer matches gold (optional)
        metadata: Additional metadata
    """
    trajectory_id: str
    question: str
    steps: List[TrajectoryStep]
    final_answer: str
    gold_answer: Optional[str] = None
    supporting_facts: Optional[List[str]] = None
    is_correct: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "supporting_facts": self.supporting_facts,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create from dictionary."""
        return cls(
            trajectory_id=data["trajectory_id"],
            question=data["question"],
            gold_answer=data.get("gold_answer"),
            supporting_facts=data.get("supporting_facts"),
            steps=[TrajectoryStep.from_dict(s) for s in data["steps"]],
            final_answer=data["final_answer"],
            is_correct=data.get("is_correct"),
            metadata=data.get("metadata", {}),
        )

    def get_prefix(self, up_to_step: int) -> List[TrajectoryStep]:
        """Get trajectory prefix up to (and including) a specific step."""
        return self.steps[:up_to_step + 1]


@dataclass
class RPELabel:
    """RPE-based label from small model.

    Attributes:
        step_id: Which step this labels
        mc_s_t: MC(s_t) - value before taking action
        mc_s_t_a_t: MC(s_t, a_t) - value after taking action
        rpe: RPE score (mc_s_t_a_t / mc_s_t)
        label: Categorical label (GOOD, BORDERLINE, BAD)
        confidence: Confidence score
        metadata: Additional info (num_rollouts, etc.)
    """
    step_id: int
    mc_s_t: float
    mc_s_t_a_t: float
    rpe: float
    label: LabelType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "mc_s_t": self.mc_s_t,
            "mc_s_t_a_t": self.mc_s_t_a_t,
            "rpe": self.rpe,
            "label": self.label.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class JudgeLabel:
    """Judge-based label from large LLM.

    Attributes:
        step_id: Which step this labels
        label: Binary label (GOOD or BAD)
        reasoning: Explanation from judge
        confidence: Judge's confidence score
        metadata: Additional info
    """
    step_id: int
    label: Literal["GOOD", "BAD"]
    reasoning: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "label": self.label,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class ConsensusLabel:
    """Consensus label combining RPE and Judge signals.

    Attributes:
        step_id: Which step this labels
        rpe_label: Original RPE label
        judge_label: Original Judge label
        consensus_label: Final consensus (0 or 1), None if filtered
        is_consensus: Whether both sources agreed
        confidence: Combined confidence score
        filtered_reason: Why it was filtered (if applicable)
    """
    step_id: int
    rpe_label: LabelType
    judge_label: Literal["GOOD", "BAD"]
    consensus_label: Optional[int]  # 0 or 1, None if filtered
    is_consensus: bool
    confidence: float
    filtered_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "rpe_label": self.rpe_label.value,
            "judge_label": self.judge_label,
            "consensus_label": self.consensus_label,
            "is_consensus": self.is_consensus,
            "confidence": self.confidence,
            "filtered_reason": self.filtered_reason,
        }


@dataclass
class LabeledTrajectory:
    """A trajectory with all labels attached.

    Attributes:
        trajectory: The original trajectory
        rpe_labels: RPE labels for each step
        judge_labels: Judge labels for each step
        consensus_labels: Consensus labels for each step
        is_filtered: Whether this trajectory should be filtered out
        filter_reason: Why it was filtered
    """
    trajectory: Trajectory
    rpe_labels: List[RPELabel] = field(default_factory=list)
    judge_labels: List[JudgeLabel] = field(default_factory=list)
    consensus_labels: List[ConsensusLabel] = field(default_factory=list)
    is_filtered: bool = False
    filter_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trajectory": self.trajectory.to_dict(),
            "rpe_labels": [label.to_dict() for label in self.rpe_labels],
            "judge_labels": [label.to_dict() for label in self.judge_labels],
            "consensus_labels": [label.to_dict() for label in self.consensus_labels],
            "is_filtered": self.is_filtered,
            "filter_reason": self.filter_reason,
        }

    def get_training_samples(self) -> List[Dict[str, Any]]:
        """Extract training samples (step, label pairs) for non-filtered steps.

        Returns:
            List of dicts with keys: step_id, question, prefix, action,
            observation, passages, label
        """
        samples = []
        for i, consensus in enumerate(self.consensus_labels):
            if consensus.consensus_label is not None:  # Not filtered
                step = self.trajectory.steps[i]
                prefix_steps = self.trajectory.get_prefix(i - 1) if i > 0 else []

                samples.append({
                    "trajectory_id": self.trajectory.trajectory_id,
                    "step_id": step.step_id,
                    "question": self.trajectory.question,
                    "prefix": [s.to_dict() for s in prefix_steps],
                    "action": step.action,
                    "observation": step.observation,
                    "passages": step.passages,
                    "label": consensus.consensus_label,
                    "confidence": consensus.confidence,
                })

        return samples
