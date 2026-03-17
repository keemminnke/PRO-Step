"""Base labeler interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..data.schemas import Trajectory


class BaseLabeler(ABC):
    """Abstract base class for all labelers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize labeler with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    @abstractmethod
    def label_trajectory(self, trajectory: Trajectory) -> List[Dict[str, Any]]:
        """Label a single trajectory.

        Args:
            trajectory: Trajectory to label

        Returns:
            List of labels, one per step
        """
        pass

    def label_batch(
        self, trajectories: List[Trajectory]
    ) -> List[List[Dict[str, Any]]]:
        """Label a batch of trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            List of label lists
        """
        return [self.label_trajectory(traj) for traj in trajectories]
