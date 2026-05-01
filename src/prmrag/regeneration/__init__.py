"""Critic-guided trajectory regeneration module.

Pipeline:
1. Classify questions into Case A (has correct) vs Case B (all wrong)
2. Case A: rejection sampling - pick best correct trajectory by critic_min
3. Case B: truncate at first BAD step → add critic feedback → regenerate
"""

from .classifier import QuestionClassifier, ClassifiedQuestion, ScoredTrajectory
from .truncator import TrajectoryTruncator, TruncationResult
from .regenerator import CriticGuidedRegenerator

__all__ = [
    'QuestionClassifier',
    'ClassifiedQuestion',
    'ScoredTrajectory',
    'TrajectoryTruncator',
    'TruncationResult',
    'CriticGuidedRegenerator',
]
