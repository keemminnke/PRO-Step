"""
PRMRAG: Consensus-Based Auto-Labeling Pipeline for RAG-CoT

A framework for automatically labeling RAG-CoT trajectories by combining
MC-based RPE signals (small model) with LLM judge signals (large model)
through consensus-based filtering.
"""

__version__ = "0.1.0"

from . import config, data, labeling, models, utils, regeneration

__all__ = ["config", "data", "labeling", "models", "utils", "regeneration"]
