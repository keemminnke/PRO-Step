"""Configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class DataConfig:
    """Data configuration."""
    input_format: str = "jsonl"
    batch_size: int = 32
    num_workers: int = 4
    max_trajectory_length: int = 50


@dataclass
class RPEConfig:
    """RPE labeler configuration."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_path: Optional[str] = None
    device: str = "cuda"
    num_rollouts: int = 4
    max_rollout_steps: int = 20
    temperature: float = 0.8
    top_p: float = 0.95
    threshold: float = 0.5  # Binary threshold: >= 0.5 → GOOD, < 0.5 → BAD
    use_cache: bool = True
    batch_inference: bool = True


@dataclass
class JudgeConfig:
    """Judge labeler configuration."""
    model_name: str = "Qwen/QwQ-32B"
    model_path: Optional[str] = None
    device: str = "cuda"
    temperature: float = 0.3
    max_tokens: int = 512
    top_p: float = 0.9
    max_retries: int = 3
    retry_delay: float = 2.0
    prompt_style: str = "versaprm"
    include_reasoning: bool = True
    use_gold_answer: bool = True
    use_supporting_facts: bool = True


@dataclass
class ConsensusConfig:
    """Consensus configuration."""
    strategy: str = "strict"
    rules: Dict[str, Any] = field(default_factory=dict)
    min_agreement: float = 0.8
    trajectory_level: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: {"rpe": 0.3, "judge": 0.7})


@dataclass
class OutputConfig:
    """Output configuration."""
    save_individual_labels: bool = True
    save_consensus_only: bool = True
    save_metadata: bool = True
    format: str = "jsonl"
    compression: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "outputs/logs"
    use_wandb: bool = False
    wandb_project: str = "prmrag"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    rpe: RPEConfig = field(default_factory=RPEConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        PipelineConfig object
    """
    config_dict = load_config(config_path)

    return PipelineConfig(
        data=DataConfig(**config_dict.get("data", {})),
        rpe=RPEConfig(**config_dict.get("rpe", {})),
        judge=JudgeConfig(**config_dict.get("judge", {})),
        consensus=ConsensusConfig(**config_dict.get("consensus", {})),
        output=OutputConfig(**config_dict.get("output", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
        seed=config_dict.get("seed", 42),
    )
