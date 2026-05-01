"""Data loading and saving utilities."""

import json
import jsonlines
from pathlib import Path
from typing import List, Union, Iterator, Dict, Any
from .schemas import Trajectory, LabeledTrajectory


def load_trajectories(
    file_path: Union[str, Path],
    format: str = "jsonl",
    limit: int = None,
) -> List[Trajectory]:
    """Load trajectories from file.

    Args:
        file_path: Path to input file
        format: File format (jsonl, json)
        limit: Maximum number of trajectories to load

    Returns:
        List of Trajectory objects
    """
    file_path = Path(file_path)
    trajectories = []

    if format == "jsonl":
        with jsonlines.open(file_path) as reader:
            for i, obj in enumerate(reader):
                if limit and i >= limit:
                    break
                trajectories.append(Trajectory.from_dict(obj))

    elif format == "json":
        with open(file_path) as f:
            data = json.load(f)
            for i, obj in enumerate(data):
                if limit and i >= limit:
                    break
                trajectories.append(Trajectory.from_dict(obj))
    else:
        raise ValueError(f"Unsupported format: {format}")

    return trajectories


def save_trajectories(
    trajectories: List[Trajectory],
    file_path: Union[str, Path],
    format: str = "jsonl",
) -> None:
    """Save trajectories to file.

    Args:
        trajectories: List of Trajectory objects
        file_path: Path to output file
        format: File format (jsonl, json)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with jsonlines.open(file_path, mode='w') as writer:
            for traj in trajectories:
                writer.write(traj.to_dict())

    elif format == "json":
        with open(file_path, 'w') as f:
            json.dump([traj.to_dict() for traj in trajectories], f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_labeled_trajectories(
    labeled_trajectories: List[LabeledTrajectory],
    file_path: Union[str, Path],
    format: str = "jsonl",
    include_filtered: bool = False,
) -> None:
    """Save labeled trajectories to file.

    Args:
        labeled_trajectories: List of LabeledTrajectory objects
        file_path: Path to output file
        format: File format (jsonl, json)
        include_filtered: Whether to include filtered trajectories
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter if needed
    if not include_filtered:
        labeled_trajectories = [
            lt for lt in labeled_trajectories if not lt.is_filtered
        ]

    if format == "jsonl":
        with jsonlines.open(file_path, mode='w') as writer:
            for lt in labeled_trajectories:
                writer.write(lt.to_dict())

    elif format == "json":
        with open(file_path, 'w') as f:
            json.dump([lt.to_dict() for lt in labeled_trajectories], f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_training_samples(
    labeled_trajectories: List[LabeledTrajectory],
    file_path: Union[str, Path],
    format: str = "jsonl",
) -> None:
    """Save training samples (step-level) extracted from labeled trajectories.

    Args:
        labeled_trajectories: List of LabeledTrajectory objects
        file_path: Path to output file
        format: File format (jsonl, json)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract all training samples
    all_samples = []
    for lt in labeled_trajectories:
        if not lt.is_filtered:
            samples = lt.get_training_samples()
            all_samples.extend(samples)

    if format == "jsonl":
        with jsonlines.open(file_path, mode='w') as writer:
            for sample in all_samples:
                writer.write(sample)

    elif format == "json":
        with open(file_path, 'w') as f:
            json.dump(all_samples, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Saved {len(all_samples)} training samples to {file_path}")


def load_hotpotqa_questions(
    file_path: Union[str, Path],
    limit: int = None,
) -> List[Dict[str, Any]]:
    """Load HotpotQA questions from file.

    Args:
        file_path: Path to HotpotQA JSONL file
        limit: Maximum number of questions to load

    Returns:
        List of question dictionaries with keys:
            - _id: Question ID
            - question: Question text
            - answer: Gold answer
            - context: List of [title, sentences] pairs
            - supporting_facts: List of supporting fact indices
    """
    file_path = Path(file_path)
    questions = []

    with jsonlines.open(file_path) as reader:
        for i, obj in enumerate(reader):
            if limit and i >= limit:
                break
            questions.append(obj)

    return questions


def load_questions_jsonl(
    file_path: Union[str, Path],
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Load question records from a JSONL file.

    This is a dataset-agnostic loader that returns raw dicts as-is.
    """
    file_path = Path(file_path)
    questions: List[Dict[str, Any]] = []

    with jsonlines.open(file_path) as reader:
        for i, obj in enumerate(reader):
            if limit and i >= limit:
                break
            questions.append(obj)

    return questions


def load_musique_questions(
    file_path: Union[str, Path],
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Load MuSiQue questions from file.

    Expected PRMRAG format is produced by `scripts/prepare_musique_questions.py`.
    """
    return load_questions_jsonl(file_path, limit=limit)
