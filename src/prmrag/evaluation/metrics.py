"""Evaluation metrics for consensus labeling."""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr


def compute_agreement_metrics(
    rpe_labels: List[str],
    judge_labels: List[str],
) -> Dict[str, float]:
    """Compute agreement metrics between RPE and Judge labels.

    Args:
        rpe_labels: List of RPE labels (GOOD, BORDERLINE, BAD)
        judge_labels: List of Judge labels (GOOD, BAD)

    Returns:
        Dictionary of metrics
    """
    # Convert to binary for comparison
    rpe_binary = [1 if label == "GOOD" else 0 for label in rpe_labels]
    judge_binary = [1 if label == "GOOD" else 0 for label in judge_labels]

    # Exact agreement rate
    agreement_rate = sum(r == j for r, j in zip(rpe_binary, judge_binary)) / len(rpe_binary)

    # Cohen's kappa
    kappa = cohen_kappa_score(rpe_binary, judge_binary)

    return {
        "agreement_rate": agreement_rate,
        "cohens_kappa": kappa,
    }


def compute_label_quality_metrics(
    predicted_labels: List[int],
    gold_labels: List[int],
) -> Dict[str, float]:
    """Compute label quality metrics against gold labels.

    Args:
        predicted_labels: Predicted labels (0 or 1)
        gold_labels: Gold labels (0 or 1)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(gold_labels, predicted_labels)
    precision = precision_score(gold_labels, predicted_labels, zero_division=0)
    recall = recall_score(gold_labels, predicted_labels, zero_division=0)
    f1 = f1_score(gold_labels, predicted_labels, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_coverage(
    total_samples: int,
    filtered_samples: int,
) -> Dict[str, float]:
    """Compute coverage metrics.

    Args:
        total_samples: Total number of samples
        filtered_samples: Number of filtered samples

    Returns:
        Dictionary of coverage metrics
    """
    kept_samples = total_samples - filtered_samples

    return {
        "total_samples": total_samples,
        "kept_samples": kept_samples,
        "filtered_samples": filtered_samples,
        "coverage_rate": kept_samples / total_samples if total_samples > 0 else 0,
        "filter_rate": filtered_samples / total_samples if total_samples > 0 else 0,
    }
