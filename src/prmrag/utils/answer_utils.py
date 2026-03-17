"""Utilities for extracting and normalizing answers from model outputs.

This module implements HotpotQA-style evaluation metrics:
- normalize_answer: Standard text normalization
- exact_match_score: Exact string match after normalization
- f1_score: Token-level F1 score
- cover_exact_match_score_1: All gold tokens in prediction (order-independent)
- cover_exact_match_score_2: Gold sequence in prediction (order-dependent)
"""

import re
import string
from collections import Counter
from typing import List, Tuple, Union


# =============================================================================
# Normalization
# =============================================================================

def normalize_answer(s: str) -> str:
    """HotpotQA standard normalization.

    1. Lowercase
    2. Remove punctuation (including quotes)
    3. Remove articles (a, an, the)
    4. Remove titles (Sir, Mr, Mrs, Dr, etc.)
    5. Fix whitespace
    6. Replace underscores with spaces
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_titles(text):
        return re.sub(r"\b(sir|mr|mrs|ms|dr|prof|professor|president|ceo|cfo|cto|coo|chairman|chairwoman|director|mayor|governor|senator|congressman|congresswoman|king|queen|prince|princess|duke|duchess|lord|lady|captain|general|colonel|admiral)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_titles(remove_punc(lower(replace_underscore(s))))))


def bool_mapping(s: str) -> str:
    """Map boolean strings to yes/no."""
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


# =============================================================================
# Evaluation Metrics
# =============================================================================

def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Compute token-level F1 score.

    Args:
        prediction: Predicted answer string
        ground_truth: Gold answer string

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    # Special handling for yes/no/noanswer
    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check exact match after normalization.

    Args:
        prediction: Predicted answer string
        ground_truth: Gold answer string

    Returns:
        True if exact match
    """
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )


def cover_exact_match_score_1(prediction: str, ground_truth: str) -> bool:
    """Check if all gold tokens appear in prediction (order-independent).

    This is the recommended metric for generative QA evaluation.

    Args:
        prediction: Predicted answer string
        ground_truth: Gold answer string

    Returns:
        True if all gold tokens are in prediction

    Examples:
        >>> cover_exact_match_score_1("Victor John Mature was an actor", "Victor Mature")
        True  # "victor" and "mature" both in prediction

        >>> cover_exact_match_score_1("alcohol content of beer", "ingredients in beer")
        False  # "ingredients" not in prediction
    """
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    pre_list = normalized_prediction.split()
    ground_list = normalized_ground_truth.split()

    # Special handling for yes/no questions
    # If gold is yes/no, check if pred's first word matches
    if normalized_ground_truth in ["yes", "no", "noanswer"]:
        if not pre_list:
            return False
        # Check if first word is yes/no
        return pre_list[0] == normalized_ground_truth

    # All gold tokens must appear in prediction (order-independent)
    return all(ground in pre_list for ground in ground_list)


def cover_exact_match_score_2(prediction: str, ground_truth: str) -> bool:
    """Check if gold sequence appears in prediction (order-dependent, contiguous).

    Args:
        prediction: Predicted answer string
        ground_truth: Gold answer string

    Returns:
        True if gold appears as contiguous subsequence in prediction

    Examples:
        >>> cover_exact_match_score_2("Victor Mature was born first", "Victor Mature")
        True  # "victor mature" appears contiguously

        >>> cover_exact_match_score_2("Victor John Mature", "Victor Mature")
        False  # "victor mature" not contiguous (John in between)
    """
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    # Special handling for yes/no
    if normalized_ground_truth in ["yes", "no", "noanswer"]:
        return normalized_prediction == normalized_ground_truth
    if normalized_prediction in ["yes", "no", "noanswer"]:
        return normalized_prediction == normalized_ground_truth

    pre_list = normalized_prediction.split()
    ground_list = normalized_ground_truth.split()

    # Check for contiguous subsequence
    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i : i + len(ground_list)] == ground_list:
            return True

    # Also check substring match
    pre_str = " ".join(pre_list)
    ground_str = " ".join(ground_list)
    if ground_str in pre_str:
        return True

    return False


# =============================================================================
# Multi-Ground-Truth Support
# =============================================================================

def metric_max_over_ground_truths(
    metric_fn,
    prediction: str,
    ground_truths: List[str]
) -> Union[bool, Tuple[float, float, float]]:
    """Compute metric over multiple ground truths and return max.

    Args:
        metric_fn: Metric function to use
        prediction: Predicted answer string
        ground_truths: List of gold answer strings

    Returns:
        Maximum score across all ground truths
    """
    scores_for_ground_truths = []

    if metric_fn.__name__ == "f1_score":
        for ground_truth in ground_truths:
            f1, prec, recall = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append((f1, prec, recall))
        return max(scores_for_ground_truths, key=lambda x: x[0])
    else:
        # For exact_match_score, cover_exact_match_score_1, cover_exact_match_score_2
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


# =============================================================================
# Main Evaluation Function
# =============================================================================

def check_answer_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth using cover_exact_match_score_1.

    This is the main function for answer matching in PRMRAG.
    Uses cover_exact_match_score_1 which checks if all gold tokens
    appear in the prediction (order-independent).

    Args:
        prediction: Predicted answer string
        ground_truth: Gold answer string

    Returns:
        True if answer matches
    """
    return cover_exact_match_score_1(prediction, ground_truth)


def compute_all_metrics(
    prediction: str,
    ground_truths: Union[str, List[str]]
) -> dict:
    """Compute all evaluation metrics.

    Args:
        prediction: Predicted answer string
        ground_truths: Gold answer string or list of strings

    Returns:
        Dictionary with all metrics
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1, prec, recall = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    cover_em_1 = metric_max_over_ground_truths(cover_exact_match_score_1, prediction, ground_truths)
    cover_em_2 = metric_max_over_ground_truths(cover_exact_match_score_2, prediction, ground_truths)

    return {
        "em": em,
        "f1": f1,
        "precision": prec,
        "recall": recall,
        "cover_em_1": cover_em_1,
        "cover_em_2": cover_em_2,
    }


# =============================================================================
# Answer Extraction (kept from original)
# =============================================================================

def _clean_answer_span(span: str) -> str:
    """Trim whitespace/brackets and trailing periods."""
    span = span.strip()
    span = re.sub(r'^[()\s]+|[()\s]+$', '', span)
    return span.rstrip('.').strip()


def extract_answer_from_text(text: str) -> str:
    """Extract the most plausible answer span from model output.

    Priority:
    1) XML <answer>...</answer> tag (preferred)
    2) Finish[answer="..."] pattern (ReAct format, legacy)
    3) After explicit "Final Answer:" marker (supports multiline)
    4) Common answer phrasings ("the answer is", "therefore, the answer is", etc.)
    5) First non-empty line that is not an intermediate/next-query marker
    """
    if not text:
        return ""

    text = text.strip()

    # 1) XML <answer>...</answer> tag (preferred)
    xml_answer_match = re.search(r'<answer>(.+?)</answer>', text, flags=re.DOTALL)
    if xml_answer_match:
        candidate = _clean_answer_span(xml_answer_match.group(1))
        if candidate:
            return candidate

    # 2) Finish[answer="..."] pattern (legacy)
    # Support both double and single quotes, and handle quotes within the answer
    # Match: Finish[answer="..."] or Finish[answer='...'] (greedy until closing bracket)
    finish_match = re.search(r'(?:Action:\s*)?Finish\[answer\s*=\s*["\'](.+?)["\']\]', text, flags=re.IGNORECASE | re.DOTALL)
    if finish_match:
        candidate = _clean_answer_span(finish_match.group(1))
        if candidate:
            return candidate

    # 2) Explicit Final Answer
    final_match = re.search(r'final\s+answer\s*:?\s*(.+)', text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        remainder = final_match.group(1).strip()

        explanation_markers = [
            r'\s+(?:because|since|as|because of|due to|owing to|given that)\s+',
            r'\s+(?:which|that|who|where|when)\s+',
            r'[.,;]\s+(?:This|It|They|He|She|The)',
        ]

        for marker in explanation_markers:
            match = re.search(marker, remainder, re.IGNORECASE)
            if match:
                remainder = remainder[:match.start()].strip()
                break

        words = remainder.split()
        if len(words) > 15:
            sentence_match = re.search(r'^([^.!?]+)[.!?]', remainder)
            if sentence_match:
                remainder = sentence_match.group(1).strip()
            else:
                remainder = ' '.join(words[:10])

        for line in remainder.splitlines():
            candidate = _clean_answer_span(line)
            if candidate:
                return candidate

        candidate = _clean_answer_span(remainder)
        if candidate:
            return candidate

    # 3) Common phrasings
    answer_patterns = [
        r'therefore,?\s+(?:the\s+answer\s+is\s*:?\s*)?(.+?)(?:(?<![A-Z])\.(?=\s+[A-Z])|[;]|\s+(?:because|since)\s+|$)',
        r'(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:(?<![A-Z])\.(?=\s+[A-Z])|[;]|\s+(?:because|since)\s+|$)',
        r'(?:in\s+)?conclusion,?\s+(.+?)(?:(?<![A-Z])\.(?=\s+[A-Z])|[;]|\s+(?:because|since)\s+|$)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = _clean_answer_span(match.group(1))
            if candidate:
                return candidate

    # 4) Parenthetical tail
    paren_match = re.search(r'\(([^)]+)\)[.,;]?\s*$', text, flags=re.DOTALL)
    if paren_match:
        candidate = _clean_answer_span(paren_match.group(1))
        if candidate:
            return candidate

    # 5) Fallback: first non-empty line
    skip_prefixes = ("intermediate answer", "missing info", "next query target")
    for line in text.splitlines():
        if not line.strip():
            continue
        lower_line = line.strip().lower()
        if lower_line.startswith(skip_prefixes):
            continue
        candidate = _clean_answer_span(line)
        if candidate:
            return candidate

    return ""
