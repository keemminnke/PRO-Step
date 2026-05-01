"""Question classifier for critic-guided trajectory regeneration.

Classifies questions into:
- Case A: At least one correct trajectory exists → rejection sampling (best critic_min)
- Case B: All trajectories are wrong → needs regeneration

Uses original trajectories (step content) + critic scores (binary classification).
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


@dataclass
class ScoredTrajectory:
    """A trajectory with critic scores."""
    trajectory_id: str
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    is_correct: bool
    critic_min: float
    critic_step_scores: List[float]  # soft scores (for ranking/voting)
    critic_step_labels: List[int]  # binary labels 0/1 (for BAD step detection)
    steps: List[Dict[str, Any]]  # from original trajectories (has think, search, documents)


@dataclass
class ClassifiedQuestion:
    """A question classified as Case A or Case B."""
    question_id: str
    question: str
    gold_answer: str
    case: str  # "A" or "B"
    best_trajectory: ScoredTrajectory
    all_trajectories: List[ScoredTrajectory]
    num_correct: int
    num_total: int


class QuestionClassifier:
    """Classify questions into Case A (has correct) vs Case B (all wrong)."""

    @staticmethod
    def _check_answer(predicted: str, gold: str) -> bool:
        """Cover EM with quote normalization (same as regenerator)."""
        if not predicted or not gold:
            return False
        pred_norm = predicted.strip().lower()
        gold_norm = gold.strip().lower()
        if pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm:
            return True
        # Strip quotes and retry
        pred_clean = pred_norm.replace('"', '').replace("'", '').replace('\u201c', '').replace('\u201d', '')
        gold_clean = gold_norm.replace('"', '').replace("'", '').replace('\u201c', '').replace('\u201d', '')
        return pred_clean == gold_clean or gold_clean in pred_clean or pred_clean in gold_clean

    @staticmethod
    def load_and_merge(
        trajectories_path: str,
        critic_scores_path: str,
    ) -> List[ScoredTrajectory]:
        """Load original trajectories and critic scores, merge by trajectory_id.

        Args:
            trajectories_path: Path to original trajectories JSONL (step content)
            critic_scores_path: Path to critic_scores JSONL (critic binary classification)

        Returns:
            List of ScoredTrajectory with merged data
        """
        # Load original trajectories (has step content: think, search, documents)
        traj_data = {}
        with open(trajectories_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    traj_data[record['trajectory_id']] = record

        # Load critic scores (has critic_step_scores, critic_min)
        critic_data = {}
        with open(critic_scores_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    critic_data[record['trajectory_id']] = record

        # Merge by trajectory_id
        merged = []
        common_ids = set(traj_data.keys()) & set(critic_data.keys())

        for traj_id in sorted(common_ids):
            traj = traj_data[traj_id]
            critic = critic_data[traj_id]

            # Extract question_id from trajectory_id
            if '_sample_' in traj_id:
                question_id = traj_id.rsplit('_sample_', 1)[0]
            else:
                question_id = traj_id

            # Binary labels: from JSONL if present, else derive from scores (threshold 0.5)
            if 'critic_step_labels' in critic:
                step_labels = critic['critic_step_labels']
            else:
                step_labels = [1 if s > 0.5 else 0 for s in critic['critic_step_scores']]

            merged.append(ScoredTrajectory(
                trajectory_id=traj_id,
                question_id=question_id,
                question=traj['question'],
                gold_answer=traj.get('gold_answer', ''),
                predicted_answer=traj.get('final_answer', traj.get('predicted_answer', '')),
                is_correct=traj['is_correct'],
                critic_min=critic['critic_min'],
                critic_step_scores=critic['critic_step_scores'],
                critic_step_labels=step_labels,
                steps=traj['steps'],
            ))

        print(f"Loaded {len(traj_data)} trajectories, {len(critic_data)} critic scores")
        print(f"Merged {len(merged)} trajectories ({len(common_ids)} common IDs)")

        return merged

    @staticmethod
    def classify_questions(
        scored_trajectories: List[ScoredTrajectory],
    ) -> Tuple[List[ClassifiedQuestion], Dict[str, Any]]:
        """Classify questions into Case A and Case B.

        Case A: At least one correct trajectory → pick best correct (highest critic_min)
        Case B: All trajectories wrong → pick least-wrong (highest critic_min)

        Args:
            scored_trajectories: List of merged trajectories

        Returns:
            Tuple of (classified_questions, stats_dict)
        """
        # Group by question_id
        by_question: Dict[str, List[ScoredTrajectory]] = defaultdict(list)
        for traj in scored_trajectories:
            by_question[traj.question_id].append(traj)

        classified = []
        case_a_count = 0
        case_b_count = 0

        for qid, trajs in sorted(by_question.items()):
            # Re-check correctness with quote normalization
            gold_answer = trajs[0].gold_answer
            for t in trajs:
                t.is_correct = QuestionClassifier._check_answer(t.predicted_answer, gold_answer)
            correct_trajs = [t for t in trajs if t.is_correct]
            question = trajs[0].question

            if correct_trajs:
                # Case A: pick correct trajectory with highest critic_min
                best = max(correct_trajs, key=lambda t: t.critic_min)
                case = "A"
                case_a_count += 1
            else:
                # Case B: pick trajectory with highest critic_min (least wrong)
                best = max(trajs, key=lambda t: t.critic_min)
                case = "B"
                case_b_count += 1

            classified.append(ClassifiedQuestion(
                question_id=qid,
                question=question,
                gold_answer=gold_answer,
                case=case,
                best_trajectory=best,
                all_trajectories=trajs,
                num_correct=len(correct_trajs),
                num_total=len(trajs),
            ))

        stats = {
            'total_questions': len(classified),
            'case_a': case_a_count,
            'case_b': case_b_count,
            'total_trajectories': len(scored_trajectories),
            'case_a_pct': case_a_count / len(classified) * 100 if classified else 0,
            'case_b_pct': case_b_count / len(classified) * 100 if classified else 0,
        }

        return classified, stats
