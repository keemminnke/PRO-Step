"""Simple trajectory generation for no-RPE branch.

This module implements trajectory generation without MC rollouts:
- Generate steps based on model's explicit actions (Search/Finish)
- Labels come from Judge model instead of RPE
- Supports efficient batch processing with vLLM
- Uses XML tag format: <think>, <search>, <answer>, <documents>
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
from tqdm import tqdm


class StepType(str, Enum):
    """Type of step in adaptive trajectory."""
    COT = "cot"          # Pure reasoning step
    RAG = "rag"          # Retrieval-augmented step
    ANSWER = "answer"    # Final answer


@dataclass
class AdaptiveStep:
    """A step in an adaptive MC-CoT + RAG trajectory.

    This includes MC values before/after for monitoring.
    """
    step_id: int
    step_type: StepType
    text: str                           # The full step content
    used_passages: List[Dict[str, Any]] # Retrieved passages (if RAG)
    mc_before: float                    # MC(s_{t-1})
    mc_after: float                     # MC(s_t)
    rpe: float                          # mc_after / mc_before
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step_id': self.step_id,
            'step_type': self.step_type.value,
            'text': self.text,
            'used_passages': self.used_passages,
            'metadata': self.metadata,
        }


@dataclass
class AdaptiveTrajectory:
    """An adaptive RAG-CoT trajectory."""
    trajectory_id: str
    question: str
    steps: List[AdaptiveStep]
    final_answer: str
    gold_answer: Optional[str] = None
    supporting_facts: Optional[List[str]] = None
    is_correct: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trajectory_id': self.trajectory_id,
            'question': self.question,
            'steps': [s.to_dict() for s in self.steps],
            'final_answer': self.final_answer,
            'gold_answer': self.gold_answer,
            'supporting_facts': self.supporting_facts,
            'is_correct': self.is_correct,
            'metadata': self.metadata,
        }


class SimpleTrajectoryGenerator:
    """Simple trajectory generator without MC rollouts.

    This generator relies on the model's explicit action (Search/Finish)
    instead of using MC-based RPE for decision making.

    Benefits:
    - Much faster (no rollouts needed)
    - Can be batch processed
    - Labels come from Judge model instead of RPE

    Flow:
    1. Generate step with model
    2. Parse action from output (Search/Finish)
    3. If Search: retrieve docs, add observation, continue
    4. If Finish: extract answer, stop
    """

    def __init__(
        self,
        policy_model,
        retriever,
        config: Dict[str, Any],
    ):
        """Initialize simple trajectory generator.

        Args:
            policy_model: Model for generation
            retriever: Retriever for RAG
            config: Configuration dict with:
                - max_steps: Maximum steps per trajectory
                - top_k_passages: Number of passages to retrieve
                - temperature: Sampling temperature
        """
        self.policy_model = policy_model
        self.retriever = retriever

        self.max_steps = config.get('max_steps', 10)
        self.top_k_passages = config.get('top_k_passages', 5)
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens_per_step = config.get('max_tokens_per_step', 4096)

    def generate_trajectory(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        supporting_facts: Optional[List[str]] = None,
        trajectory_id: Optional[str] = None,
    ) -> Optional[AdaptiveTrajectory]:
        """Generate a trajectory using action-based flow.

        Args:
            question: The question to answer
            gold_answer: Gold answer (for evaluation only)
            supporting_facts: Supporting facts (optional)
            trajectory_id: Unique ID for trajectory

        Returns:
            AdaptiveTrajectory or None if generation failed
        """
        import re

        trajectory_id = trajectory_id or f"traj_{hash(question)}"

        steps = []
        context = ""  # Accumulated retrieved passages
        previous_contents = []  # Track step contents for continuation prompt

        for step_num in range(1, self.max_steps + 1):
            # Build prompt — all steps go into the assistant turn (matches KTO training format)
            if step_num == 1:
                prompt = self.policy_model.format_prompt_for_qwen(
                    self._build_initial_prompt(question)
                )
            else:
                prompt = self.policy_model.format_continuation_prompt_for_qwen(
                    question, previous_contents
                )

            # Generate step (stop before <documents> to prevent hallucination)
            response = self.policy_model.generate(
                prompt=prompt,
                max_tokens=self.max_tokens_per_step,
                temperature=self.temperature,
                top_p=0.95,
                stop_sequences=["<documents>", "\n<documents>"],
            )

            # Parse response
            step_content = self._parse_step_response(response, step_num)

            # Track for continuation
            previous_contents.append(step_content)

            # Detect action
            action_type, action_input = self._parse_action(step_content)

            if action_type == "finish":
                # Final answer step
                step = AdaptiveStep(
                    step_id=step_num,
                    step_type=StepType.ANSWER,
                    text=step_content,
                    used_passages=[],
                    mc_before=0.0,  # Placeholder - Judge will provide labels
                    mc_after=0.0,
                    rpe=0.0,
                    metadata={'action': 'finish'},
                )
                steps.append(step)
                break

            elif action_type == "search":
                # RAG step - retrieve and continue
                query = action_input or question
                passages = self.retriever.retrieve(query, top_k=self.top_k_passages)

                # Format observation as XML <documents>
                observation = self._format_observation(passages)
                full_content = f"{step_content}\n{observation}"

                step = AdaptiveStep(
                    step_id=step_num,
                    step_type=StepType.RAG,
                    text=full_content,
                    used_passages=passages,
                    mc_before=0.0,
                    mc_after=0.0,
                    rpe=0.0,
                    metadata={'action': 'search', 'query': query},
                )
                steps.append(step)

                # Update context for next step
                context = observation
                # Update previous content with observation
                previous_contents[-1] = full_content

            else:
                # Pure reasoning step (no action detected)
                step = AdaptiveStep(
                    step_id=step_num,
                    step_type=StepType.COT,
                    text=step_content,
                    used_passages=[],
                    mc_before=0.0,
                    mc_after=0.0,
                    rpe=0.0,
                    metadata={'action': 'reason'},
                )
                steps.append(step)

        # Extract final answer
        final_answer = self._extract_final_answer(steps)
        is_correct = self._check_answer(final_answer, gold_answer) if gold_answer else None

        return AdaptiveTrajectory(
            trajectory_id=trajectory_id,
            question=question,
            steps=steps,
            final_answer=final_answer,
            gold_answer=gold_answer,
            supporting_facts=supporting_facts,
            is_correct=is_correct,
            metadata={
                'num_steps': len(steps),
                'generator': 'simple',
            },
        )

    def _build_initial_prompt(self, question: str) -> str:
        """Build prompt for first step (instruction handled in system prompt)."""
        return f"Question: {question}"

    def _build_continuation_prompt(
        self,
        question: str,
        previous_contents: List[str],
    ) -> str:
        """Build prompt for continuation with previous steps.

        Args:
            question: The question being answered
            previous_contents: List of previous step contents (including observations)
        """
        steps_text = "\n\n".join(previous_contents)
        return f"Question: {question}\n\n{steps_text}"

    def _parse_step_response(self, response: str, step_num: int) -> str:
        """Parse step content from response."""
        import re

        # Remove any future steps
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        # Remove "Step N:" prefix if present
        step_prefix = re.match(rf'^Step\s+{step_num}:\s*', response)
        if step_prefix:
            response = response[step_prefix.end():]

        return response.strip()

    def _parse_action(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse action from step content.

        Supports both XML format and legacy format:

        XML format (preferred):
        - <search>query</search>
        - <answer>final answer</answer>

        Legacy format (backward compatible):
        - Action: Finish[answer="..."]
        - Action: Search[query="..."]

        Returns:
            Tuple of (action_type, action_input)
            action_type: 'search', 'finish', 'reason', or None
        """
        import re

        # === XML format (preferred) ===

        # Check for <answer>...</answer> tag
        answer_match = re.search(r'<answer>(.+?)</answer>', content, re.DOTALL)
        if answer_match:
            return ('finish', answer_match.group(1).strip())

        # Check for <search>...</search> tag
        search_match = re.search(r'<search>(.+?)</search>', content, re.DOTALL)
        if search_match:
            return ('search', search_match.group(1).strip())

        # === Legacy format (backward compatible) ===

        # Check for Finish action
        finish_match = re.search(
            r'Action:\s*Finish\[answer=["\']?(.+?)["\']?\]',
            content,
            re.IGNORECASE | re.DOTALL
        )
        if finish_match:
            return ('finish', finish_match.group(1).strip())

        # Check for Search action
        search_match = re.search(
            r'Action:\s*Search\[query=["\']?(.+?)["\']?\]',
            content,
            re.IGNORECASE | re.DOTALL
        )
        if search_match:
            return ('search', search_match.group(1).strip())

        # Check for Reason action
        reason_match = re.search(
            r'Action:\s*Reason\[content=["\']?(.+?)["\']?\]',
            content,
            re.IGNORECASE | re.DOTALL
        )
        if reason_match:
            return ('reason', reason_match.group(1).strip())

        # If <think> exists but no action tag, it's a reason step
        think_match = re.search(r'<think>(.+?)</think>', content, re.DOTALL)
        if think_match:
            return ('reason', think_match.group(1).strip())


        # Fallback: Check for simple finish markers (only if no <think> tag)
        if any(marker in content.lower() for marker in ['final answer:', 'the answer is', 'therefore, the answer']):
            return ('finish', None)

        return (None, None)

    def _format_observation(self, passages: List[Dict[str, Any]]) -> str:
        """Format retrieved passages as XML <documents> tag."""
        obs_parts = []
        for i, p in enumerate(passages, 1):
            title = p.get('title', 'Unknown')
            text = p.get('text', p.get('content', ''))[:500]
            obs_parts.append(f"[{i}] {title}: {text}")
        return "<documents>\n" + "\n".join(obs_parts) + "\n</documents>"

    def _extract_final_answer(self, steps: List[AdaptiveStep]) -> str:
        """Extract final answer from trajectory."""
        import re

        if not steps:
            return ""

        # Check last step for Finish action
        last_content = steps[-1].text

        # Try XML <answer>...</answer> pattern (preferred)
        answer_match = re.search(r'<answer>(.+?)</answer>', last_content, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        # Try Finish[answer="..."] pattern (legacy)
        finish_match = re.search(
            r'Finish\[answer=["\']?(.+?)["\']?\]',
            last_content,
            re.IGNORECASE | re.DOTALL
        )
        if finish_match:
            return finish_match.group(1).strip()

        # Try "Final Answer:" pattern
        final_match = re.search(r'final\s+answer:\s*(.+)', last_content, re.IGNORECASE)
        if final_match:
            return final_match.group(1).strip().split('\n')[0]

        # Try "the answer is" pattern
        answer_match = re.search(r'the\s+answer\s+is\s*:?\s*(.+)', last_content, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip().split('\n')[0]

        return ""

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold."""
        if not predicted or not gold:
            return False
        pred_norm = predicted.strip().lower()
        gold_norm = gold.strip().lower()
        return pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm

    def generate_batch(
        self,
        questions: List[Dict[str, Any]],
        num_samples: int = 1,
        show_progress: bool = True,
    ) -> List[AdaptiveTrajectory]:
        """Generate trajectories for multiple questions using vLLM batch processing.

        This method processes multiple trajectories in parallel by batching
        the same step across all active trajectories, maximizing vLLM efficiency.

        For no-RPE pipeline:
        - Generate num_samples trajectories per question (e.g., 16)
        - All trajectories processed in parallel
        - Judge model will label each step afterwards

        Args:
            questions: List of question dicts with 'question', 'gold_answer', etc.
            num_samples: Number of trajectories to generate per question (default: 1)
            show_progress: Show progress bar

        Returns:
            List of generated trajectories (len = len(questions) * num_samples)
        """
        import re

        total_trajectories = len(questions) * num_samples
        if show_progress:
            print(f"Generating {total_trajectories} trajectories ({len(questions)} questions x {num_samples} samples)")

        # Initialize trajectory states (questions * num_samples)
        states = []
        for q_data in questions:
            question_id = q_data.get('_id', q_data.get('id', str(hash(q_data['question']))))
            for sample_idx in range(num_samples):
                states.append({
                    'question': q_data['question'],
                    'gold_answer': q_data.get('gold_answer', q_data.get('answer')),
                    'supporting_facts': q_data.get('supporting_facts'),
                    'question_id': question_id,
                    'sample_idx': sample_idx,
                    'trajectory_id': f"{question_id}_sample_{sample_idx}",
                    'steps': [],
                    'previous_contents': [],  # Track step contents for continuation
                    'finished': False,
                    'current_step': 1,
                })

        # Process steps in batches
        for step_num in range(1, self.max_steps + 1):
            # Get active (unfinished) trajectories
            active_indices = [i for i, s in enumerate(states) if not s['finished']]

            if not active_indices:
                break

            if show_progress:
                print(f"  Step {step_num}: {len(active_indices)} active trajectories")

            # Build prompts for all active trajectories
            # All steps go into the assistant turn (matches KTO training format)
            formatted_prompts = []
            skipped_for_length = []
            max_model_len = getattr(self.policy_model, 'max_model_len', None)
            if max_model_len is None and hasattr(self.policy_model, 'llm'):
                try:
                    max_model_len = self.policy_model.llm.llm_engine.model_config.max_model_len
                except Exception:
                    pass
            for idx in active_indices:
                state = states[idx]
                if step_num == 1:
                    prompt = self.policy_model.format_prompt_for_qwen(
                        self._build_initial_prompt(state['question'])
                    )
                else:
                    prompt = self.policy_model.format_continuation_prompt_for_qwen(
                        state['question'],
                        state['previous_contents'],
                    )
                # Check prompt length to avoid exceeding max_model_len
                if max_model_len is not None:
                    token_len = len(self.policy_model.tokenizer.encode(prompt))
                    if token_len >= max_model_len - 10:
                        skipped_for_length.append(idx)
                        continue
                formatted_prompts.append(prompt)

            # Remove skipped indices from active_indices
            if skipped_for_length:
                active_indices = [i for i in active_indices if i not in skipped_for_length]
                for idx in skipped_for_length:
                    states[idx]['finished'] = True
                if show_progress:
                    print(f"    Skipped {len(skipped_for_length)} trajectories (prompt too long)")
                if not active_indices:
                    continue

            # Batch generate with vLLM
            # Stop before <documents> to prevent model from hallucinating observations
            stop_sequences = ["<documents>", "\n<documents>"]
            responses = self.policy_model.batch_generate(
                prompts=formatted_prompts,
                max_tokens=self.max_tokens_per_step,
                temperature=self.temperature,
                top_p=0.95,
                stop_sequences=stop_sequences,
            )

            # Phase 1: Parse all responses and identify actions
            parsed_results = []  # (idx, step_content, action_type, action_input)
            search_actions = []  # (result_idx, state_idx, query)

            for i, idx in enumerate(active_indices):
                state = states[idx]
                response = responses[i]

                # Parse response
                step_content = self._parse_step_response(response, step_num)

                # Track for continuation
                state['previous_contents'].append(step_content)

                # Detect action
                action_type, action_input = self._parse_action(step_content)
                parsed_results.append((idx, step_content, action_type, action_input))

                # Collect search queries for batch retrieval
                if action_type == "search":
                    query = action_input or state['question']
                    search_actions.append((len(parsed_results) - 1, idx, query))

            # Debug: count action types
            action_counts = {'search': 0, 'finish': 0, 'reason': 0, 'none': 0}
            for _, _, action_type, _ in parsed_results:
                if action_type == 'search':
                    action_counts['search'] += 1
                elif action_type == 'finish':
                    action_counts['finish'] += 1
                elif action_type == 'reason':
                    action_counts['reason'] += 1
                else:
                    action_counts['none'] += 1
            print(f"    Actions: search={action_counts['search']}, finish={action_counts['finish']}, reason={action_counts['reason']}, none={action_counts['none']}")

            # Phase 2: Batch retrieval for all Search actions
            all_passages = {}  # result_idx -> passages
            if search_actions and self.retriever is not None:
                queries = [sa[2] for sa in search_actions]

                # Batch retrieve if supported, otherwise sequential
                if hasattr(self.retriever, 'batch_retrieve'):
                    passages_list = self.retriever.batch_retrieve(queries, top_k=self.top_k_passages)
                else:
                    passages_list = [self.retriever.retrieve(q, top_k=self.top_k_passages) for q in queries]

                # Map back to result indices
                for (result_idx, state_idx, query), passages in zip(search_actions, passages_list):
                    all_passages[result_idx] = passages

            # Phase 3: Build steps with retrieved passages
            for result_idx, (idx, step_content, action_type, action_input) in enumerate(parsed_results):
                state = states[idx]

                if action_type == "finish":
                    # Final answer step
                    step = AdaptiveStep(
                        step_id=step_num,
                        step_type=StepType.ANSWER,
                        text=step_content,
                        used_passages=[],
                        mc_before=0.0,
                        mc_after=0.0,
                        rpe=0.0,
                        metadata={'action': 'answer'},
                    )
                    state['steps'].append(step)
                    state['finished'] = True

                elif action_type == "search":
                    # RAG step - use pre-fetched passages
                    query = action_input or state['question']
                    passages = all_passages.get(result_idx, [])

                    # Format observation as XML <documents>
                    observation = self._format_observation(passages)
                    full_content = f"{step_content}\n{observation}"

                    step = AdaptiveStep(
                        step_id=step_num,
                        step_type=StepType.RAG,
                        text=full_content,
                        used_passages=passages,
                        mc_before=0.0,
                        mc_after=0.0,
                        rpe=0.0,
                        metadata={'action': 'search', 'query': query},
                    )
                    state['steps'].append(step)

                    # Update previous content with observation
                    state['previous_contents'][-1] = full_content

                else:
                    # Pure reasoning step
                    step = AdaptiveStep(
                        step_id=step_num,
                        step_type=StepType.COT,
                        text=step_content,
                        used_passages=[],
                        mc_before=0.0,
                        mc_after=0.0,
                        rpe=0.0,
                        metadata={'action': 'reason'},
                    )
                    state['steps'].append(step)

                state['current_step'] = step_num + 1

        # Build final trajectories
        all_trajectories = []
        for state in states:
            if not state['steps']:
                continue

            final_answer = self._extract_final_answer(state['steps'])
            is_correct = self._check_answer(final_answer, state['gold_answer']) if state['gold_answer'] else None

            trajectory = AdaptiveTrajectory(
                trajectory_id=state['trajectory_id'],
                question=state['question'],
                steps=state['steps'],
                final_answer=final_answer,
                gold_answer=state['gold_answer'],
                supporting_facts=state['supporting_facts'],
                is_correct=is_correct,
                metadata={
                    'num_steps': len(state['steps']),
                    'generator': 'simple_batch',
                    'question_id': state.get('question_id'),
                    'sample_idx': state.get('sample_idx'),
                },
            )
            all_trajectories.append(trajectory)

        return all_trajectories
