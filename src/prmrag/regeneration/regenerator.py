"""Critic-guided trajectory regenerator.

Uses multi-turn chat format (GenPRM pattern):
  Turn 1 (user): Question
  Turn 2 (assistant): Full original trajectory (all steps)
  Turn 3 (user): Critic feedback pointing out bad step + reasoning

This keeps the feedback in-distribution for the model since it's a natural
multi-turn conversation pattern (user corrects the assistant).
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from .truncator import TruncationResult

SYSTEM_PROMPT = """You are a helpful assistant who is good at answering questions with multi-turn search engine calling. To answer questions, you must first reason through the available information using <think> and </think>. If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with relevant documents enclosed in <documents> and </documents>. You can search as many times as you want. Once you have sufficient information or if you find no further external knowledge is needed, directly provide a concise final answer using <answer> and </answer> without detailed illustrations."""


class CriticGuidedRegenerator:
    """Regenerate trajectories using multi-turn critic feedback (GenPRM pattern)."""

    def __init__(
        self,
        policy_model,
        retriever,
        max_steps: int = 10,
        top_k_passages: int = 5,
        temperature: float = 0.7,
    ):
        self.policy_model = policy_model
        self.retriever = retriever
        self.max_steps = max_steps
        self.top_k_passages = top_k_passages
        self.temperature = temperature

    def _reconstruct_step_text(self, step: Dict[str, Any]) -> str:
        """Reconstruct a step's text from its parsed fields."""
        parts = []
        if step.get('think'):
            parts.append(f"<think>{step['think']}</think>")
        if step.get('search'):
            parts.append(f"<search>{step['search']}</search>")
        if step.get('documents'):
            parts.append(f"<documents>\n{step['documents']}\n</documents>")
        if step.get('answer'):
            parts.append(f"<answer>{step['answer']}</answer>")
        return "\n".join(parts)

    def _build_multi_turn_prompt(self, truncation_result: TruncationResult) -> str:
        """Build multi-turn prompt with full original trajectory (documents preserved)
        + feedback with query rewriting guidance.

        Messages:
          system: Adaptive RAG system prompt
          user: Question
          assistant: Full original trajectory (all steps WITH documents)
          user: Critic feedback + re-read docs instruction + query rewriting guidance

        Returns:
            Formatted prompt string ready for model.generate()
        """
        # Include ALL original steps with documents in assistant turn
        # So the model can re-read the retrieved information
        all_step_texts = []
        original_queries = []
        for step in truncation_result.all_original_steps:
            all_step_texts.append(self._reconstruct_step_text(step))
            if step.get('search'):
                original_queries.append(step['search'])
        full_trajectory = "\n\n".join(all_step_texts)

        # Build feedback with:
        # 1. What went wrong (bad step + critic reasoning)
        # 2. Re-read instruction for existing documents
        # 3. Query rewriting guidance (don't repeat same queries)
        bad_step_text = self._reconstruct_step_text(truncation_result.bad_step)
        reasoning = truncation_result.critic_reasoning

        feedback_parts = [
            "Your answer was incorrect. This step had a problem:",
            f">{bad_step_text}",
        ]
        if reasoning:
            feedback_parts.append(f"\nReason: {reasoning}")

        feedback_parts.append(
            "\nPlease re-read the retrieved documents above carefully. "
            "The correct answer might already be in the documents you retrieved."
        )

        if original_queries:
            queries_str = ", ".join(f'"{q}"' for q in original_queries)
            feedback_parts.append(
                f"\nIf you need to search again, do NOT repeat these queries: {queries_str}. "
                "Use a DIFFERENT search query with alternative keywords or phrasing."
            )

        feedback_parts.append(
            "\nNow try again with <think> and either <search> (different query) or <answer>."
        )
        critic_content = "\n".join(feedback_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {truncation_result.question}"},
            {"role": "assistant", "content": full_trajectory},
            {"role": "user", "content": critic_content},
        ]

        return self.policy_model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def regenerate_single(
        self,
        truncation_result: TruncationResult,
    ) -> Dict[str, Any]:
        """Regenerate a single trajectory using multi-turn critic feedback.

        1. Build multi-turn prompt (user→assistant→user feedback)
        2. Generate new steps iteratively (search→retrieve→continue)
        3. Return combined trajectory
        """
        # Build base prompt with multi-turn format
        base_prompt = self._build_multi_turn_prompt(truncation_result)

        # Generate new steps from the feedback point
        new_steps = []
        accumulated_response = ""
        remaining_steps = self.max_steps - len(truncation_result.good_steps)

        for step_offset in range(remaining_steps):
            step_num = truncation_result.bad_step_id + step_offset

            # Current prompt = base + accumulated new content
            current_prompt = base_prompt + accumulated_response

            response = self.policy_model.generate(
                prompt=current_prompt,
                max_tokens=800,
                temperature=self.temperature,
                top_p=0.95,
                stop_sequences=["<documents>", "\n<documents>"],
            )

            step_content = self._parse_step_response(response, step_num)
            action_type, action_input = self._parse_action(step_content)

            if action_type == "finish":
                new_steps.append({
                    'step_id': step_num,
                    'step_type': 'answer',
                    'text': step_content,
                    **self._parse_step_fields(step_content),
                })
                accumulated_response += step_content
                break

            elif action_type == "search":
                query = action_input or truncation_result.question
                passages = self.retriever.retrieve(query, top_k=self.top_k_passages)
                observation = self._format_observation(passages)
                full_content = f"{step_content}\n{observation}"

                new_steps.append({
                    'step_id': step_num,
                    'step_type': 'search',
                    'text': full_content,
                    **self._parse_step_fields(full_content),
                })
                accumulated_response += full_content + "\n\n"

            else:
                new_steps.append({
                    'step_id': step_num,
                    'step_type': 'reason',
                    'text': step_content,
                    **self._parse_step_fields(step_content),
                })
                accumulated_response += step_content + "\n\n"

        # Extract final answer
        final_answer = self._extract_final_answer(new_steps)
        is_correct = self._check_answer(final_answer, truncation_result.gold_answer)

        # Combine good steps + new steps
        all_steps = []
        for step in truncation_result.good_steps:
            all_steps.append({
                'step_id': step['step_id'],
                'step_type': step['step_type'],
                'think': step.get('think', ''),
                'search': step.get('search', ''),
                'documents': step.get('documents', ''),
                'answer': step.get('answer', ''),
                'source': 'original',
            })
        for step in new_steps:
            step['source'] = 'regenerated'
            all_steps.append(step)

        return {
            'trajectory_id': truncation_result.trajectory_id + '_regen',
            'question_id': truncation_result.question_id,
            'question': truncation_result.question,
            'gold_answer': truncation_result.gold_answer,
            'predicted_answer': final_answer,
            'is_correct': is_correct,
            'case': 'B',
            'steps': all_steps,
            'metadata': {
                'original_trajectory_id': truncation_result.trajectory_id,
                'truncated_at_step': truncation_result.bad_step_id,
                'num_good_steps': len(truncation_result.good_steps),
                'num_new_steps': len(new_steps),
                'num_total_steps': len(all_steps),
                'original_predicted_answer': truncation_result.original_predicted_answer,
                'critic_reasoning': truncation_result.critic_reasoning[:200],
            },
        }

    # === Helper methods ===

    def _parse_step_response(self, response: str, step_num: int) -> str:
        """Parse step content from response, removing future steps."""
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()].strip()

        step_prefix = re.match(rf'^Step\s+{step_num}:\s*', response)
        if step_prefix:
            response = response[step_prefix.end():]

        return response.strip()

    def _parse_action(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse action from step content (XML format)."""
        answer_match = re.search(r'<answer>(.+?)</answer>', content, re.DOTALL)
        if answer_match:
            return ('finish', answer_match.group(1).strip())

        search_match = re.search(r'<search>(.+?)</search>', content, re.DOTALL)
        if search_match:
            return ('search', search_match.group(1).strip())

        # Legacy format
        finish_match = re.search(
            r'Action:\s*Finish\[answer=["\']?(.+?)["\']?\]',
            content, re.IGNORECASE | re.DOTALL
        )
        if finish_match:
            return ('finish', finish_match.group(1).strip())

        search_match = re.search(
            r'Action:\s*Search\[query=["\']?(.+?)["\']?\]',
            content, re.IGNORECASE | re.DOTALL
        )
        if search_match:
            return ('search', search_match.group(1).strip())

        think_match = re.search(r'<think>(.+?)</think>', content, re.DOTALL)
        if think_match:
            return ('reason', think_match.group(1).strip())

        if any(marker in content.lower() for marker in
               ['final answer:', 'the answer is', 'therefore, the answer']):
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

    def _parse_step_fields(self, content: str) -> Dict[str, str]:
        """Parse XML fields from step content."""
        fields = {}
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            fields['think'] = think_match.group(1).strip()
        search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
        if search_match:
            fields['search'] = search_match.group(1).strip()
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            fields['answer'] = answer_match.group(1).strip()
        docs_match = re.search(r'<documents>(.*?)</documents>', content, re.DOTALL)
        if docs_match:
            fields['documents'] = docs_match.group(1).strip()
        return fields

    def _extract_final_answer(self, steps: List[Dict[str, Any]]) -> str:
        """Extract final answer from generated steps."""
        if not steps:
            return ""

        last_content = steps[-1].get('text', '')

        answer_match = re.search(r'<answer>(.+?)</answer>', last_content, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        if steps[-1].get('answer'):
            return steps[-1]['answer']

        finish_match = re.search(
            r'Finish\[answer=["\']?(.+?)["\']?\]',
            last_content, re.IGNORECASE | re.DOTALL
        )
        if finish_match:
            return finish_match.group(1).strip()

        final_match = re.search(r'final\s+answer:\s*(.+)', last_content, re.IGNORECASE)
        if final_match:
            return final_match.group(1).strip().split('\n')[0]

        answer_match = re.search(r'the\s+answer\s+is\s*:?\s*(.+)', last_content, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip().split('\n')[0]

        return ""

    def _check_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold (cover EM with quote normalization)."""
        if not predicted or not gold:
            return False
        pred_norm = predicted.strip().lower()
        gold_norm = gold.strip().lower()
        # Basic cover EM
        if pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm:
            return True
        # Strip quotes and retry (handles: Levon "Fred" Agabashian vs Fred Agabashian)
        pred_clean = pred_norm.replace('"', '').replace("'", '').replace('\u201c', '').replace('\u201d', '')
        gold_clean = gold_norm.replace('"', '').replace("'", '').replace('\u201c', '').replace('\u201d', '')
        return pred_clean == gold_clean or gold_clean in pred_clean or pred_clean in gold_clean
