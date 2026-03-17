"""LLM Judge labeler (VersaPRM style) using vLLM."""

import json
import re
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .base_labeler import BaseLabeler
from ..data.schemas import Trajectory, JudgeLabel


class JudgeLabeler(BaseLabeler):
    """LLM Judge labeler using large language model evaluation.

    Following VersaPRM style:
    - Uses large LLM (e.g., 70B+) to judge step quality
    - Provides gold answer and supporting facts as context
    - Evaluates each step's contribution to reaching correct answer
    - Returns GOOD/BAD label with reasoning

    Now uses vLLM backend for fast inference.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_client=None,
    ):
        """Initialize Judge labeler.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Name of judge model
                - temperature: Sampling temperature
                - max_tokens: Max tokens for generation
                - max_retries: Max retry attempts
                - retry_delay: Delay between retries
                - prompt_style: Prompt template style
                - use_gold_answer: Whether to provide gold answer
                - use_supporting_facts: Whether to provide supporting facts
                - gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.7)
                - tensor_parallel_size: Number of GPUs for vLLM (default: 1)
            model_client: Pre-configured model client (optional, uses vLLM PolicyModel if None)
        """
        super().__init__(config)

        self.model_name = config.get("model_name", "Qwen/QwQ-32B")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2048)  # Increased for QwQ-32B long reasoning
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.prompt_style = config.get("prompt_style", "versaprm")
        self.use_gold_answer = config.get("use_gold_answer", True)
        self.use_supporting_facts = config.get("use_supporting_facts", True)

        # vLLM-specific settings
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.8)
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.max_model_len = config.get("max_model_len", 32768)

        self.model_client = model_client

        # Initialize vLLM model if no client provided
        if self.model_client is None:
            self._initialize_vllm_model()

    def _initialize_vllm_model(self):
        """Initialize vLLM model for judge labeling."""
        from ..models import load_policy_model

        model_config = {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'tensor_parallel_size': self.tensor_parallel_size,
            'max_model_len': self.max_model_len,
        }

        print(f"Initializing Judge model with vLLM: {self.model_name}")
        self.model_client = load_policy_model(model_config)

    def label_trajectory(self, trajectory: Trajectory) -> List[JudgeLabel]:
        """Label a trajectory using LLM judge (batch mode - all steps at once).

        Args:
            trajectory: Trajectory to label

        Returns:
            List of JudgeLabel objects, one per step
        """
        # Build prompt for whole trajectory
        prompt = self._build_whole_trajectory_prompt(trajectory)

        # Call LLM once for all steps
        response = self._call_llm_with_retry(prompt)

        # Parse JSON response to get labels for all steps
        labels = self._parse_json_response(response, len(trajectory.steps))

        return labels

    def _parse_step_sections(self, step) -> dict:
        """Parse step content into XML tag sections.

        XML format:
        <think>reasoning</think>
        <search>query</search> or <answer>answer</answer>
        <documents>retrieved passages</documents>

        Returns:
            dict with: think, search, answer, documents, action_type
        """
        # Get full content from either 'text', 'content', or 'action' field
        full_content = getattr(step, 'text', None) or getattr(step, 'content', None) or getattr(step, 'action', '') or ''

        think = ''
        search = ''
        answer = ''
        documents = ''
        action_type = 'reason'

        # Extract XML tags
        think_match = re.search(r'<think>(.*?)</think>', full_content, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()

        search_match = re.search(r'<search>(.*?)</search>', full_content, re.DOTALL)
        if search_match:
            search = search_match.group(1).strip()
            action_type = 'search'

        answer_match = re.search(r'<answer>(.*?)</answer>', full_content, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            action_type = 'answer'

        docs_match = re.search(r'<documents>(.*?)</documents>', full_content, re.DOTALL)
        if docs_match:
            documents = docs_match.group(1).strip()

        # Fallback to step attributes if XML not found
        if not think:
            think = getattr(step, 'think', '') or getattr(step, 'thought', '') or ''
        if not documents:
            documents = getattr(step, 'observation', '') or getattr(step, 'documents', '') or ''

        return {
            'think': think,
            'search': search,
            'answer': answer,
            'documents': documents,
            'action_type': action_type,
        }

    def _build_whole_trajectory_prompt(self, trajectory: Trajectory) -> str:
        """Build prompt for evaluating all steps in one call.

        Strict Process Supervision: rigorously evaluate search strategy and evidence-based reasoning.
        """
        # 1. Build full trajectory (XML tag format)
        interaction_history = []
        for i, step in enumerate(trajectory.steps, 1):
            sections = self._parse_step_sections(step)

            step_text = f"## Step {i}\n"

            # Think (reasoning)
            if sections['think']:
                step_text += f"<think>{sections['think']}</think>\n"

            # Action (search or answer)
            if sections['action_type'] == 'search' and sections['search']:
                step_text += f"<search>{sections['search']}</search>\n"
            elif sections['action_type'] == 'answer' and sections['answer']:
                step_text += f"<answer>{sections['answer']}</answer>\n"

            # Documents (retrieved passages)
            if sections['documents']:
                step_text += f"<documents>{sections['documents']}</documents>\n"

            interaction_history.append(step_text)

        history_str = "\n\n".join(interaction_history)
        gold_answer = trajectory.gold_answer if self.use_gold_answer else "N/A"

        # Show final answer clearly
        final_answer_section = ""
        if trajectory.final_answer:
            final_answer_section = f"## Model's Final Answer\n{trajectory.final_answer}"

        # 2. Strict Process Supervisor prompt (XML tag format)
        prompt = f"""You are a strict process supervisor for a multi-hop question answering agent that uses retrieval-augmented generation (RAG). Your task is to evaluate each step of the agent's trajectory and assign a binary label: GOOD or BAD.

The agent operates using four XML-tagged actions:
- <think>...</think>  Internal reasoning
- <search>...</search>  Retrieval query
- <documents>...</documents>  Retrieved passages (system-provided)
- <answer>...</answer>  Final answer submission

---

# Labeling Principles

1. The default label is BAD. Assign GOOD only when a step makes a clear, verifiable contribution toward answering the question.
2. No information gain means BAD. Steps that restate the question, repeat prior reasoning, or produce generic plans without new insight are BAD.
3. Do not use your own world knowledge. Any factual claim not grounded in <documents> is treated as hallucination.

---

# Evaluation Criteria

**R1. Entity and Relation Grounding**
- BAD if <think> misidentifies an entity (e.g., fictional vs. real person), reverses a relation (e.g., son vs. father), or drifts to a namesake.
- GOOD if the step maintains correct entity and relation grounding consistent with the question.

**R2. Search Steps (<search>)**
- GOOD if: (a) the query is specific and grounded in the question's entities and relations, (b) the returned <documents> contain information directly relevant to the question, and (c) if a prior search failed, the agent uses a meaningfully different strategy.
- BAD if: the query repeats a failed attempt, is too vague, targets the wrong entity, is derived from a hallucinated claim, or the returned <documents> are empty or irrelevant.

**R3. Reasoning Steps (<think> only)**
- GOOD if the step extracts new useful information from prior <documents> that advances toward the answer, or identifies a specific knowledge gap with a concrete next search plan.
- BAD if the step merely restates the question, repeats prior reasoning, makes a generic plan, summarizes without new insight, or asserts unsupported factual claims.

**R4. Answer Steps (<answer> or final step)**
This rule applies to any step with <answer> and to the last step of the trajectory.
- BAD if any of the following hold:
  (a) The answer is empty, uncertain, or a non-response (e.g., "Unknown", "N/A").
  (b) The reasoning expresses uncertainty (e.g., "cannot determine", "not sure").
  (c) The answer requires a logical leap not supported by <documents>.
  (d) The agent concludes despite insufficient evidence (premature termination).
  (e) The answer contradicts information in <documents>.
- GOOD if and only if: the answer is specific, logically derivable from the accumulated <documents>, and the reasoning chain is sound.

**R5. Recovery from Retrieval Failure**
When the preceding <documents> were irrelevant or empty:
- GOOD if the agent issues a new <search> with a meaningfully different query.
- BAD if the agent proceeds to <answer> as if the search succeeded, or retries with the same query.

**R6. Unsupported Answer (Overconfidence)**
If the agent produces <answer> without any prior successful search:
- BAD. Correct answers without evidence are penalized as lucky guesses.

---

# Input

**Question:** {trajectory.question}
**Ground Truth Answer (for reference only - do NOT use this to directly judge correctness):** {gold_answer}

## Trajectory
{history_str}
{final_answer_section}

---

# Output Format

Return a JSON array with one entry per step. Each entry must have "step" (integer), "label" ("GOOD" or "BAD"), and "reasoning" (string).

```json
[
  {{"step": 1, "label": "GOOD", "reasoning": "..."}},
  {{"step": 2, "label": "BAD", "reasoning": "..."}}
]
```

Evaluate all {len(trajectory.steps)} steps."""

        return prompt

    def _parse_json_response(self, response: str, num_steps: int) -> List[JudgeLabel]:
        """Parse JSON response from whole-trajectory evaluation.

        Args:
            response: Raw LLM response containing JSON
            num_steps: Expected number of steps

        Returns:
            List of JudgeLabel objects
        """
        labels = []

        # Extract JSON from response (handle ```json ... ``` blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON array
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = None

        parsed_results = []
        if json_str:
            try:
                parsed_results = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common issues
                try:
                    # Remove trailing commas
                    fixed = re.sub(r',\s*]', ']', json_str)
                    fixed = re.sub(r',\s*}', '}', fixed)
                    parsed_results = json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        # Build labels from parsed results
        for step_idx in range(num_steps):
            # Find matching result
            result = None
            for r in parsed_results:
                if r.get('step') == step_idx + 1:
                    result = r
                    break

            if result:
                raw = str(result.get('label', '')).strip().upper()
                if raw == 'GOOD':
                    label = 'GOOD'
                else:
                    # "NOT GOOD", "BAD", "NEUTRAL", etc. → all BAD
                    label = 'BAD'
                reasoning = result.get('reasoning', '')  # No truncation
            else:
                # Default if parsing failed
                label = 'BAD'
                reasoning = 'Failed to parse response'

            labels.append(JudgeLabel(
                step_id=step_idx,
                label=label,
                reasoning=reasoning,
                metadata={
                    "model_name": self.model_name,
                    "prompt_style": "whole_trajectory",
                },
            ))

        return labels

    def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with retry logic.

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt)
                return response
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"All retries failed: {e}")
                    # Return a default response on failure (default BAD)
                    return """```json
[{"step": 1, "label": "BAD", "reasoning": "Unable to evaluate due to model error."}]
```"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM using vLLM backend.

        Args:
            prompt: Input prompt

        Returns:
            LLM response
        """
        if self.model_client is None:
            raise RuntimeError("Model client not initialized")

        # Format Judge prompt with chat template
        # QwQ-32B is a chat model and needs proper formatting
        formatted_prompt = self._format_judge_prompt_for_chat(prompt)

        # Use raw generate() with formatted chat template
        response = self.model_client.generate(
            prompt=formatted_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response

    def _format_judge_prompt_for_chat(self, user_message: str) -> str:
        """Format Judge prompt for chat model (e.g., QwQ-32B).

        Args:
            user_message: VersaPRM-style judge prompt

        Returns:
            Formatted prompt with chat template
        """
        if hasattr(self.model_client, 'tokenizer') and hasattr(self.model_client.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": user_message}
            ]
            return self.model_client.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to manual Qwen format
            return f"""<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

    def label_batch(
        self,
        trajectories: List[Trajectory],
        show_progress: bool = True,
    ) -> List[List[JudgeLabel]]:
        """Label a batch of trajectories using vLLM batch processing.

        This method batches all trajectory prompts and processes them in a single
        vLLM call for much faster inference.

        Args:
            trajectories: List of trajectories
            show_progress: Whether to show progress bar

        Returns:
            List of label lists
        """
        if not trajectories:
            return []

        # Build prompts for all trajectories
        prompts = []
        num_steps_list = []
        for traj in trajectories:
            prompt = self._build_whole_trajectory_prompt(traj)
            formatted = self._format_judge_prompt_for_chat(prompt)
            prompts.append(formatted)
            num_steps_list.append(len(traj.steps))

        if show_progress:
            print(f"  Judge labeling {len(trajectories)} trajectories in batch...")

        # Batch generate with vLLM
        if hasattr(self.model_client, 'batch_generate'):
            responses = self.model_client.batch_generate(
                prompts=prompts,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        else:
            # Fallback to sequential if batch not available
            responses = []
            iterator = tqdm(prompts, desc="Judge labeling") if show_progress else prompts
            for prompt in iterator:
                response = self.model_client.generate(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                responses.append(response)

        # Parse all responses
        all_labels = []
        for response, num_steps in zip(responses, num_steps_list):
            labels = self._parse_json_response(response, num_steps)
            all_labels.append(labels)

        if show_progress:
            print(f"  ✓ Judge labeling complete")

        return all_labels


def create_judge_labeler(config: Dict[str, Any]) -> JudgeLabeler:
    """Create JudgeLabeler with vLLM backend.

    Args:
        config: Configuration with model settings

    Returns:
        JudgeLabeler instance with vLLM model
    """
    return JudgeLabeler(config)