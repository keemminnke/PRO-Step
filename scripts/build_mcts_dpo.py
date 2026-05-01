#!/usr/bin/env python3
"""Build MCTS-based DPO dataset — Critic-guided MCTS (no simulation).

MCTS: UCB selection → Expansion → Retrieval → Backpropagation.
No simulation to terminal — Critic PRM replaces rollout value estimation.
  - ReasonRAG uses same policy model + gold answer for evaluation
  - We use separately trained Critic PRM (no gold answer) — our contribution

Reward: F1(pred, gold) * β^depth for terminal, critic score for non-terminal.
DPO pairs: sibling nodes with combined reward difference > threshold.

Usage:
    python scripts/build_mcts_dpo.py \
        --input outputs/all_5000q_critic_v9_filtered.jsonl \
        --output outputs/mcts_dpo_5000q.jsonl \
        --critic-model outputs/critic_model_v9_3000q/final_model
"""

import sys
import os
import json
import re
import gc
import math
import string
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

SYSTEM_PROMPT = """You are a helpful assistant who is good at answering questions with multi-turn search engine calling. To answer questions, you must first reason through the available information using <think> and </think>. If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with relevant documents enclosed in <documents> and </documents>. You can search as many times as you want. Once you have sufficient information or if you find no further external knowledge is needed, directly provide your final answer. Ensure your answer is concise, using nouns or short phrases whenever possible. Conclude with: "So the answer is <answer>answer</answer>"."""


# ─────────────────────────────────────────────────────────────────────────────
# F1 scoring utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(c for c in t if c not in string.punctuation)
    def lower(t): return t.lower()
    return white_space_fix(remove_punc(lower(s))).strip()


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred, gt = normalize_answer(prediction), normalize_answer(ground_truth)
    if pred in ["yes", "no", "noanswer"] and pred != gt:
        return 0.0
    if gt in ["yes", "no", "noanswer"] and pred != gt:
        return 0.0
    pred_tokens, gt_tokens = pred.split(), gt.split()
    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(gt_tokens) if gt_tokens else 0.0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BGE Retriever (transformers + FAISS, GPU-accelerated encoding)
# ─────────────────────────────────────────────────────────────────────────────

class BGERetriever:
    """BGE retriever using transformers + FAISS. Encoder on GPU for speed (~420MB)."""

    def __init__(self, model_path, index_path, corpus_path, top_k=3, device="cuda:1"):
        import faiss
        import numpy as np
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.top_k = top_k
        self._torch = torch
        self._np = np
        self._device = device

        print(f"[Retriever] Loading BGE model: {model_path} → {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_CACHE)
        self.model = AutoModel.from_pretrained(
            model_path, cache_dir=HF_CACHE, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

        # Load FAISS index — auto-build IVF for faster search if Flat index provided
        print(f"[Retriever] Loading FAISS index: {index_path}")
        ivf_path = index_path.replace(".index", "_IVF4096.index")
        if os.path.exists(ivf_path):
            print(f"  Using cached IVF index: {ivf_path}")
            self.index = faiss.read_index(ivf_path)
        elif "Flat" in index_path:
            print(f"  Building IVF index from Flat for faster search...")
            flat_index = faiss.read_index(index_path)
            d = flat_index.d
            ntotal = flat_index.ntotal
            metric = flat_index.metric_type
            print(f"  Flat index: {ntotal:,} vectors × {d} dim")

            # Reconstruct all vectors
            print(f"  Reconstructing vectors from Flat index...")
            xb = flat_index.reconstruct_n(0, ntotal).astype("float32")

            # Build IVF index (nlist=4096 → search probes ~3% of clusters)
            nlist = 4096
            if metric == faiss.METRIC_INNER_PRODUCT:
                quantizer = faiss.IndexFlatIP(d)
            else:
                quantizer = faiss.IndexFlatL2(d)
            ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)

            # Train on subset (fast)
            train_size = min(200_000, ntotal)
            print(f"  Training IVF (nlist={nlist}) on {train_size:,} vectors...")
            ivf_index.train(xb[:train_size])

            # Add all vectors
            print(f"  Adding {ntotal:,} vectors to IVF index...")
            batch = 500_000
            for start in range(0, ntotal, batch):
                end = min(start + batch, ntotal)
                ivf_index.add(xb[start:end])
                if end < ntotal:
                    print(f"    {end:,}/{ntotal:,}")

            # Save for future runs
            print(f"  Saving IVF index → {ivf_path}")
            faiss.write_index(ivf_index, ivf_path)
            self.index = ivf_index
            del flat_index, xb
        else:
            self.index = faiss.read_index(index_path)

        # Set nprobe for IVF (128/4096 ≈ 3% → ~97% recall, ~30-50x faster than Flat)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 128
            print(f"  IVF nprobe: {self.index.nprobe}")
        print(f"  Index size: {self.index.ntotal:,} vectors")

        print(f"[Retriever] Loading corpus: {corpus_path}")
        self.corpus = []
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    self.corpus.append(json.loads(line))
        print(f"  Corpus size: {len(self.corpus):,} documents")

    def _encode(self, texts: List[str], batch_size=512):
        import torch.nn.functional as F
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with self._torch.no_grad():
                outputs = self.model(**encoded)
            # Mean pooling (matches FlashRAG index_builder --pooling_method mean)
            mask = encoded["attention_mask"]
            hidden = outputs.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
            emb = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.float().cpu().numpy())
        return self._np.vstack(all_embs)

    def search(self, queries: List[str], top_k: int = None) -> List[List[Dict]]:
        if not queries:
            return []
        top_k = top_k or self.top_k
        embeddings = self._encode(queries)  # BGE: no prefix needed
        scores, indices = self.index.search(embeddings.astype("float32"), top_k)
        results = []
        for idx_list in indices:
            docs = []
            for i in idx_list:
                if 0 <= i < len(self.corpus):
                    docs.append(self.corpus[i])
            results.append(docs)
        return results


def format_docs(docs: List[Dict]) -> str:
    """Format retrieved docs like SearchR1Pipeline."""
    parts = []
    for idx, doc in enumerate(docs):
        contents = doc.get("contents", "")
        title = contents.split("\n")[0]
        text = "\n".join(contents.split("\n")[1:])
        parts.append(f"Doc {idx+1}(Title: {title}) {text}")
    return f"\n\n<documents>\n" + "\n".join(parts) + "\n</documents>\n\n"


def parse_step(text: str) -> Dict:
    """Parse generated text into step components."""
    full = "<think>" + text
    step = {"think": "", "search_query": "", "answer_text": ""}

    think_m = re.search(r"<think>(.*?)(?:</think>|$)", full, re.DOTALL)
    if think_m:
        step["think"] = think_m.group(1).strip()

    search_m = re.search(r"<search>(.*?)(?:</search>|$)", full, re.DOTALL)
    answer_m = re.search(r"<answer>(.*?)(?:</answer>|$)", full, re.DOTALL)

    if search_m:
        step["search_query"] = search_m.group(1).strip()
    elif answer_m:
        step["answer_text"] = answer_m.group(1).strip()

    return step


# ─────────────────────────────────────────────────────────────────────────────
# MCTS Tree (per-question)
# ─────────────────────────────────────────────────────────────────────────────

class MCTSTree:
    """MCTS tree for a single question."""

    def __init__(self, qid, question, gold_answer, max_children, max_depth):
        self.qid = qid
        self.question = question
        self.gold_answer = gold_answer
        self.max_children = max_children
        self.max_depth = max_depth
        self.nodes = {}
        self._next_id = 0

        self._expandable_count = 0  # non-terminal nodes with < max_children

        # Root node (depth 0)
        root = self._new_node(None, 0)
        self.root_id = root["id"]
        self._expandable_count = 1  # root is expandable

    def _new_node(self, parent_id, depth):
        n = {
            "id": self._next_id,
            "parent_id": parent_id,
            "children_ids": [],
            "depth": depth,
            "think": "",
            "search_query": "",
            "answer_text": "",
            "documents": "",
            "is_terminal": False,
            # MCTS stats (for UCB selection during tree building)
            "visit_count": 0,
            "total_value": 0.0,
            # Final reward (for DPO extraction, computed post-MCTS)
            "f1_reward": None,
            # Critic PRM scores (inline during MCTS or post-scoring)
            "critic_score": None,   # binary 0/1, backprop uses score * β^depth
            "critic_feedback": "",  # diagnostic reasoning text
        }
        self.nodes[self._next_id] = n
        self._next_id += 1
        if parent_id is not None:
            self.nodes[parent_id]["children_ids"].append(n["id"])
        return n

    def select(self, c=1.414):
        """UCB1 selection from root to expandable/terminal leaf."""
        nid = self.root_id
        while True:
            n = self.nodes[nid]
            if n["is_terminal"]:
                return nid
            # Not fully expanded → expand here
            if len(n["children_ids"]) < self.max_children:
                return nid
            # Fully expanded → UCB among children
            best, best_ucb = None, -float("inf")
            pv = max(n["visit_count"], 1)
            for cid in n["children_ids"]:
                ch = self.nodes[cid]
                if ch["visit_count"] == 0:
                    return cid  # Unvisited → explore first
                q = ch["total_value"] / ch["visit_count"]
                ucb = q + c * math.sqrt(math.log(pv) / ch["visit_count"])
                if ucb > best_ucb:
                    best_ucb, best = ucb, cid
            if best is None:
                return nid
            nid = best

    def expand(self, parent_id, step):
        """Add child node from parsed step dict. Returns the new child or None if full."""
        parent = self.nodes[parent_id]
        if len(parent["children_ids"]) >= self.max_children:
            return None
        child = self._new_node(parent_id, parent["depth"] + 1)
        child["think"] = step.get("think", "")
        child["search_query"] = step.get("search_query", "")
        child["answer_text"] = step.get("answer_text", "")
        child["is_terminal"] = bool(child["answer_text"]) or child["depth"] >= self.max_depth

        # Update expandable count
        if len(parent["children_ids"]) >= self.max_children:
            self._expandable_count -= 1  # parent no longer expandable
        if not child["is_terminal"]:
            self._expandable_count += 1  # new non-terminal child is expandable
        return child

    def backprop(self, nid, value):
        """Update visit count and value from node to root."""
        cur = nid
        while cur is not None:
            self.nodes[cur]["visit_count"] += 1
            self.nodes[cur]["total_value"] += value
            cur = self.nodes[cur]["parent_id"]

    def retroactive_update(self, nid, value_delta):
        """Add value correction from node to root WITHOUT changing visit counts.

        Used when critic scores arrive after temporary 0.0 backprop.
        N stays the same, W += delta → Q = W/N gets corrected.
        """
        cur = nid
        while cur is not None:
            self.nodes[cur]["total_value"] += value_delta
            cur = self.nodes[cur]["parent_id"]

    def get_path(self, nid):
        """Path from root to node (inclusive, root first)."""
        path = []
        cur = nid
        while cur is not None:
            path.append(cur)
            cur = self.nodes[cur]["parent_id"]
        path.reverse()
        return path

    def build_prefix(self, nid):
        """Build assistant prefix text from root's children to this node."""
        path = self.get_path(nid)
        parts = []
        for pid in path[1:]:  # skip root
            n = self.nodes[pid]
            if n["think"]:
                parts.append(f"<think>{n['think']}</think>")
            if n["search_query"]:
                parts.append(f"<search>{n['search_query']}</search>")
            elif n["answer_text"]:
                parts.append(f"<answer>{n['answer_text']}</answer>")
            if n["documents"]:
                parts.append(n["documents"])
        return "\n".join(parts)

    @property
    def is_fully_explored(self):
        """True if no more expansion possible (all leaves terminal or max-expanded)."""
        return self._expandable_count <= 0

    def to_dicts(self):
        """Serialize all nodes for caching."""
        result = []
        for n in self.nodes.values():
            entry = dict(n)
            entry["question_id"] = self.qid
            entry["question"] = self.question
            entry["gold_answer"] = self.gold_answer
            result.append(entry)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Critic PRM helpers (reused by inline MCTS scoring + Phase 3 post-scoring)
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = (
    "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
    "The trajectory uses XML tags: <think> for reasoning, <search> for queries, <answer> for final answers, "
    "<documents> for retrieved passages. "
    "Analyze each step's logical soundness and evidence grounding. "
    "First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."
)


def build_critic_prompt(tree, node, tokenizer):
    """Build critic evaluation prompt for a single node (reusable)."""
    path = tree.get_path(node["id"])
    input_parts = [f"Question: {tree.question}", f"Gold Answer: {tree.gold_answer}", ""]

    # Previous steps
    prev_ids = path[1:-1]  # exclude root and current
    if prev_ids:
        input_parts.append("Previous Steps:")
        for j, pid in enumerate(prev_ids):
            prev = tree.nodes[pid]
            input_parts.append(f"## Step {j+1}")
            if prev["think"]:
                input_parts.append(f"<think>{prev['think']}</think>")
            if prev["search_query"]:
                input_parts.append(f"<search>{prev['search_query']}</search>")
            elif prev["answer_text"]:
                input_parts.append(f"<answer>{prev['answer_text']}</answer>")
            if prev["documents"]:
                doc_text = prev["documents"].strip()
                doc_text = re.sub(r"^\s*<documents>\s*", "", doc_text)
                doc_text = re.sub(r"\s*</documents>\s*$", "", doc_text)
                input_parts.append(f"<documents>{doc_text}</documents>")
            input_parts.append("")

    # Current step
    input_parts.append("Current Step to Evaluate:")
    input_parts.append(f"## Step {len(prev_ids) + 1}")
    if node["think"]:
        input_parts.append(f"<think>{node['think']}</think>")
    if node["search_query"]:
        input_parts.append(f"<search>{node['search_query']}</search>")
    elif node["answer_text"]:
        input_parts.append(f"<answer>{node['answer_text']}</answer>")
    if node["documents"]:
        doc_text = node["documents"].strip()
        doc_text = re.sub(r"^\s*<documents>\s*", "", doc_text)
        doc_text = re.sub(r"\s*</documents>\s*$", "", doc_text)
        input_parts.append(f"<documents>{doc_text}</documents>")
    input_parts.append("")
    input_parts.append("Task: Evaluate the quality of the Current Step. "
                       "Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(input_parts)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_critic_output(text):
    """Parse critic output text into (score, feedback).

    Returns:
        (int, str): (binary score 0/1, diagnostic reasoning text)
    """
    text = text.strip()

    # Extract reasoning feedback
    reasoning_match = re.search(
        r"\[REASONING\](.*?)(?:\[/REASONING\]|$)", text, re.DOTALL
    )
    feedback = reasoning_match.group(1).strip() if reasoning_match else text

    # Binary label from text
    score = 1 if "1" in text else 0
    return score, feedback


# ─────────────────────────────────────────────────────────────────────────────
# MCTS Runner (batched across questions with vLLM)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(tree, node_id, tokenizer):
    """Build vLLM prompt string for generating next step from node."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {tree.question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix = tree.build_prefix(node_id)
    if prefix:
        prompt += prefix + "\n<think>"
    else:
        prompt += "<think>"
    return prompt


def run_mcts(questions, retriever, args):
    """Critic-guided MCTS: UCB selection → Expansion → Retrieval → Critic → Backprop.

    Inline critic scoring: non-terminal nodes scored immediately after expansion.
    Terminal nodes get F1*β^depth reward, non-terminal get critic score for UCB guidance.
    Without --critic-model, falls back to blind mode (backprop 0.0 for non-terminal).
    """
    from vllm import LLM, SamplingParams
    import torch

    # Determine GPU memory split
    has_critic = args.critic_model is not None
    if has_critic:
        policy_gpu_mem = 0.45
        critic_gpu_mem = 0.30
        # 합계 0.75 per GPU — leave headroom for BGE retriever + fragmentation
        print(f"\n[MCTS] Dual-model mode: policy={policy_gpu_mem}, critic={critic_gpu_mem}")
    else:
        policy_gpu_mem = args.gpu_memory
        print(f"\n[MCTS] Policy-only mode (blind): gpu_memory={policy_gpu_mem}")

    # Load policy model
    print(f"[MCTS] Loading policy: {args.policy_model}")
    llm = LLM(
        model=args.policy_model,
        tensor_parallel_size=2,
        gpu_memory_utilization=policy_gpu_mem,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
        seed=42,
    )
    tokenizer = llm.get_tokenizer()

    # Load critic model for inline scoring (AlphaZero-style value estimation)
    critic_llm = None
    lora_request = None
    critic_sampling = None
    critic_tokenizer = None
    if has_critic:
        from vllm.lora.request import LoRARequest
        print(f"[MCTS] Loading critic: {args.critic_base} + LoRA")
        try:
            critic_llm = LLM(
                model=args.critic_base,
                tensor_parallel_size=2,
                enable_lora=True,
                max_lora_rank=64,
                gpu_memory_utilization=critic_gpu_mem,
                max_model_len=16384,
                trust_remote_code=True,
                download_dir=HF_CACHE,
            )
            critic_tokenizer = critic_llm.get_tokenizer()
            critic_model_abs = os.path.abspath(args.critic_model)
            lora_request = LoRARequest("critic", 1, critic_model_abs)
            critic_sampling = SamplingParams(temperature=0.0, max_tokens=512)
            print(f"[MCTS] Both models loaded (policy + critic)")
        except Exception as e:
            print(f"[MCTS] Critic loading failed: {e}")
            print(f"[MCTS] Falling back to policy-only mode (blind MCTS)")
            critic_llm = None

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
        n=1,  # MCTS: one child per expansion
    )

    # Check for checkpoint to resume
    start_rollout = 0
    ckpt_rollout = load_checkpoint(args.tree_cache)
    if ckpt_rollout is not None and ckpt_rollout > 0:
        print(f"\n[Checkpoint] Found checkpoint at rollout {ckpt_rollout}, resuming...")
        trees = load_trees(args.tree_cache, args.max_children, args.max_depth)
        start_rollout = ckpt_rollout
    else:
        # Create per-question trees from scratch
        trees = {}
        for q in questions:
            trees[q["question_id"]] = MCTSTree(
                q["question_id"], q["question"], q["gold_answer"],
                max_children=args.max_children, max_depth=args.max_depth,
            )

    print(f"[MCTS] {len(trees):,} trees, {args.max_rollouts} rollouts, "
          f"K={args.max_children}, depth={args.max_depth}")
    if start_rollout > 0:
        print(f"[MCTS] Resuming from rollout {start_rollout + 1}")

    pending_nonterminal = []  # accumulated unscored nodes between critic rounds

    pbar = tqdm(range(start_rollout, args.max_rollouts), desc="MCTS rollouts",
                unit="rollout", initial=start_rollout, total=args.max_rollouts)
    for rollout in pbar:
        # ── Selection: pick one leaf per tree to expand ──
        to_expand = {}  # qid -> node_id
        for qid, tree in trees.items():
            if tree.is_fully_explored:
                continue
            leaf_id = tree.select()
            leaf = tree.nodes[leaf_id]
            if leaf["is_terminal"] or len(leaf["children_ids"]) >= tree.max_children:
                continue
            to_expand[qid] = leaf_id

        if not to_expand:
            pbar.write(f"[MCTS] Rollout {rollout+1}: all trees fully explored, stopping")
            break

        # ── Expansion: generate 1 child per selected node ──
        prompts, prompt_qids = [], []
        for qid, nid in to_expand.items():
            prompt = build_prompt(trees[qid], nid, tokenizer)
            prompts.append(prompt)
            prompt_qids.append(qid)

        outputs = llm.generate(prompts, sampling)

        # Parse and create children
        search_needed = []
        new_nonterminal = []  # (qid, child_id) for inline critic scoring

        for qid, output in zip(prompt_qids, outputs):
            tree = trees[qid]
            parent_id = to_expand[qid]
            step = parse_step(output.outputs[0].text)

            if not step["think"] and not step["search_query"] and not step["answer_text"]:
                tree.backprop(parent_id, 0.0)
                continue

            child = tree.expand(parent_id, step)
            if child is None:
                tree.backprop(parent_id, 0.0)
                continue

            if child["is_terminal"]:
                if child["answer_text"]:
                    f1 = compute_f1(child["answer_text"], tree.gold_answer)
                    reward = f1 * (args.beta ** child["depth"])
                    child["f1_reward"] = reward
                    tree.backprop(child["id"], reward)
                else:
                    child["f1_reward"] = 0.0
                    tree.backprop(child["id"], 0.0)
            else:
                # Non-terminal: collect for retrieval + inline critic scoring
                if child["search_query"]:
                    search_needed.append((qid, child["id"]))
                new_nonterminal.append((qid, child["id"]))

        # ── Retrieval for search nodes ──
        if search_needed:
            queries = [trees[qid].nodes[nid]["search_query"]
                       for qid, nid in search_needed]
            docs_list = retriever.search(queries, top_k=args.top_k)
            for (qid, nid), docs in zip(search_needed, docs_list):
                trees[qid].nodes[nid]["documents"] = format_docs(docs)

        # ── Temporary backprop 0.0 for non-terminal (N 올리되 W=0) ──
        for qid, nid in new_nonterminal:
            trees[qid].backprop(nid, 0.0)
        pending_nonterminal.extend(new_nonterminal)

        # ── Periodic Critic Scoring → Retroactive Update ──
        is_critic_round = (
            critic_llm is not None
            and pending_nonterminal
            and ((rollout + 1) % args.critic_interval == 0
                 or rollout == args.max_rollouts - 1)
        )
        if is_critic_round:
            c_prompts = []
            for qid, nid in pending_nonterminal:
                cp = build_critic_prompt(
                    trees[qid], trees[qid].nodes[nid], critic_tokenizer
                )
                c_prompts.append(cp)
            c_outputs = critic_llm.generate(
                c_prompts, critic_sampling, lora_request=lora_request
            )
            for (qid, nid), cout in zip(pending_nonterminal, c_outputs):
                score, feedback = parse_critic_output(cout.outputs[0].text)
                node = trees[qid].nodes[nid]
                node["critic_score"] = score
                node["critic_feedback"] = feedback
                # Retroactive: N은 유지, W에 진짜 점수 보정 (0→discounted)
                discounted = float(score) * (args.beta ** node["depth"])
                trees[qid].retroactive_update(nid, discounted)
            pbar.write(f"  [Critic] Scored {len(pending_nonterminal):,} nodes "
                       f"(rollouts {rollout+2-args.critic_interval}-{rollout+1})")
            pending_nonterminal = []

        # ── Progress ──
        total_nodes = sum(len(t.nodes) for t in trees.values())
        terminal = sum(1 for t in trees.values() for n in t.nodes.values() if n["is_terminal"])
        scored = sum(1 for t in trees.values() for n in t.nodes.values()
                     if n.get("critic_score") is not None)
        pbar.set_postfix(
            active=len(to_expand),
            nodes=f"{total_nodes:,}",
            terminal=terminal,
            scored=scored,
        )

        # ── Checkpoint: save every 5 rollouts (or final) ──
        if (rollout + 1) % 5 == 0 or rollout == args.max_rollouts - 1:
            save_checkpoint(trees, args.tree_cache, rollout + 1)

    # Free both models
    del llm
    if critic_llm is not None:
        del critic_llm
    gc.collect()
    torch.cuda.empty_cache()

    # Final stats
    total_nodes = sum(len(t.nodes) for t in trees.values())
    terminal_nodes = sum(
        1 for t in trees.values() for n in t.nodes.values() if n["is_terminal"]
    )
    print(f"\n[MCTS] Done — {total_nodes:,} total nodes, "
          f"{terminal_nodes:,} terminal, avg {total_nodes/len(trees):.1f}/q")

    return trees


# ─────────────────────────────────────────────────────────────────────────────
# Post-MCTS: F1 Scoring + Bottom-up Propagation
# ─────────────────────────────────────────────────────────────────────────────

def score_and_propagate(trees, beta):
    """Score terminal nodes with F1 * β^depth, propagate bottom-up."""
    leaf_count = 0

    for tree in trees.values():
        # Score terminal nodes
        for n in tree.nodes.values():
            if n["f1_reward"] is not None:
                continue  # already scored during MCTS
            if n["answer_text"]:
                f1 = compute_f1(n["answer_text"], tree.gold_answer)
                n["f1_reward"] = f1 * (beta ** n["depth"])
                leaf_count += 1
            elif n["is_terminal"]:
                n["f1_reward"] = 0.0

        # Bottom-up propagation (N-weighted mean, like ReasonRAG paper)
        max_d = max(n["depth"] for n in tree.nodes.values())
        for d in range(max_d, -1, -1):
            for n in tree.nodes.values():
                if n["depth"] != d or n["f1_reward"] is not None:
                    continue
                children = [
                    tree.nodes[cid] for cid in n["children_ids"]
                    if tree.nodes[cid]["f1_reward"] is not None
                ]
                if children:
                    total_n = sum(c["visit_count"] for c in children)
                    if total_n > 0:
                        n["f1_reward"] = sum(
                            c["f1_reward"] * c["visit_count"] for c in children
                        ) / total_n
                    else:
                        n["f1_reward"] = sum(
                            c["f1_reward"] for c in children
                        ) / len(children)
                else:
                    n["f1_reward"] = 0.0

    # Stats
    all_rewards = [
        n["f1_reward"]
        for t in trees.values()
        for n in t.nodes.values()
        if n["depth"] > 0 and n["f1_reward"] is not None
    ]
    if all_rewards:
        import statistics
        print(f"\n[Score] {len(all_rewards):,} nodes scored — "
              f"mean: {statistics.mean(all_rewards):.3f}, "
              f"median: {statistics.median(all_rewards):.3f}, "
              f"max: {max(all_rewards):.3f}")
    if leaf_count:
        print(f"  (additional {leaf_count:,} leaves scored post-MCTS)")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Critic PRM Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_with_critic(trees, critic_model, critic_base, gpu_memory):
    """Score non-root steps with critic PRM (1=good, 0=bad).

    Skips nodes already scored inline during MCTS.
    Saves both binary score AND diagnostic feedback text.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch

    # Collect only unscored non-root nodes
    step_nodes = []  # (tree, node)
    already_scored = 0
    for tree in trees.values():
        for n in tree.nodes.values():
            if n["depth"] > 0:
                if n.get("critic_score") is not None:
                    already_scored += 1
                else:
                    step_nodes.append((tree, n))

    if already_scored > 0:
        print(f"\n[Critic] {already_scored:,} nodes already scored inline, skipping")
    if not step_nodes:
        print(f"[Critic] All nodes already scored, nothing to do")
        return

    print(f"[Critic] Scoring {len(step_nodes):,} remaining steps...")

    critic_llm = LLM(
        model=critic_base,
        tensor_parallel_size=2,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=gpu_memory,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
    )

    critic_model_abs = os.path.abspath(critic_model)
    lora_request = LoRARequest("critic", 1, critic_model_abs)
    critic_sampling = SamplingParams(temperature=0.0, max_tokens=512)
    critic_tok = critic_llm.get_tokenizer()

    # Build prompts using shared helper
    prompts = []
    for tree, n in step_nodes:
        prompts.append(build_critic_prompt(tree, n, critic_tok))

    # Batch inference
    print(f"[Critic] Running inference on {len(prompts):,} prompts...")
    outputs = critic_llm.generate(prompts, critic_sampling, lora_request=lora_request)

    for (tree, n), out in zip(step_nodes, outputs):
        score, feedback = parse_critic_output(out.outputs[0].text)
        n["critic_score"] = score
        n["critic_feedback"] = feedback

    # Stats (all nodes including inline-scored)
    all_scores = []
    for tree in trees.values():
        for n in tree.nodes.values():
            if n["depth"] > 0 and n.get("critic_score") is not None:
                all_scores.append(n["critic_score"])
    total = len(all_scores)
    good = sum(all_scores)
    print(f"[Critic] Total: {total:,} — GOOD: {good:,} ({100*good/total:.1f}%), "
          f"BAD: {total-good:,} ({100*(total-good)/total:.1f}%)")

    del critic_llm
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# DPO Pair Extraction
# ─────────────────────────────────────────────────────────────────────────────

def step_to_text(node):
    parts = []
    if node["think"]:
        parts.append(f"<think>{node['think']}</think>")
    if node["search_query"]:
        parts.append(f"<search>{node['search_query']}</search>")
    elif node["answer_text"]:
        parts.append(f"<answer>{node['answer_text']}</answer>")
    if node.get("documents"):
        parts.append(node["documents"].strip())
    return "\n".join(parts)


def get_action_type(node):
    """Classify step action type (like ReasonRAG paper)."""
    if node["search_query"]:
        return "Search"
    elif node["answer_text"]:
        return "Answer"
    elif node["think"]:
        return "Reason"
    return "Other"


def extract_dpo_pairs(trees, min_diff, critic_alpha=0.0):
    """Extract DPO pairs: F1 primary, Critic PRM as post-scoring bonus.

    Combined reward: combined = f1_reward + α * critic_score
      - F1 is the base signal (consistent with MCTS UCB)
      - Critic adds bonus for good steps (α > 0)
      - If critic not available, falls back to pure F1
    """
    pairs = []
    stats = {
        "total": 0, "same_text": 0, "no_score": 0, "other_chosen": 0,
        "accepted": 0, "no_diff": 0,
        "critic_flipped": 0,  # cases where critic changed F1-only ordering
    }

    use_critic = critic_alpha > 0

    def combined_reward(node):
        """Compute combined reward = f1_reward + α * critic_score."""
        r = node["f1_reward"]
        if use_critic and node.get("critic_score") is not None:
            r += critic_alpha * node["critic_score"]
        return r

    for tree in trees.values():
        for n in tree.nodes.values():
            if len(n["children_ids"]) < 2:
                continue

            children = [tree.nodes[cid] for cid in n["children_ids"]]

            # Build prompt messages (shared prefix up to parent)
            prompt_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {tree.question}"},
            ]
            if n["depth"] > 0:
                prefix = tree.build_prefix(n["id"])
                if prefix:
                    prompt_msgs.append({"role": "assistant", "content": prefix.strip()})

            # Compare all pairs of children
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    a, b = children[i], children[j]
                    stats["total"] += 1

                    if a.get("f1_reward") is None or b.get("f1_reward") is None:
                        stats["no_score"] += 1
                        continue

                    a_text = step_to_text(a)
                    b_text = step_to_text(b)
                    if a_text == b_text:
                        stats["same_text"] += 1
                        continue

                    # Combined reward
                    a_combined = combined_reward(a)
                    b_combined = combined_reward(b)
                    diff = abs(a_combined - b_combined)

                    if diff < min_diff:
                        stats["no_diff"] += 1
                        continue

                    chosen = a if a_combined > b_combined else b
                    rejected = b if a_combined > b_combined else a

                    # Track if critic flipped the F1-only ordering
                    if use_critic:
                        f1_winner = a if a["f1_reward"] > b["f1_reward"] else b
                        if chosen != f1_winner and a["f1_reward"] != b["f1_reward"]:
                            stats["critic_flipped"] += 1

                    # Skip if chosen is "Other" action type
                    if get_action_type(chosen) == "Other":
                        stats["other_chosen"] += 1
                        continue

                    stats["accepted"] += 1
                    pairs.append({
                        "prompt": prompt_msgs,
                        "chosen": [{"role": "assistant", "content": step_to_text(chosen)}],
                        "rejected": [{"role": "assistant", "content": step_to_text(rejected)}],
                        "question_id": tree.qid,
                        "step_level": chosen["depth"],
                        "source": "mcts",
                        "chosen_f1": chosen["f1_reward"],
                        "rejected_f1": rejected["f1_reward"],
                        "chosen_combined": combined_reward(chosen),
                        "rejected_combined": combined_reward(rejected),
                        "chosen_critic": chosen.get("critic_score"),
                        "rejected_critic": rejected.get("critic_score"),
                        "chosen_action_type": get_action_type(chosen),
                        "rejected_action_type": get_action_type(rejected),
                    })

    alpha_str = f", critic α={critic_alpha}" if use_critic else ", no critic"
    print(f"\n[DPO] Pair extraction (F1 primary + PRM post-scoring{alpha_str}):")
    print(f"  Total comparisons: {stats['total']:,}")
    print(f"  Filtered (same text): {stats['same_text']:,}")
    print(f"  Filtered (no score): {stats['no_score']:,}")
    print(f"  Filtered (no diff < {min_diff}): {stats['no_diff']:,}")
    print(f"  Filtered (other chosen): {stats['other_chosen']:,}")
    if use_critic:
        print(f"  Critic-flipped pairs: {stats['critic_flipped']:,}")
    print(f"  Final pairs: {len(pairs):,}")

    step_dist = Counter(p["step_level"] for p in pairs)
    for s in sorted(step_dist):
        print(f"    Step {s}: {step_dist[s]:,}")

    q_coverage = len(set(p["question_id"] for p in pairs))
    print(f"  Questions covered: {q_coverage:,}")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Tree Cache (save/load) + Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_trees(trees, path):
    """Save all trees' nodes to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for tree in trees.values():
            for entry in tree.to_dicts():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    print(f"[Save] {count:,} nodes → {path}")


def save_checkpoint(trees, cache_path, completed_rollout):
    """Save tree checkpoint with rollout metadata (every rollout)."""
    meta_path = cache_path + ".meta"
    # Save trees
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    count = 0
    with open(cache_path, "w") as f:
        for tree in trees.values():
            for entry in tree.to_dicts():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    # Save metadata
    with open(meta_path, "w") as f:
        json.dump({"completed_rollout": completed_rollout, "num_nodes": count}, f)


def load_checkpoint(cache_path):
    """Load checkpoint and return (completed_rollout) or None if no checkpoint."""
    meta_path = cache_path + ".meta"
    if not os.path.exists(cache_path) or not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("completed_rollout", 0)


def load_trees(path, max_children, max_depth):
    """Load trees from JSONL cache."""
    by_qid = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                qid = d["question_id"]
                if qid not in by_qid:
                    by_qid[qid] = {
                        "question": d["question"],
                        "gold_answer": d["gold_answer"],
                        "nodes": [],
                    }
                by_qid[qid]["nodes"].append(d)

    trees = {}
    for qid, info in by_qid.items():
        # Create tree without calling __init__ (avoid creating a duplicate root)
        tree = object.__new__(MCTSTree)
        tree.qid = qid
        tree.question = info["question"]
        tree.gold_answer = info["gold_answer"]
        tree.max_children = max_children
        tree.max_depth = max_depth
        tree.nodes = {}
        max_id = -1
        for nd in info["nodes"]:
            nid = nd["id"]
            # Remove serialization-only fields
            nd.pop("question_id", None)
            nd.pop("question", None)
            nd.pop("gold_answer", None)
            # Ensure new fields have defaults (backward compat)
            nd.setdefault("critic_score", None)
            nd.setdefault("critic_feedback", "")
            tree.nodes[nid] = nd
            if nid > max_id:
                max_id = nid
        tree._next_id = max_id + 1
        # Find root (parent_id is None)
        tree.root_id = next(
            nid for nid, n in tree.nodes.items() if n["parent_id"] is None
        )
        # Recompute expandable count
        tree._expandable_count = sum(
            1 for n in tree.nodes.values()
            if not n["is_terminal"] and len(n["children_ids"]) < max_children
        )
        trees[qid] = tree

    print(f"[Load] {len(trees):,} trees, "
          f"{sum(len(t.nodes) for t in trees.values()):,} nodes")
    return trees


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build MCTS-based DPO dataset (ReasonRAG style)")
    p.add_argument("--input", type=str, required=True,
                   help="Scored trajectories JSONL (trajectory_id, question, gold_answer)")
    p.add_argument("--output", type=str, required=True,
                   help="Output DPO dataset")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of questions (for debugging)")

    # MCTS — matched to ReasonRAG
    p.add_argument("--max-children", type=int, default=2,
                   help="Max children per node (K=2 in ReasonRAG)")
    p.add_argument("--max-depth", type=int, default=7,
                   help="Max tree depth (max_iter=7 in ReasonRAG)")
    p.add_argument("--max-rollouts", type=int, default=64,
                   help="MCTS rollouts per question (64 in ReasonRAG)")

    # Models
    p.add_argument("--policy-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gpu-memory", type=float, default=0.85)

    # Retriever
    p.add_argument("--retriever-model", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--index-path", type=str,
                   default="data/indexes/bge_Flat.index")
    p.add_argument("--corpus-path", type=str,
                   default="data/kilt/kilt_corpus_flashrag.jsonl")
    p.add_argument("--top-k", type=int, default=3)

    # Reward
    p.add_argument("--beta", type=float, default=0.9,
                   help="F1 discount factor: reward = F1 * beta^depth")
    p.add_argument("--min-reward-diff", type=float, default=0.01,
                   help="Min combined reward difference for DPO pair")

    # Critic PRM (post-scoring)
    p.add_argument("--critic-model", type=str, default=None,
                   help="Critic LoRA adapter path (skip critic if not provided)")
    p.add_argument("--critic-base", type=str,
                   default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                   help="Critic base model")
    p.add_argument("--critic-alpha", type=float, default=0.3,
                   help="Critic weight: combined = f1_reward + alpha * critic_score")
    p.add_argument("--critic-interval", type=int, default=4,
                   help="Score with critic every N rollouts (default: 4)")

    # Cache
    p.add_argument("--tree-cache", type=str, default="outputs/mcts_tree_cache.jsonl")

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("MCTS DPO Builder (ReasonRAG style + Critic PRM)")
    print(f"  K={args.max_children}, depth={args.max_depth}, rollouts={args.max_rollouts}")
    print(f"  F1 reward: F1 * {args.beta}^depth")
    if args.critic_model:
        print(f"  Critic PRM: {args.critic_model} (α={args.critic_alpha})")
        print(f"  Combined: f1_reward + {args.critic_alpha} * critic_score")
    else:
        print(f"  Critic: disabled (pure F1)")
    print("=" * 70)

    # ── Extract unique questions from scored trajectories ──
    print(f"\n[Data] Loading questions from {args.input}")
    questions = {}
    with open(args.input) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                qid = t["trajectory_id"].rsplit("_sample_", 1)[0]
                if qid not in questions:
                    questions[qid] = {
                        "question_id": qid,
                        "question": t["question"],
                        "gold_answer": t["gold_answer"],
                    }
    questions = list(questions.values())
    if args.limit:
        questions = questions[:args.limit]
    print(f"  {len(questions):,} unique questions")

    # ── Check if MCTS already fully completed ──
    ckpt_rollout = load_checkpoint(args.tree_cache)
    mcts_done = (ckpt_rollout is not None and ckpt_rollout >= args.max_rollouts)

    if mcts_done:
        print(f"\n[Cache] MCTS already completed ({ckpt_rollout} rollouts), loading tree...")
        trees = load_trees(args.tree_cache, args.max_children, args.max_depth)
    else:
        # ── Load retriever (stays in memory during MCTS) ──
        retriever = BGERetriever(
            model_path=args.retriever_model,
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            top_k=args.top_k,
        )

        # ── Run MCTS (resumes from checkpoint if available) ──
        print("\n" + "=" * 70)
        print("[Phase 1] MCTS Tree Building")
        print("=" * 70)
        trees = run_mcts(questions, retriever, args)

        # Free retriever
        del retriever
        gc.collect()

        # Save final tree cache
        save_trees(trees, args.tree_cache)

    # ── Post-MCTS: F1 Scoring + Propagation ──
    print("\n" + "=" * 70)
    print("[Phase 2] F1 Scoring + Bottom-up Propagation")
    print("=" * 70)
    score_and_propagate(trees, args.beta)

    # ── Critic PRM Post-Scoring (optional) ──
    critic_alpha = 0.0
    if args.critic_model:
        print("\n" + "=" * 70)
        print("[Phase 3] Critic PRM Post-Scoring")
        print("=" * 70)
        score_with_critic(trees, args.critic_model, args.critic_base, args.gpu_memory)
        critic_alpha = args.critic_alpha

        # Save tree with critic scores
        save_trees(trees, args.tree_cache)
    else:
        print("\n[Phase 3] Critic skipped (no --critic-model)")

    # ── DPO Pair Extraction ──
    print("\n" + "=" * 70)
    print("[Phase 4] DPO Pair Extraction")
    print("=" * 70)
    pairs = extract_dpo_pairs(trees, args.min_reward_diff, critic_alpha=critic_alpha)

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n[Done] Saved {len(pairs):,} DPO pairs → {args.output}")

    # Save scored tree
    scored_path = args.output.replace(".jsonl", "_tree.jsonl")
    save_trees(trees, scored_path)

    print("=" * 70)


if __name__ == "__main__":
    main()
