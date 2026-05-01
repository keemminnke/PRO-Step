# PRO-STEP: Step-level Process Reward Optimization for Retrieval-Augmented Generation

> A self-improving framework for agentic Retrieval-Augmented Generation, using only open-source supervision (no closed-API teacher). Trains a Qwen2.5-7B-Instruct policy on its own MCTS trajectories, scored by an open-source 8B Process Reward Model, via step-level Direct Preference Optimization.

## 🤗 Released artifacts

### Models
- **Policy**: [DORAEMONG/PRO-STEP-Policy-7B](https://huggingface.co/DORAEMONG/PRO-STEP-Policy-7B) — Qwen2.5-7B-Instruct + DPO + outcome filter + α=0.3 PRM
- **Process Reward Model**: [DORAEMONG/PRO-STEP-PRM-8B](https://huggingface.co/DORAEMONG/PRO-STEP-PRM-8B) — LoRA over DeepSeek-R1-0528-Qwen3-8B

### Datasets
- **DPO Preference Pairs**: [DORAEMONG/PRO-STEP-Preference-Data](https://huggingface.co/datasets/DORAEMONG/PRO-STEP-Preference-Data) — 15,877 step-level outcome-filtered pairs
- **PRM Training Annotations**: [DORAEMONG/PRO-STEP-PRM-Data](https://huggingface.co/datasets/DORAEMONG/PRO-STEP-PRM-Data) — ~109K step labels across 31,728 trajectories

## Performance (5-dataset, identical FlashRAG eval pipeline)

| Method | Train data | HotpotQA | PopQA | 2Wiki | Bamboogle | Musique | **AVG** |
|---|---|---|---|---|---|---|---|
| Search-R1 | ~90,000 | 37.88 / 49.56 | **40.65** / 46.78 | 34.87 / 42.50 | 33.60 / 43.55 | 12.99 / 21.23 | 32.00 / 40.72 |
| ReasonRAG | ~5,000 | 36.37 / 47.51 | 37.78 / 44.87 | 39.80 / 46.32 | **38.40** / 46.86 | 10.59 / 19.22 | 32.59 / 40.96 |
| StepSearch | ~19,000 | 38.72 / 50.67 | 39.24 / 44.97 | 40.38 / 47.12 | 33.60 / 44.16 | **13.82 / 23.06** | 33.15 / 42.00 |
| **PRO-STEP (ours)** ★ | **5,000** | **38.73 / 51.63** | 40.47 / **47.37** | **44.07 / 51.43** | 36.80 / **47.63** | 12.49 / 22.41 | **34.51 / 44.09** |

EM / F1 (Strict EM, token-F1). Bootstrap 95% CI: vs Search-R1 +2.51 EM [+1.01, +4.06], vs ReasonRAG +1.93 EM [+0.46, +3.36]. Strongest result on **2WikiMultiHopQA**: +9.20 EM over Search-R1 (p<10⁻⁷⁵).

## Pipeline overview

```
                          ┌──────────────────────┐
                          │  Qwen2.5-7B-Instruct │  ← policy backbone
                          └──────────┬───────────┘
                                     │ MCTS rollout
                                     │ (K=3 branching, depth 7, 64 rollouts/q)
                                     ↓
                  ┌────────────────────────────────┐
                  │  Open-source 8B PRM (ours)     │  ← step-level scoring
                  │  (DeepSeek-R1-0528-Qwen3-8B    │     V(s) = Q̄(s) + α·r̂(s)
                  │  fine-tuned on QwQ-32B labels) │
                  └────────────────┬───────────────┘
                                   │
                                   │ Outcome filter
                                   │ (chosen_F1 ≥ 0.2 AND Δf1 ≥ 0.2)
                                   ↓
                  ┌────────────────────────────────┐
                  │  15,877 step-level pref pairs  │
                  └────────────────┬───────────────┘
                                   │ Step-level DPO
                                   │ (β=0.1, doc-token masking)
                                   ↓
                          ┌─────────────────┐
                          │  PRO-STEP Policy│  ← released model
                          └─────────────────┘
```

## Quick start

### Inference with the policy

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("DORAEMONG/PRO-STEP-Policy-7B", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("DORAEMONG/PRO-STEP-Policy-7B")

# Use with FlashRAG SearchR1Pipeline or any agentic-RAG inference loop
# System prompt: see paper Appendix A
```

### Inference with the PRM

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", torch_dtype="auto", device_map="auto")
prm  = PeftModel.from_pretrained(base, "DORAEMONG/PRO-STEP-PRM-8B")
# See paper Appendix A for the MCTS scoring prompt
```

## Repository structure

```
scripts/
  generate_trajectories.py      # Stage 1: PRM training data generation
  build_mcts_dpo.py             # Stage 2: PRM-guided MCTS rollout + pair extraction
  train_dpo_policy.py           # Stage 3: step-level DPO training
  eval_flashrag.py              # FlashRAG-based evaluation
src/
  prmrag/                       # Core library
PAPER_RESULTS.md                # Full results, ablations, statistical tests
REVIEWER_RESPONSE.md            # Detailed reviewer responses
```

## Key contributions

1. **Self-improving framework with open-source supervision** — uses only Qwen2.5-7B-Instruct and an open-source 8B PRM, no closed-API teacher
2. **PRM directly inside the MCTS value function** — V(s) = Q̄(s) + α·r̂(s), unlike ReasonRAG's UCB-only auxiliary use
3. **Sample efficiency** — 5,000 training questions vs Search-R1's ~90,000 (18× less data)
4. **Multi-hop dominance** — +9.20 EM on 2WikiMultiHopQA over Search-R1 (p<10⁻⁷⁵)
5. **PRM dominates learned baselines** — beats VersaPRM by +15.0 EM and Math-PRM by +8.4 EM on 2Wiki BoN at K=128
6. **Open release** — full pipeline (model weights, PRM, preference data, training labels) on Hugging Face

## Citation

```bibtex
@article{prostep2026,
  title={PRO-STEP: Step-level Process Reward Optimization for Retrieval-Augmented Generation},
  author={...},
  year={2026}
}
```

## License

MIT for code and models; CC-BY-SA-4.0 for derived datasets.
