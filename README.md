# PRO-Step: Step-level Process Reward Optimization for Retrieval-Augmented Generation

## Overview

PRO-STEP integrates **process-level supervision** into RAG to mitigate error propagation in multi-hop reasoning. Our approach trains a **generative PRM** that evaluates both logical validity and evidential grounding of intermediate steps, then employs **PRM-guided MCTS** to construct step-level preference pairs for **DPO** training.

### Key Components
1. **Generative PRM**: DeepSeek-R1-Distill-8B + LoRA, trained on QwQ-32B judge labels (109,664 step annotations from 2,000 questions)
2. **PRM-guided MCTS**: Constructs preference pairs that contrast valid reasoning steps against flawed ones (13,183 pairs from 5,000 questions)
3. **Step-level DPO**: Qwen2.5-7B-Instruct policy with document masking
4. **Evaluation**: FlashRAG SearchR1Pipeline on 5 QA benchmarks (PopQA, HotpotQA, 2WikiMultiHopQA, Bamboogle, MuSiQue)

## Project Structure

```
src/prmrag/
├── config/          # Configuration
├── data/            # Data schemas & loaders
├── labeling/        # Judge labeling (QwQ-32B)
├── models/          # vLLM inference wrapper
├── retrieval/       # BGE, E5, BM25 retrievers
├── training/        # Generative PRM, DPO, SFT, KTO trainers
└── utils/           # EM/F1 metrics

scripts/
├── generate_trajectories.py       # Trajectory generation (Stage 1)
├── judge_label_qwq.py             # QwQ-32B step annotation (Stage 1)
├── train_generative_model.py      # Generative PRM training (Stage 1)
├── build_mcts_dpo.py              # PRM-guided MCTS + DPO pair extraction (Stage 2)
├── train_dpo_policy.py            # DPO training (document masking)
├── train_sft_policy.py            # SFT training
├── train_kto_policy.py            # KTO training
├── merge_lora.py                  # Merge LoRA into base model
├── eval_flashrag.py               # Evaluation (FlashRAG)
├── compare_voting_methods.py      # PRM test-time scaling (Figure 3)
└── regenerate_with_generative.py  # PRM-guided regeneration
```

## Requirements

```bash
pip install -r requirements.txt
# flash-attn (optional, install separately):
#   pip install flash-attn --no-build-isolation
```

## Data Preparation

1. **Corpus**: 2018 Wikipedia dump (5.9M passages) from [KILT](https://github.com/facebookresearch/KILT)
2. **Retriever**: BGE-base-en-v1.5 with FAISS IVF4096 index (nprobe=128)
3. **QA Datasets**: HotpotQA, PopQA, 2WikiMultiHopQA, Bamboogle, MuSiQue (via [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG))
4. **Training Questions**: 2,000 HotpotQA + 2,000 MuSiQue + 1,000 2WikiMulti (5,000 total)

## Reproduction Guide

### Stage 1: Process Reward Modeling

```bash
# 1a. Generate candidate trajectories (16 per question, 2,000 questions)
python scripts/generate_trajectories.py \
    --questions data/raw/questions/hotpotqa_train.jsonl \
    --num-questions 2000 \
    --num-trajectories 16 \
    --output outputs/trajectories.jsonl

# 1b. Generate step annotations with QwQ-32B
python scripts/judge_label_qwq.py \
    --input outputs/trajectories.jsonl \
    --output outputs/judge_labels.jsonl

# 1c. Train generative PRM (DeepSeek-R1-Distill-8B + LoRA)
torchrun --nproc_per_node=2 scripts/train_generative_model.py \
    --train-data outputs/judge_labels.jsonl \
    --output-dir outputs/generative_model
```

### Stage 2: Policy Optimization

```bash
# 2a. PRM-guided MCTS + DPO pair extraction
python scripts/build_mcts_dpo.py \
    --questions data/raw/questions/hotpotqa_train.jsonl \
    --num-questions 5000 \
    --generative-model outputs/generative_model/final_model \
    --generative-alpha 0.3 \
    --output outputs/mcts_dpo.jsonl

# 2b. DPO training with document masking
torchrun --nproc_per_node=2 scripts/train_dpo_policy.py \
    --dpo-dataset outputs/mcts_dpo.jsonl \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-dir outputs/dpo_policy \
    --beta 0.1 --num-epochs 1 --lora-r 64 --lora-alpha 128

# 2c. Merge LoRA
python scripts/merge_lora.py \
    --base Qwen/Qwen2.5-7B-Instruct \
    --lora outputs/dpo_policy/final_model \
    --output outputs/dpo_policy/merged_model
```

### Evaluation

```bash
# Single dataset
python scripts/eval_flashrag.py \
    --model outputs/dpo_policy/merged_model \
    --dataset hotpotqa \
    --temperature 0 \
    --tag eval_hotpotqa

# PRM test-time scaling (Figure 3)
python scripts/compare_voting_methods.py \
    --trajectories outputs/prm_scaling/trajectories.jsonl \
    --generative-model outputs/generative_model/final_model \
    --k-values 1,2,4,8,16,32,64,128
```

## Hyperparameters

### MCTS (Table 5)

| Parameter | Value |
|-----------|-------|
| Branching factor (K) | 2 |
| Max depth | 7 |
| Iterations per question | 64 |
| F1 discount (γ) | 0.9 (Reward = F1 · 0.9^d) |
| PRM weight (α) | 0.3 (Combined = Reward + 0.3 · PRM) |
| PRM scoring interval | Every 4 iterations |
| Retrieved docs per search | 3 |
| Preference margin (δ) | 0.01 |

### Policy Training (Table 6)

| Parameter | SFT | DPO | KTO |
|-----------|-----|-----|-----|
| Base Model | Qwen2.5-7B-Instruct | same | same |
| Objective | Next-token | Sigmoid (β=0.1) | KTO (β=0.1) |
| Learning Rate | 2e-5 | 2e-5 | 5e-6 |
| Effective Batch Size | 16 | 16 | 16 |
| Epochs | 1 | 1 | 1 |
| Max Length | 8,192 | 8,192 | 8,192 |
| Document Masking | Yes | Yes | Yes |
| LoRA r / α | 64 / 128 | 64 / 128 | 64 / 128 |

### Generative PRM

| Parameter | Value |
|-----------|-------|
| Base model | DeepSeek-R1-Distill-8B |
| LoRA r | 16 |
| Learning rate | 1e-5 |
| Training samples | 109,664 steps |

## License

MIT
