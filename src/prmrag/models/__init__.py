"""Model loading and inference utilities.

By default, uses vLLM for faster inference on GH200/GPU systems.
Falls back to HuggingFace Transformers if vLLM is not available.
"""

import os

# Check if user wants to force HF Transformers backend
USE_HF_BACKEND = os.environ.get('PRMRAG_USE_HF_BACKEND', '0') == '1'

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Use vLLM by default if available, unless user explicitly requests HF backend
if VLLM_AVAILABLE and not USE_HF_BACKEND:
    from .policy_model_vllm import PolicyModelVLLM as PolicyModel
    from .policy_model_vllm import load_policy_model
    print("✓ Using vLLM backend for PolicyModel")
else:
    from .policy_model import PolicyModel, load_policy_model
    if USE_HF_BACKEND:
        print("✓ Using HuggingFace Transformers backend (forced by PRMRAG_USE_HF_BACKEND)")
    else:
        print("⚠ vLLM not available, falling back to HuggingFace Transformers backend")

# Also export the specific implementations for explicit usage
from .policy_model import PolicyModel as PolicyModelHF
from .policy_model import load_policy_model as load_policy_model_hf

if VLLM_AVAILABLE:
    from .policy_model_vllm import PolicyModelVLLM
    from .policy_model_vllm import load_policy_model as load_policy_model_vllm
else:
    PolicyModelVLLM = None
    load_policy_model_vllm = None

__all__ = [
    'PolicyModel',
    'load_policy_model',
    'PolicyModelHF',
    'load_policy_model_hf',
    'PolicyModelVLLM',
    'load_policy_model_vllm',
    'VLLM_AVAILABLE',
]
