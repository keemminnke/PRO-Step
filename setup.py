from setuptools import setup, find_packages

setup(
    name="prostep",
    version="1.0.0",
    description="Process Reward Models for Multi-Step Retrieval-Augmented Generation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "vllm>=0.14.0",
        "peft>=0.18.0",
        "trl>=0.29.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "flashrag",
    ],
)
