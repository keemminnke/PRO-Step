from setuptools import setup, find_packages

setup(
    name="prmrag",
    version="0.1.0",
    description="Consensus-Based Auto-Labeling Pipeline for RAG-CoT",
    author="PRMRAG Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "jsonlines>=4.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "llm": [
            "openai>=1.3.0",
            "anthropic>=0.8.0",
        ],
        "fast": [
            "vllm>=0.2.0",
        ],
    },
)
