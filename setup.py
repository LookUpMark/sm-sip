"""
SM-SIP: Semantic & Multilingual Salient Information Prompting

Extension of the SigExt framework for controllable abstractive
summarization with semantic supervision and multilingual support.
"""

from setuptools import setup, find_packages

setup(
    name="sm-sip",
    version="1.0.0",
    description="Semantic & Multilingual Salient Information Prompting",
    author="SM-SIP Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers",
        "spacy",
        "rouge_score",
        "bert_score",
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-huggingface",
        "huggingface_hub",
        "numpy<2.0",
        "scipy>1.10",
        "tqdm",
        "matplotlib",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "vllm": ["vllm"],
    },
)
