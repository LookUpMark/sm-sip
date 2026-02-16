"""
Centralized configuration dataclasses for all SM-SIP experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class SigExtConfig:
    """Configuration for a SigExt keyphrase extraction model."""

    model_id: str
    skip_samples: int = 0
    threshold: float = 0.60
    max_length: int = 2048

    # Pre-defined Italian WITS configurations
    ITALIAN_CONFIGS = {
        "10k-60t": {"model_id": "LookUpMark/sigext-wits-it-10k-060t", "skip_samples": 25000, "threshold": 0.60},
        "25k-60t": {"model_id": "LookUpMark/sigext-wits-it-25k-060t", "skip_samples": 25000, "threshold": 0.60},
        "25k-65t": {"model_id": "LookUpMark/sigext-wits-it-25k-065t", "skip_samples": 25000, "threshold": 0.65},
        "25k-70t": {"model_id": "LookUpMark/sigext-wits-it-25k-070t", "skip_samples": 25000, "threshold": 0.70},
    }

    # Pre-defined English ArXiv configuration
    ENGLISH_CONFIGS = {
        "1k-60t": {"model_id": "LookUpMark/sigext-arxiv-en-1k-060t", "skip_samples": 0, "threshold": 0.60},
    }

    @classmethod
    def from_preset(cls, lang: str, config_name: str) -> "SigExtConfig":
        """Create a SigExtConfig from a named preset.

        Args:
            lang: "it" or "en"
            config_name: e.g. "10k-60t", "25k-65t", "1k-60t"
        """
        configs = cls.ITALIAN_CONFIGS if lang == "it" else cls.ENGLISH_CONFIGS
        if config_name not in configs:
            raise ValueError(f"Unknown config '{config_name}' for lang '{lang}'. Available: {list(configs.keys())}")
        return cls(**configs[config_name])


@dataclass
class InferenceConfig:
    """Configuration for inference (summary generation)."""

    lang: Literal["it", "en"] = "it"
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    quantization: Literal["4bit", "8bit"] = "8bit"
    prompt_type: Literal["zero_shot", "few_shot", "source_aware"] = "source_aware"
    num_test_samples: int = 100
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = False  # greedy by default
    output_dir: str = "./results"
    seed: int = 42  # Reproducibility seed


@dataclass
class EvalConfig:
    """Configuration for evaluation (metrics + LLM-as-Judge)."""

    lang: Literal["it", "en"] = "it"
    judge_model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    use_judge: bool = True
    use_vllm: bool = False  # Use vLLM engine for judge (faster)
    rouge_metrics: List[str] = field(default_factory=lambda: ["rouge1", "rougeL"])
    bert_score_lang: Optional[str] = None  # Auto-detected from lang if None
    seed: int = 42  # Reproducibility seed

    def __post_init__(self):
        if self.bert_score_lang is None:
            self.bert_score_lang = self.lang


@dataclass
class TrainingConfig:
    """Configuration for SigExt model training."""

    lang: Literal["it", "en"] = "it"
    base_model_id: str = "allenai/longformer-base-4096"
    dataset_name: str = "silvia-casola/WITS"  # HuggingFace dataset
    num_samples: int = 25000
    similarity_threshold: float = 0.60
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_length: int = 2048
    output_model_name: str = "sigext-wits-it-25k-060t"
    push_to_hub: bool = True
    seed: int = 42  # Reproducibility seed


@dataclass
class AblationConfig:
    """Configuration for ablation study experiments."""

    name: str = "default"
    description: str = ""

    # Axes to vary
    quantizations: List[str] = field(default_factory=lambda: ["4bit", "8bit"])
    prompt_types: List[str] = field(default_factory=lambda: ["zero_shot", "few_shot", "source_aware"])
    temperatures: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    sigext_configs: List[str] = field(default_factory=lambda: ["10k-60t"])
    languages: List[str] = field(default_factory=lambda: ["it", "en"])
    judge_models: List[str] = field(default_factory=lambda: ["Qwen/Qwen2.5-14B-Instruct"])
