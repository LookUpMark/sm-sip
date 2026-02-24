"""
Centralized configuration dataclasses for all SM-SIP experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict


@dataclass
class SigExtConfig:
    """Configuration for a SigExt keyphrase extraction model."""

    model_id: str
    skip_samples: int = 0
    threshold: float = 0.60
    max_length: int = 2048

    # Available base models for SigExt training
    BASE_MODELS = {
        "allenai": "allenai/longformer-base-4096",
        "xlmr": "markussagen/xlm-roberta-longformer-base-4096",
    }

    # Standard ablation parameters
    SAMPLE_SIZES = [1000, 2500, 5000]
    THRESHOLDS = [0.60, 0.70, 0.80]
    SEED = 42

    # Allowed base models per language:
    #   EN -> allenai (English-specific) + xlmr (multilingual)
    #   IT -> xlmr only (no Italian-specific Longformer exists)
    LANG_BASE_MODELS = {
        "it": ["xlmr"],
        "en": ["allenai", "xlmr"],
    }

    LANG_DATASETS = {
        "it": {"dataset": "silvia-casola/WITS", "prefix": "wits", "skip": 25000},
        "en": {"dataset": "ccdv/arxiv-summarization", "prefix": "arxiv", "skip": 0},
    }

    # ---------- Italian WITS configs (xlmr only) ----------
    ITALIAN_CONFIGS = {
        # xlmr × 3 sizes × 3 thresholds = 9
        "xlmr-1k-060t":   {"model_id": "LookUpMark/sigext-wits-it-xlmr-1k-060t",   "skip_samples": 25000, "threshold": 0.60},
        "xlmr-1k-070t":   {"model_id": "LookUpMark/sigext-wits-it-xlmr-1k-070t",   "skip_samples": 25000, "threshold": 0.70},
        "xlmr-1k-080t":   {"model_id": "LookUpMark/sigext-wits-it-xlmr-1k-080t",   "skip_samples": 25000, "threshold": 0.80},
        "xlmr-2500-060t": {"model_id": "LookUpMark/sigext-wits-it-xlmr-2500-060t", "skip_samples": 25000, "threshold": 0.60},
        "xlmr-2500-070t": {"model_id": "LookUpMark/sigext-wits-it-xlmr-2500-070t", "skip_samples": 25000, "threshold": 0.70},
        "xlmr-2500-080t": {"model_id": "LookUpMark/sigext-wits-it-xlmr-2500-080t", "skip_samples": 25000, "threshold": 0.80},
        "xlmr-5k-060t":   {"model_id": "LookUpMark/sigext-wits-it-xlmr-5k-060t",   "skip_samples": 25000, "threshold": 0.60},
        "xlmr-5k-070t":   {"model_id": "LookUpMark/sigext-wits-it-xlmr-5k-070t",   "skip_samples": 25000, "threshold": 0.70},
        "xlmr-5k-080t":   {"model_id": "LookUpMark/sigext-wits-it-xlmr-5k-080t",   "skip_samples": 25000, "threshold": 0.80},
    }

    # ---------- English ArXiv configs (allenai + xlmr) ----------
    ENGLISH_CONFIGS = {
        # allenai × 3 sizes × 3 thresholds = 9
        "allenai-1k-060t":   {"model_id": "LookUpMark/sigext-arxiv-en-allenai-1k-060t",   "skip_samples": 0, "threshold": 0.60},
        "allenai-1k-070t":   {"model_id": "LookUpMark/sigext-arxiv-en-allenai-1k-070t",   "skip_samples": 0, "threshold": 0.70},
        "allenai-1k-080t":   {"model_id": "LookUpMark/sigext-arxiv-en-allenai-1k-080t",   "skip_samples": 0, "threshold": 0.80},
        "allenai-2500-060t": {"model_id": "LookUpMark/sigext-arxiv-en-allenai-2500-060t", "skip_samples": 0, "threshold": 0.60},
        "allenai-2500-070t": {"model_id": "LookUpMark/sigext-arxiv-en-allenai-2500-070t", "skip_samples": 0, "threshold": 0.70},
        "allenai-2500-080t": {"model_id": "LookUpMark/sigext-arxiv-en-allenai-2500-080t", "skip_samples": 0, "threshold": 0.80},
        "allenai-5k-060t":   {"model_id": "LookUpMark/sigext-arxiv-en-allenai-5k-060t",   "skip_samples": 0, "threshold": 0.60},
        "allenai-5k-070t":   {"model_id": "LookUpMark/sigext-arxiv-en-allenai-5k-070t",   "skip_samples": 0, "threshold": 0.70},
        "allenai-5k-080t":   {"model_id": "LookUpMark/sigext-arxiv-en-allenai-5k-080t",   "skip_samples": 0, "threshold": 0.80},
        # xlmr × 3 sizes × 3 thresholds = 9
        "xlmr-1k-060t":   {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-1k-060t",   "skip_samples": 0, "threshold": 0.60},
        "xlmr-1k-070t":   {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-1k-070t",   "skip_samples": 0, "threshold": 0.70},
        "xlmr-1k-080t":   {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-1k-080t",   "skip_samples": 0, "threshold": 0.80},
        "xlmr-2500-060t": {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-2500-060t", "skip_samples": 0, "threshold": 0.60},
        "xlmr-2500-070t": {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-2500-070t", "skip_samples": 0, "threshold": 0.70},
        "xlmr-2500-080t": {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-2500-080t", "skip_samples": 0, "threshold": 0.80},
        "xlmr-5k-060t":   {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-5k-060t",   "skip_samples": 0, "threshold": 0.60},
        "xlmr-5k-070t":   {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-5k-070t",   "skip_samples": 0, "threshold": 0.70},
        "xlmr-5k-080t":   {"model_id": "LookUpMark/sigext-arxiv-en-xlmr-5k-080t",   "skip_samples": 0, "threshold": 0.80},
    }

    # Default preset per language (used by inference/other ablation notebooks)
    DEFAULT_PRESET = {
        "it": "xlmr-5k-060t",
        "en": "xlmr-5k-060t",
    }

    @classmethod
    def from_preset(cls, lang: str, config_name: str) -> "SigExtConfig":
        """Create a SigExtConfig from a named preset.

        Args:
            lang: "it" or "en"
            config_name: e.g. "xlmr-5k-060t", "allenai-2500-070t"
        """
        configs = cls.ITALIAN_CONFIGS if lang == "it" else cls.ENGLISH_CONFIGS
        if config_name not in configs:
            raise ValueError(f"Unknown config '{config_name}' for lang '{lang}'. Available: {list(configs.keys())}")
        return cls(**configs[config_name])

    @classmethod
    def get_default(cls, lang: str) -> "SigExtConfig":
        """Get the default SigExtConfig for a language."""
        return cls.from_preset(lang, cls.DEFAULT_PRESET[lang])


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
    base_model_id: str = "markussagen/xlm-roberta-longformer-base-4096"
    dataset_name: str = "silvia-casola/WITS"  # HuggingFace dataset
    num_samples: int = 5000
    similarity_threshold: float = 0.60
    epochs: int = 10  # Max epochs (early stopping will likely stop sooner)
    patience: int = 2  # Stop after N epochs without val loss improvement
    val_split: float = 0.1  # Fraction of data for validation
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_length: int = 2048
    output_model_name: str = "sigext-wits-it-xlmr-5k-060t"
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
    sigext_configs: List[str] = field(default_factory=lambda: ["xlmr-5k-060t"])
    languages: List[str] = field(default_factory=lambda: ["it", "en"])
    judge_models: List[str] = field(default_factory=lambda: ["Qwen/Qwen2.5-14B-Instruct"])

