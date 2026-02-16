"""
LLM loading and LangChain pipeline creation for summary generation and judging.

Supports quantized loading (4-bit / 8-bit) via BitsAndBytes and
creates LangChain chains for both summary generation and G-Eval judging.
"""

from typing import Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, set_seed as hf_set_seed
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sm_sip.utils.seed import set_seed, DEFAULT_SEED


def load_llm(
    model_id: str,
    quantization: str = "8bit",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
    seed: int = DEFAULT_SEED,
) -> Tuple:
    """Load a quantized LLM for text generation.

    Args:
        model_id: HuggingFace model ID.
        quantization: "4bit" or "8bit".
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        do_sample: Whether to use sampling.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (model, tokenizer, hf_pipeline).
    """
    # Set all seeds for reproducibility
    set_seed(seed)
    hf_set_seed(seed)
    print(f"  Loading LLM ({quantization}, seed={seed}): {model_id}...")

    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
    )

    return model, tokenizer, gen_pipe


def create_summary_chain(gen_pipe, prompt_template: str):
    """Create a LangChain chain for summary generation.

    Args:
        gen_pipe: HuggingFace text-generation pipeline.
        prompt_template: Prompt template string with {source} and {keyphrases} placeholders.

    Returns:
        LangChain chain (prompt | llm | parser).
    """
    llm = HuggingFacePipeline(pipeline=gen_pipe)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["source", "keyphrases"],
    )
    return prompt | llm | StrOutputParser()


def create_judge_chain(gen_pipe, judge_template: str):
    """Create a LangChain chain for LLM-as-Judge evaluation.

    Args:
        gen_pipe: HuggingFace text-generation pipeline.
        judge_template: Judge prompt template with {source}, {reference}, {generated} placeholders.

    Returns:
        LangChain chain (prompt | llm | parser).
    """
    llm = HuggingFacePipeline(pipeline=gen_pipe)
    prompt = PromptTemplate(
        template=judge_template,
        input_variables=["source", "reference", "generated"],
    )
    return prompt | llm | StrOutputParser()
