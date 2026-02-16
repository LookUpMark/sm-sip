"""
G-Eval LLM-as-Judge prompt templates.

Provides both a unified prompt (single call evaluating all dimensions)
and per-dimension prompts (for fine-grained evaluation).

All prompts follow the Llama-3.1 chat format.
"""

# Unified judge prompt — evaluates all 4 dimensions in one call
JUDGE_PROMPT_UNIFIED = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are an expert summary evaluator. You have access to the full SOURCE document "
    "and the REFERENCE summary.\n"
    "Evaluate if the GENERATED summary is faithful to the source content.\n"
    "IMPORTANT: The generated summary may include information from the SOURCE that is NOT "
    "in the reference. This is acceptable as long as all information is verifiable in the SOURCE.\n\n"
    "Respond ONLY with valid JSON. Use double quotes. Max 15 words per reason.\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "SOURCE DOCUMENT:\n{source}\n\n"
    "REFERENCE SUMMARY:\n{reference}\n\n"
    "GENERATED SUMMARY:\n{generated}\n\n---\n"
    "Rate (1-5 each):\n"
    "1. FAITHFULNESS: All facts verifiable in SOURCE? (1=fabricated, 5=all verified)\n"
    "2. COMPLETENESS: Covers main points? (1=missing, 5=complete)\n"
    "3. CONCISENESS: Fluid, not repetitive? (1=verbose, 5=concise)\n"
    "4. ABSTRACTION: Rephrases vs copies? (1=verbatim, 5=novel phrasing)\n\n"
    '{{\"faithfulness\": X, \"faithfulness_reason\": \"...\", '
    '\"completeness\": X, \"completeness_reason\": \"...\", '
    '\"conciseness\": X, \"conciseness_reason\": \"...\", '
    '\"abstraction\": X, \"abstraction_reason\": \"...\"}}\n'
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

# Per-dimension G-Eval prompts (for Qwen/vLLM-based evaluation)
JUDGE_PROMPTS = {
    "faithfulness": (
        "<|im_start|>system\n"
        "You are evaluating the factual correctness of a generated summary.\n"
        "Score from 1 to 5 where:\n"
        "1 = Contains fabricated facts not in source\n"
        "5 = All facts are verifiable in the source document\n"
        "<|im_end|>\n<|im_start|>user\n"
        "SOURCE:\n{source}\n\nGENERATED SUMMARY:\n{generated}\n\n"
        "Score (1-5): <|im_end|>\n<|im_start|>assistant\n"
    ),
    "completeness": (
        "<|im_start|>system\n"
        "You are evaluating the completeness of a generated summary.\n"
        "Score from 1 to 5 where:\n"
        "1 = Missing most key points from the source\n"
        "5 = Covers all main points comprehensively\n"
        "<|im_end|>\n<|im_start|>user\n"
        "SOURCE:\n{source}\n\nREFERENCE:\n{reference}\n\nGENERATED:\n{generated}\n\n"
        "Score (1-5): <|im_end|>\n<|im_start|>assistant\n"
    ),
    "conciseness": (
        "<|im_start|>system\n"
        "You are evaluating the conciseness of a generated summary.\n"
        "Score from 1 to 5 where:\n"
        "1 = Verbose and repetitive\n"
        "5 = Concise with no redundancy\n"
        "<|im_end|>\n<|im_start|>user\n"
        "GENERATED SUMMARY:\n{generated}\n\n"
        "Score (1-5): <|im_end|>\n<|im_start|>assistant\n"
    ),
    "abstraction": (
        "<|im_start|>system\n"
        "You are evaluating the abstraction quality of a generated summary.\n"
        "Score from 1 to 5 where:\n"
        "1 = Mostly verbatim copies from source\n"
        "5 = Fully rephrased with novel wording\n"
        "<|im_end|>\n<|im_start|>user\n"
        "SOURCE:\n{source}\n\nGENERATED SUMMARY:\n{generated}\n\n"
        "Score (1-5): <|im_end|>\n<|im_start|>assistant\n"
    ),
}


def get_judge_prompt(prompt_type: str = "unified") -> str:
    """Get a judge prompt template.

    Args:
        prompt_type: "unified" for all-in-one, or a dimension name
                     ("faithfulness", "completeness", "conciseness", "abstraction").

    Returns:
        Prompt template string.
    """
    if prompt_type == "unified":
        return JUDGE_PROMPT_UNIFIED
    if prompt_type not in JUDGE_PROMPTS:
        raise ValueError(
            f"Unknown judge prompt type '{prompt_type}'. "
            f"Available: 'unified', {list(JUDGE_PROMPTS.keys())}"
        )
    return JUDGE_PROMPTS[prompt_type]
