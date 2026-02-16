"""
LLM-as-Judge evaluation: response parsing and scoring.

Handles robust JSON parsing of LLM judge responses and provides
a high-level function for evaluating a single sample.
"""

import re
import json
from typing import Dict, Optional

JUDGE_DIMENSIONS = ["faithfulness", "completeness", "conciseness", "abstraction"]
DEFAULT_SCORES = {k: 3 for k in JUDGE_DIMENSIONS}
DEFAULT_SCORES.update({f"{k}_reason": "Unable to evaluate" for k in JUDGE_DIMENSIONS})


def parse_judge_response(text: str) -> Dict:
    """Robustly parse JSON response from LLM judge.

    Attempts multiple parsing strategies:
    1. Direct JSON extraction
    2. Quote/trailing comma cleanup
    3. Regex-based key-value extraction

    Args:
        text: Raw LLM judge output text.

    Returns:
        Dict with dimension scores and reasons.
    """
    # Method 1: Try direct JSON parsing
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Method 2: Clean common issues and retry
    if json_match:
        json_str = json_match.group()
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r",\s*}", "}", json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Method 3: Extract individual values with regex
    scores = {}
    for key in JUDGE_DIMENSIONS:
        match = re.search(rf'"{key}"\s*:\s*(\d)', text, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))
        reason_match = re.search(rf'"{key}_reason"\s*:\s*"([^"]*)"', text)
        if reason_match:
            scores[f"{key}_reason"] = reason_match.group(1)

    return scores


def llm_judge_evaluate(
    source: str,
    generated: str,
    reference: str,
    judge_chain,
    response_separator: str = "assistant<|end_header_id|>",
) -> Dict:
    """Use LLM judge to evaluate a generated summary.

    Args:
        source: Source document.
        generated: Generated summary.
        reference: Reference summary.
        judge_chain: LangChain chain for judge evaluation.
        response_separator: String to split response and extract judge output.

    Returns:
        Dict with scores for each dimension (1-5) and reasons.
    """
    try:
        result = judge_chain.invoke({
            "source": source,
            "reference": reference,
            "generated": generated,
        })

        result_text = result.split(response_separator)[-1].strip()
        scores = parse_judge_response(result_text)

        # Validate and clamp scores
        for key in JUDGE_DIMENSIONS:
            if key not in scores:
                scores[key] = 3
            else:
                scores[key] = max(1, min(5, int(scores[key])))
            reason_key = f"{key}_reason"
            if reason_key not in scores:
                scores[reason_key] = ""

        return scores

    except Exception as e:
        print(f"      Judge error: {e}")
        return DEFAULT_SCORES.copy()
