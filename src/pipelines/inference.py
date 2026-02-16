"""
Inference pipeline: generates summaries using SigExt + LLM.

Orchestrates the full generation flow: load data → extract
salient sentences → generate summaries via LangChain chain.
"""

from typing import List, Dict, Optional
from tqdm.auto import tqdm


def run_inference(
    processed_data: List[Dict],
    summary_chain,
    response_separator: str = "assistant<|end_header_id|>",
) -> List[Dict]:
    """Generate summaries for preprocessed data using LLM chain.

    Args:
        processed_data: List of dicts from preprocess_dataset(),
                       each with 'source', 'reference', 'keyphrases', 'salient_sentences'.
        summary_chain: LangChain chain for summary generation.
        response_separator: String to split LLM response.

    Returns:
        List of dicts with added 'generated_summary' field.
    """
    results = []
    for item in tqdm(processed_data, desc="    Generating Summaries"):
        try:
            res = summary_chain.invoke({
                "source": item["source"],
                "keyphrases": item["keyphrases"],
            })
            gen_summary = res.split(response_separator)[-1].strip()

            results.append({
                **item,
                "generated_summary": gen_summary,
            })
        except Exception as e:
            print(f"    Generation error: {e}")
            results.append({
                **item,
                "generated_summary": "",
            })

    print(f"  Generated {len(results)} summaries ({sum(1 for r in results if r['generated_summary'])} successful)")
    return results
