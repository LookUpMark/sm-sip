"""
Inference pipeline: generates summaries using SigExt + LLM.

Orchestrates the full generation flow: load data -> extract
salient sentences -> generate summaries via LangChain chain.
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


def run_inference_pipeline(
    lang: str,
    sigext_preset: str,
    quantization: str = "8bit",
    prompt_type: str = "source_aware",
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_samples: int = 100,
    output_file: str = None,
    seed: int = 42,
) -> list:
    """End-to-end inference pipeline: data → SigExt → LLM → summaries → save.

    Args:
        lang: Language code ("it" or "en").
        sigext_preset: SigExt preset name (e.g. "xlmr-5k-060t").
        quantization: LLM quantization ("4bit" or "8bit").
        prompt_type: Prompt strategy for generation.
        llm_model_id: HuggingFace model ID for LLM.
        num_samples: Number of test samples.
        output_file: If set, save results to this path.
        seed: Random seed.

    Returns:
        List of result dicts with generated summaries.
    """
    from datetime import datetime
    from sm_sip.config import SigExtConfig
    from sm_sip.data import get_test_data
    from sm_sip.models import (
        load_sigext_model, unload_sigext_model,
        load_llm, create_summary_chain, preprocess_dataset,
    )
    from sm_sip.prompts import get_summary_prompt
    from sm_sip.utils.gpu import clear_gpu_memory
    from sm_sip.utils.io import save_results

    sc = SigExtConfig.from_preset(lang, sigext_preset)

    # 1. Load data + SigExt preprocessing (CPU)
    data = get_test_data(lang=lang, num_samples=num_samples, skip_samples=sc.skip_samples)
    sm, st = load_sigext_model(sc.model_id, device="cpu")
    processed = preprocess_dataset(data, sm, st, lang=lang)
    unload_sigext_model(sm, st)
    print(f"  Preprocessed {len(processed)} samples.")

    # 2. LLM inference (GPU)
    llm_model, llm_tokenizer, pipe = load_llm(llm_model_id, quantization, seed=seed)
    chain = create_summary_chain(pipe, get_summary_prompt(lang, prompt_type))
    results = run_inference(processed, chain)

    # 3. Save & cleanup
    if output_file:
        save_results({
            "run_info": {
                "timestamp": datetime.now().isoformat(),
                "sigext_model": sc.model_id,
                "llm_model": llm_model_id,
                "quantization": quantization,
                "prompt_type": prompt_type,
                "seed": seed,
                "num_samples": len(results),
            },
            "samples": results,
        }, output_file)
        print(f"  Results saved to {output_file}")

    del llm_model, llm_tokenizer, pipe
    clear_gpu_memory()
    print("  Done!")
    return results
