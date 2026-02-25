"""
Ablation study pipeline.

Provides a unified function for running ablation experiments across
different axes (thresholds, base models, quantization, prompts,
decoding, judge models, cross-lingual).
"""

from typing import Dict, List, Optional
from sm_sip.config import SigExtConfig
from sm_sip.data import get_test_data
from sm_sip.models import (
    load_sigext_model, unload_sigext_model,
    load_llm, create_summary_chain, preprocess_dataset,
)
from sm_sip.prompts import get_summary_prompt
from sm_sip.pipelines.inference import run_inference
from sm_sip.pipelines.evaluation import run_evaluation
from sm_sip.utils.gpu import clear_gpu_memory
from sm_sip.utils.io import save_results


def run_ablation_experiment(
    presets: Dict[str, List[str]],
    num_samples: int = 50,
    quantization: str = "8bit",
    prompt_type: str = "source_aware",
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    temperatures: Optional[List[float]] = None,
    judge_models: Optional[List[str]] = None,
    bert_model_type: Optional[str] = None,
    output_file: Optional[str] = None,
    ablation_name: str = "default",
) -> Dict:
    """Run an ablation experiment across languages and presets.

    Handles the full loop: load data → SigExt → LLM → inference → evaluation.
    Supports all ablation axes via optional parameters.

    Args:
        presets: Mapping of language to list of SigExt preset names.
                 Example: {"it": ["xlmr-5k-060t"], "en": ["xlmr-5k-060t"]}
        num_samples: Number of test samples per language.
        quantization: LLM quantization level ("4bit" or "8bit").
        prompt_type: Prompt type for summary generation.
        llm_model_id: HuggingFace model ID for the LLM.
        temperatures: If set, run decoding ablation across these temperatures.
        judge_models: If set, run judge model ablation across these models.
        bert_model_type: If set, also compute BERTScore with this unified model.
        output_file: If set, save results to this path.
        ablation_name: Name tag for the ablation study.

    Returns:
        Dict of results keyed by experiment identifier.
    """
    results = {}

    for lang, preset_list in presets.items():
        print(f'\n{"=" * 60}')
        print(f'  Language: {lang.upper()}')
        print(f'{"=" * 60}')

        data = get_test_data(
            lang=lang, num_samples=num_samples,
            skip_samples=SigExtConfig.LANG_DATASETS[lang]["skip"],
        )

        # --- Decoding temperature ablation ---
        if temperatures is not None:
            sc = SigExtConfig.from_preset(lang, preset_list[0])
            sm, st = load_sigext_model(sc.model_id, device="cpu")
            proc = preprocess_dataset(data, sm, st, lang=lang)
            unload_sigext_model(sm, st)

            for temp in temperatures:
                key = f"{lang}_greedy" if temp == 0 else f"{lang}_temp_{temp}"
                print(f'\n  -> {key}')
                _, _, pipe = load_llm(
                    llm_model_id, quantization,
                    temperature=temp, do_sample=temp > 0,
                )
                chain = create_summary_chain(pipe, get_summary_prompt(lang, prompt_type))
                res = run_inference(proc, chain)
                metrics = run_evaluation(res, lang=lang)
                # Add abstraction score
                from sm_sip.metrics.abstraction import compute_abstraction_score
                import numpy as np
                metrics["abstraction"] = {
                    "mean": float(np.mean([
                        compute_abstraction_score(r["source"], r["generated_summary"])
                        for r in res
                    ]))
                }
                results[key] = metrics
                clear_gpu_memory()
            continue

        # --- Judge model ablation ---
        if judge_models is not None:
            sc = SigExtConfig.from_preset(lang, preset_list[0])
            sm, st = load_sigext_model(sc.model_id, device="cpu")
            proc = preprocess_dataset(data, sm, st, lang=lang)
            unload_sigext_model(sm, st)

            from sm_sip.models import create_judge_chain, create_remote_judge_chain, load_remote_llm
            from sm_sip.prompts import get_judge_prompt
            from sm_sip.metrics.judge import llm_judge_evaluate

            _, _, pipe = load_llm(llm_model_id, quantization)
            chain = create_summary_chain(pipe, get_summary_prompt(lang, prompt_type))
            res = run_inference(proc, chain)
            clear_gpu_memory()

            for judge_id in judge_models:
                key = f"{lang}_{judge_id.split('/')[-1]}"
                print(f'\n  -> Judge: {key}')
                
                # Heuristic for remote models (OpenRouter/OpenAI)
                is_remote = any(x in judge_id.lower() for x in ["gpt-", "claude", "gemini", "deepseek", "openai/", "anthropic/"])
                
                if is_remote:
                    jllm = load_remote_llm(judge_id)
                    jchain = create_remote_judge_chain(jllm, get_judge_prompt(lang))
                else:
                    _, _, jpipe = load_llm(judge_id, quantization)
                    jchain = create_judge_chain(jpipe, get_judge_prompt(lang))

                scores = {"faithfulness": [], "completeness": [], "conciseness": [], "abstraction": []}
                from tqdm.auto import tqdm
                for r in tqdm(res, desc=f"    Judging ({key})"):
                    try:
                        js = llm_judge_evaluate(r["source"], r["generated_summary"], r["reference"], jchain)
                        for dim in scores:
                            scores[dim].append(js.get(dim, 3))
                    except Exception as e:
                        print(f"    Judge error: {e}")

                import numpy as np
                results[key] = {dim: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} for dim, vals in scores.items() if vals}
                clear_gpu_memory()
            continue

        # --- Standard ablation (presets, thresholds, base models, quantization, prompts) ---
        _, _, pipe = load_llm(llm_model_id, quantization)
        chain = create_summary_chain(pipe, get_summary_prompt(lang, prompt_type))

        for preset in preset_list:
            key = f"{lang}_{preset}"
            print(f'\n  -> {key}')

            try:
                sc = SigExtConfig.from_preset(lang, preset)
                sm, st = load_sigext_model(sc.model_id, device="cpu")
                proc = preprocess_dataset(data, sm, st, lang=lang)
                unload_sigext_model(sm, st)

                res = run_inference(proc, chain)
                metrics = run_evaluation(res, lang=lang)
                results[key] = metrics

                # Optional: unified BERTScore for cross-lingual comparison
                if bert_model_type:
                    metrics_unified = run_evaluation(res, lang=lang, bert_model_type=bert_model_type)
                    results[key] = {
                        "default_bert": metrics,
                        "unified_bert": metrics_unified,
                    }

                print(f'    BERTScore: {metrics.get("bert_score", {}).get("mean", 0):.4f}  '
                      f'ROUGE-1: {metrics.get("rouge1", {}).get("mean", 0):.4f}  '
                      f'KIR: {metrics.get("kir", {}).get("mean", 0):.4f}')

            except Exception as e:
                print(f'    FAILED: {e}')
                results[key] = {"error": str(e)}

        clear_gpu_memory()

    # Save results
    if output_file:
        save_results({"ablation": ablation_name, "results": results}, output_file)
        print(f'\nResults saved to {output_file}')

    return results
