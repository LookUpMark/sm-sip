"""
Evaluation pipeline: compute all metrics on generated summaries.

Supports both basic evaluation (BERT-Score, ROUGE, KIR) and
enhanced evaluation (+ abstraction metrics + LLM-as-Judge).
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm.auto import tqdm

from sm_sip.metrics.traditional import compute_bert_score, compute_rouge, compute_kir
from sm_sip.metrics.abstraction import (
    compute_abstraction_score,
    compute_compression_ratio,
    compute_novel_ngrams,
)
from sm_sip.metrics.judge import llm_judge_evaluate


def run_evaluation(
    results: List[Dict],
    lang: str = "it",
) -> Dict:
    """Compute traditional metrics on generated summaries.

    Args:
        results: List of dicts with 'reference', 'generated_summary', 'salient_sentences'.
        lang: Language for BERT-Score.

    Returns:
        Dict with aggregated metrics.
    """
    predictions = [r["generated_summary"] for r in results]
    references = [r["reference"] for r in results]
    salient = [r.get("salient_sentences", []) for r in results]

    bert = compute_bert_score(predictions, references, lang=lang)
    rouge = compute_rouge(predictions, references, metrics=["rouge1", "rougeL"])
    kir = compute_kir(predictions, salient)

    result = {
        "bert_score": {"mean": bert["mean"], "std": bert["std"]},
        "rouge1": {"mean": rouge["rouge1"]["mean"], "std": rouge["rouge1"]["std"]},
        "rougeL": {"mean": rouge["rougeL"]["mean"], "std": rouge["rougeL"]["std"]},
        "kir": {"mean": kir["mean"], "std": kir["std"]},
    }
    return result


def run_enhanced_evaluation(
    processed_data: List[Dict],
    summary_chain,
    judge_chain=None,
    lang: str = "it",
    response_separator: str = "assistant<|end_header_id|>",
) -> Tuple[Dict, List[Dict]]:
    """Run full enhanced evaluation: generation + all metrics + judge.

    This is the main evaluation loop that:
    1. Generates summaries
    2. Computes traditional metrics (BERT-Score, ROUGE, KIR)
    3. Computes abstraction metrics (novel n-grams, compression, abstraction)
    4. Optionally runs LLM-as-Judge scoring

    Args:
        processed_data: Preprocessed data with 'source', 'reference', 'keyphrases', 'salient_sentences'.
        summary_chain: LangChain chain for summary generation.
        judge_chain: Optional LangChain chain for judge evaluation.
        lang: Language for BERT-Score and sentence processing.
        response_separator: String to split LLM response.

    Returns:
        Tuple of (metrics_dict, samples_list).
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    metrics = {
        # Traditional (bert computed in batch at end)
        "rouge1": [], "rougeL": [], "kir": [],
        # Abstraction
        "abstraction": [], "compression": [], "novel_ngrams": [],
        # LLM Judge
        "judge_faithfulness": [], "judge_completeness": [],
        "judge_conciseness": [], "judge_abstraction": [],
    }
    samples = []

    for item in tqdm(processed_data, desc="    Generating & Evaluating"):
        try:
            keys_text = item["keyphrases"]
            salient_sents = item["salient_sentences"]

            # Generate summary
            res = summary_chain.invoke({"source": item["source"], "keyphrases": keys_text})
            gen_summary = res.split(response_separator)[-1].strip()

            # === TRADITIONAL METRICS ===
            rouge_scores = scorer.score(item["reference"], gen_summary)
            metrics["rouge1"].append(rouge_scores["rouge1"].fmeasure)
            metrics["rougeL"].append(rouge_scores["rougeL"].fmeasure)

            # KIR
            kir_score = 0.0
            if salient_sents:
                gen_lower = gen_summary.lower()
                hits = 0
                for sent in salient_sents:
                    words = [w.lower() for w in sent.split() if len(w) > 4]
                    if words:
                        word_hits = sum(1 for w in words if w in gen_lower)
                        if word_hits / len(words) > 0.3:
                            hits += 1
                kir_score = hits / len(salient_sents)
            metrics["kir"].append(kir_score)

            # === ABSTRACTION METRICS ===
            abstraction = compute_abstraction_score(item["source"], gen_summary)
            compression = compute_compression_ratio(item["source"], gen_summary)
            novel = compute_novel_ngrams(item["source"], gen_summary)

            metrics["abstraction"].append(abstraction)
            metrics["compression"].append(compression)
            metrics["novel_ngrams"].append(novel)

            # === LLM JUDGE ===
            judge_scores = {}
            if judge_chain is not None:
                judge_scores = llm_judge_evaluate(
                    item["source"], gen_summary, item["reference"], judge_chain
                )
                metrics["judge_faithfulness"].append(judge_scores.get("faithfulness", 3))
                metrics["judge_completeness"].append(judge_scores.get("completeness", 3))
                metrics["judge_conciseness"].append(judge_scores.get("conciseness", 3))
                metrics["judge_abstraction"].append(judge_scores.get("abstraction", 3))

            # Store sample details (bert_score filled after loop)
            samples.append({
                "source": item["source"][:500] + "..." if len(item["source"]) > 500 else item["source"],
                "reference": item["reference"],
                "salient_sentences": salient_sents,
                "generated_summary": gen_summary,
                "scores": {
                    "rouge1": float(rouge_scores["rouge1"].fmeasure),
                    "rougeL": float(rouge_scores["rougeL"].fmeasure),
                    "kir": float(kir_score),
                    "abstraction": float(abstraction),
                    "compression": float(compression),
                    "novel_ngrams": float(novel),
                    "judge": judge_scores,
                },
            })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # === BERT-Score in batch on CPU (avoids GPU VRAM contention with LLM) ===
    if samples:
        from bert_score import score as bert_score_fn
        import torch
        predictions = [s["generated_summary"] for s in samples]
        references = [s["reference"] for s in samples]
        print("  Computing BERTScore on CPU (batch)...")
        with torch.no_grad():
            _, _, F1 = bert_score_fn(
                predictions, references, lang=lang, verbose=False, device="cpu"
            )
        bert_scores = F1.numpy().tolist()
        for i, sc in enumerate(bert_scores):
            samples[i]["scores"]["bert"] = float(sc)
    else:
        bert_scores = []

    # Aggregate metrics
    aggregated = {}
    for key, values in metrics.items():
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    if bert_scores:
        aggregated["bert"] = {
            "mean": float(np.mean(bert_scores)),
            "std": float(np.std(bert_scores)),
        }

    return aggregated, samples
