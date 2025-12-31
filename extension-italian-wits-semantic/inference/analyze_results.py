#!/usr/bin/env python3
"""
Results Analysis for SM-SIP Italian Extension
Analyzes all inference results and generates summary statistics.
"""

import json
import os
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results")
ENHANCED_DIR = Path("results_enhanced")

def load_all_results():
    """Load all result files from results directory."""
    results = {}
    for f in sorted(RESULTS_DIR.glob("results_*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as file:
            data = json.load(file)
            key = f.stem.replace("results_", "")
            results[key] = data
    return results

def extract_metrics(data):
    """Extract key metrics from a result file."""
    m = data["metrics"]
    return {
        "bert": m["bert_score"]["mean"],
        "rouge": m["rouge1"]["mean"],
        "kir": m["kir"]["mean"],
        "bert_std": m["bert_score"]["std"],
        "rouge_std": m["rouge1"]["std"],
        "kir_std": m["kir"]["std"],
    }

def analyze_by_dimension(results):
    """Analyze results by different dimensions."""
    
    # Group by config
    configs = {"10k-60t": [], "25k-60t": [], "25k-65t": [], "25k-70t": []}
    for key, data in results.items():
        config = data["run_info"]["config"]
        configs[config].append(extract_metrics(data))
    
    print("\n[1] BY TRAINING CONFIG")
    print("-" * 50)
    for config, metrics_list in configs.items():
        bert_avg = np.mean([m["bert"] for m in metrics_list])
        rouge_avg = np.mean([m["rouge"] for m in metrics_list])
        kir_avg = np.mean([m["kir"] for m in metrics_list])
        print(f"{config}: BERT={bert_avg:.4f} ROUGE={rouge_avg:.4f} KIR={kir_avg:.2%}")
    
    # Group by quantization
    quants = {"4bit": [], "8bit": []}
    for key, data in results.items():
        quant = data["run_info"]["quantization"]
        quants[quant].append(extract_metrics(data))
    
    print("\n[2] BY QUANTIZATION")
    print("-" * 50)
    for quant, metrics_list in quants.items():
        bert_avg = np.mean([m["bert"] for m in metrics_list])
        rouge_avg = np.mean([m["rouge"] for m in metrics_list])
        kir_avg = np.mean([m["kir"] for m in metrics_list])
        print(f"{quant}: BERT={bert_avg:.4f} ROUGE={rouge_avg:.4f} KIR={kir_avg:.2%}")
    
    # Group by inference type
    shots = {"zero-shot": [], "few-shot": []}
    for key, data in results.items():
        shot = data["run_info"]["inference_type"]
        shots[shot].append(extract_metrics(data))
    
    print("\n[3] BY INFERENCE TYPE")
    print("-" * 50)
    for shot, metrics_list in shots.items():
        bert_avg = np.mean([m["bert"] for m in metrics_list])
        rouge_avg = np.mean([m["rouge"] for m in metrics_list])
        kir_avg = np.mean([m["kir"] for m in metrics_list])
        print(f"{shot}: BERT={bert_avg:.4f} ROUGE={rouge_avg:.4f} KIR={kir_avg:.2%}")

def find_best_configs(results):
    """Find best configurations for each metric."""
    print("\n[4] BEST CONFIGURATIONS")
    print("-" * 50)
    
    best_bert = max(results.items(), key=lambda x: x[1]["metrics"]["bert_score"]["mean"])
    best_rouge = max(results.items(), key=lambda x: x[1]["metrics"]["rouge1"]["mean"])
    best_kir = max(results.items(), key=lambda x: x[1]["metrics"]["kir"]["mean"])
    
    print(f"Best BERT:  {best_bert[0]} = {best_bert[1]['metrics']['bert_score']['mean']:.4f}")
    print(f"Best ROUGE: {best_rouge[0]} = {best_rouge[1]['metrics']['rouge1']['mean']:.4f}")
    print(f"Best KIR:   {best_kir[0]} = {best_kir[1]['metrics']['kir']['mean']:.2%}")

def generate_latex_tables(results):
    """Generate LaTeX tables for paper."""
    print("\n[5] LATEX TABLES")
    print("-" * 50)
    
    # Few-shot table
    print("\n% Few-Shot Results")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Config} & \\textbf{Quant} & \\textbf{BERT} & \\textbf{ROUGE-1} & \\textbf{KIR} \\\\")
    print("\\midrule")
    
    few_shot = [(k, v) for k, v in results.items() if "few_shot" in k]
    for key, data in sorted(few_shot):
        m = extract_metrics(data)
        config = data["run_info"]["config"]
        quant = data["run_info"]["quantization"]
        print(f"{config} & {quant} & {m['bert']:.4f} & {m['rouge']:.4f} & {m['kir']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    # Zero-shot table
    print("\n% Zero-Shot Results")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Config} & \\textbf{Quant} & \\textbf{BERT} & \\textbf{ROUGE-1} & \\textbf{KIR} \\\\")
    print("\\midrule")
    
    zero_shot = [(k, v) for k, v in results.items() if "zero_shot" in k]
    for key, data in sorted(zero_shot):
        m = extract_metrics(data)
        config = data["run_info"]["config"]
        quant = data["run_info"]["quantization"]
        print(f"{config} & {quant} & {m['bert']:.4f} & {m['rouge']:.4f} & {m['kir']:.4f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

def analyze_enhanced_results():
    """Analyze enhanced results with abstraction metrics."""
    enhanced_file = ENHANCED_DIR / "results_enhanced.json"
    if not enhanced_file.exists():
        print("\n[6] ENHANCED RESULTS: Not found")
        return
    
    with open(enhanced_file) as f:
        data = json.load(f)
    
    print("\n[6] ENHANCED RESULTS (Optimized Prompt)")
    print("-" * 50)
    
    m = data["metrics"]
    print(f"Samples: {data['run_info']['num_samples']}")
    print(f"Prompt:  {data['run_info']['prompt_type']}")
    print()
    print("Traditional:")
    print(f"  BERT:  {m['bert']['mean']:.4f} +/- {m['bert']['std']:.4f}")
    print(f"  ROUGE: {m['rouge']['mean']:.4f} +/- {m['rouge']['std']:.4f}")
    print(f"  KIR:   {m['kir']['mean']:.2%}")
    print()
    print("Abstraction:")
    print(f"  Score: {m['abstraction']['mean']:.4f}")
    print(f"  Novel: {m['novel_ngrams']['mean']:.2%}")
    print(f"  Compression: {m['compression']['mean']:.2%}")
    print()
    print("LLM Judge:")
    print(f"  Faithfulness: {m['judge_faithfulness']['mean']:.2f}/5")
    print(f"  Completeness: {m['judge_completeness']['mean']:.2f}/5")
    print(f"  Conciseness:  {m['judge_conciseness']['mean']:.2f}/5")
    print(f"  Abstraction:  {m['judge_abstraction']['mean']:.2f}/5")
    print(f"  Overall:      {m['judge_overall']['mean']:.2f}/5")

def main():
    print("=" * 60)
    print("SM-SIP Italian Extension - Results Analysis")
    print("=" * 60)
    
    results = load_all_results()
    print(f"\nLoaded {len(results)} result files")
    
    analyze_by_dimension(results)
    find_best_configs(results)
    generate_latex_tables(results)
    analyze_enhanced_results()
    
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
