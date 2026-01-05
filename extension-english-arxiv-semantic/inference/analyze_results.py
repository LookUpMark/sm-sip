#!/usr/bin/env python3
"""
Results Analysis for English ArXiv Extension
Analyzes inference results and generates summary statistics.
"""

import json
import os
from pathlib import Path
import numpy as np

ENHANCED_DIR = Path("results_enhanced")

def analyze_enhanced_results():
    """Analyze enhanced results with abstraction metrics."""
    enhanced_file = ENHANCED_DIR / "results_enhanced.json"
    if not enhanced_file.exists():
        print(f"\n[ERROR] Results file not found at: {enhanced_file}")
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
    print(f"  BERT:    {m['bert']['mean']:.4f} +/- {m['bert']['std']:.4f}")
    if "rouge1" in m:
        print(f"  ROUGE-1: {m['rouge1']['mean']:.4f} +/- {m['rouge1']['std']:.4f}")
    if "rouge" in m:
        print(f"  ROUGE:   {m['rouge']['mean']:.4f} +/- {m['rouge']['std']:.4f}")
    print(f"  KIR:     {m['kir']['mean']:.2%}")
    print()
    print("Abstraction:")
    print(f"  Score:       {m['abstraction']['mean']:.4f}")
    print(f"  Novel:       {m['novel_ngrams']['mean']:.2%}")
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
    print("SM-SIP English ArXiv Extension - Results Analysis")
    print("=" * 60)
    
    analyze_enhanced_results()
    
    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
