#!/usr/bin/env python3
"""
Unified Results Analysis for SM-SIP

Analyzes results for both Italian (WITS) and English (ArXiv) extensions.

Usage:
    python scripts/analyze_results.py                    # Analyze all
    python scripts/analyze_results.py --lang it          # Italian only
    python scripts/analyze_results.py --lang en          # English only
    python scripts/analyze_results.py --latex             # Include LaTeX tables
"""

import argparse
from sm_sip.analysis.aggregator import load_all_results, load_enhanced_results, group_by, find_best_configs
from sm_sip.analysis.latex import generate_table, generate_enhanced_table


def analyze_italian(results_dir: str = "results/italian/base", enhanced_dir: str = "results/italian/decoding", show_latex: bool = False):
    """Run analysis on Italian WITS results."""
    print("=" * 60)
    print("SM-SIP ITALIAN EXTENSION - Results Analysis")
    print("=" * 60)

    results = load_all_results(results_dir)
    if not results:
        print("  No results found.")
        return

    # Group by dimensions
    group_by(results, "config")
    group_by(results, "quantization")
    group_by(results, "inference_type")

    # Best configs
    find_best_configs(results)

    # LaTeX tables
    if show_latex:
        print("\n  LATEX TABLES")
        print("  " + "-" * 50)
        print(generate_table(results, "inference_type", "few-shot", "Few-Shot Results"))
        print()
        print(generate_table(results, "inference_type", "zero-shot", "Zero-Shot Results"))

    # Enhanced results
    for name in ["greedy.json", "temp_0.1.json"]:
        data = load_enhanced_results(f"{enhanced_dir.rstrip('/')}/{name}")
        if data:
            print(f"\n  ENHANCED RESULTS ({name})")
            print("  " + "-" * 50)
            if show_latex:
                print(generate_enhanced_table(data, f"Enhanced - {name}"))


def analyze_english(results_path: str = "results/english/arxiv_enhanced.json", show_latex: bool = False):
    """Run analysis on English ArXiv results."""
    print("=" * 60)
    print("SM-SIP ENGLISH ARXIV EXTENSION - Results Analysis")
    print("=" * 60)

    data = load_enhanced_results(results_path)
    if not data:
        return

    m = data["metrics"]
    print(f"\n  Samples: {data['run_info']['num_samples']}")
    print(f"  Prompt:  {data['run_info']['prompt_type']}")

    print("\n  Traditional:")
    for key, label in [("bert", "BERT"), ("rouge1", "ROUGE-1"), ("rouge", "ROUGE"), ("kir", "KIR")]:
        if key in m:
            fmt = ".2%" if key == "kir" else ".4f"
            print(f"    {label}: {m[key]['mean']:{fmt}}")

    print("\n  Abstraction:")
    for key, label in [("abstraction", "Score"), ("novel_ngrams", "Novel"), ("compression", "Compression")]:
        if key in m:
            print(f"    {label}: {m[key]['mean']:.4f}")

    print("\n  LLM Judge:")
    for key, label in [
        ("judge_faithfulness", "Faithfulness"),
        ("judge_completeness", "Completeness"),
        ("judge_conciseness", "Conciseness"),
        ("judge_abstraction", "Abstraction"),
        ("judge_overall", "Overall"),
    ]:
        if key in m:
            print(f"    {label}: {m[key]['mean']:.2f}/5")

    if show_latex:
        print(generate_enhanced_table(data, "ArXiv Enhanced Results"))


def main():
    parser = argparse.ArgumentParser(description="SM-SIP Results Analysis")
    parser.add_argument("--lang", choices=["it", "en", "all"], default="all", help="Language to analyze")
    parser.add_argument("--latex", action="store_true", help="Include LaTeX tables")
    args = parser.parse_args()

    if args.lang in ("it", "all"):
        analyze_italian(show_latex=args.latex)
    if args.lang in ("en", "all"):
        analyze_english(show_latex=args.latex)

    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
