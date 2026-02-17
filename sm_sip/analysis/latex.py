"""
LaTeX table generation from results.

Generates formatted LaTeX tables for the paper, supporting
grouping by configuration, quantization, and inference type.
"""

from typing import Dict, List, Optional
from sm_sip.analysis.aggregator import extract_metrics


def generate_table(
    results: Dict[str, dict],
    group_filter: Optional[str] = None,
    filter_value: Optional[str] = None,
    caption: str = "Results",
) -> str:
    """Generate a LaTeX table from results.

    Args:
        results: Dict of result key -> data.
        group_filter: Optional filter dimension (e.g. "inference_type").
        filter_value: Value to filter on (e.g. "few-shot").
        caption: Table caption.

    Returns:
        LaTeX table string.
    """
    # Filter results if needed
    filtered = {}
    for key, data in results.items():
        if group_filter and filter_value:
            if data.get("run_info", {}).get(group_filter) != filter_value:
                continue
        filtered[key] = data

    # Build table
    lines = []
    lines.append(f"% {caption}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Config} & \\textbf{Quant} & \\textbf{BERT} & \\textbf{ROUGE-1} & \\textbf{KIR} \\\\")
    lines.append("\\midrule")

    for key, data in sorted(filtered.items()):
        m = extract_metrics(data)
        config = data.get("run_info", {}).get("config", key)
        quant = data.get("run_info", {}).get("quantization", "—")
        bert = m.get("bert", 0)
        rouge = m.get("rouge", 0)
        kir = m.get("kir", 0)
        lines.append(f"{config} & {quant} & {bert:.4f} & {rouge:.4f} & {kir:.4f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def generate_enhanced_table(data: dict, caption: str = "Enhanced Results") -> str:
    """Generate LaTeX table for enhanced results with all metrics.

    Args:
        data: Enhanced results JSON data.
        caption: Table caption.

    Returns:
        LaTeX table string.
    """
    m = data["metrics"]

    lines = []
    lines.append(f"% {caption}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} \\\\")
    lines.append("\\midrule")

    # Traditional
    for key, label in [("bert", "BERT-F1"), ("rouge1", "ROUGE-1"), ("rouge", "ROUGE"), ("kir", "KIR")]:
        if key in m:
            lines.append(f"{label} & {m[key]['mean']:.4f} & {m[key]['std']:.4f} \\\\")

    lines.append("\\midrule")

    # Abstraction
    for key, label in [("abstraction", "Abstraction"), ("novel_ngrams", "Novel n-grams"), ("compression", "Compression")]:
        if key in m:
            lines.append(f"{label} & {m[key]['mean']:.4f} & — \\\\")

    lines.append("\\midrule")

    # Judge
    for key, label in [
        ("judge_faithfulness", "Faithfulness"),
        ("judge_completeness", "Completeness"),
        ("judge_conciseness", "Conciseness"),
        ("judge_abstraction", "Abstraction (Judge)"),
        ("judge_overall", "Overall"),
    ]:
        if key in m:
            lines.append(f"{label} & {m[key]['mean']:.2f}/5 & — \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)
