"""
Results aggregation and analysis.

Loads JSON result files from the results directory and provides
grouping, filtering, and best-config finding utilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def load_all_results(
    results_dir: str = "results",
    pattern: str = "results_*.json",
    exclude: List[str] = None,
) -> Dict[str, dict]:
    """Load all result JSON files from a directory.

    Args:
        results_dir: Path to results directory.
        pattern: Glob pattern for result files.
        exclude: Filenames to exclude (e.g. ["summary.json"]).

    Returns:
        Dict mapping result key (filename stem) to parsed JSON data.
    """
    if exclude is None:
        exclude = ["summary.json"]

    results_path = Path(results_dir)
    results = {}

    for f in sorted(results_path.glob(pattern)):
        if f.name in exclude:
            continue
        with open(f) as fp:
            data = json.load(fp)
            key = f.stem.replace("results_", "")
            results[key] = data

    print(f"  Loaded {len(results)} result files from {results_dir}")
    return results


def load_enhanced_results(filepath: str) -> Optional[dict]:
    """Load a single enhanced results JSON file.

    Args:
        filepath: Path to the enhanced results file.

    Returns:
        Parsed JSON data or None if file doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"  Enhanced results not found: {filepath}")
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(data: dict) -> Dict[str, float]:
    """Extract key metrics from a result file.

    Args:
        data: Parsed JSON result data.

    Returns:
        Dict with metric values and standard deviations.
    """
    m = data["metrics"]
    result = {}

    # Traditional metrics
    for key in ["bert_score", "bert"]:
        if key in m:
            result["bert"] = m[key]["mean"]
            result["bert_std"] = m[key]["std"]
            break

    for key in ["rouge1", "rouge"]:
        if key in m:
            result["rouge"] = m[key]["mean"]
            result["rouge_std"] = m[key]["std"]
            break

    if "kir" in m:
        result["kir"] = m["kir"]["mean"]
        result["kir_std"] = m["kir"]["std"]

    # Abstraction metrics (if present)
    for key in ["abstraction", "compression", "novel_ngrams"]:
        if key in m:
            result[key] = m[key]["mean"]

    # Judge metrics (if present)
    for key in ["judge_faithfulness", "judge_completeness", "judge_conciseness", "judge_abstraction", "judge_overall"]:
        if key in m:
            result[key] = m[key]["mean"]

    return result


def group_by(
    results: Dict[str, dict],
    dimension: str,
) -> Dict[str, List[Dict[str, float]]]:
    """Group results by a dimension from run_info.

    Args:
        results: Dict of result key -> data.
        dimension: Dimension to group by (e.g. "config", "quantization", "inference_type").

    Returns:
        Dict mapping dimension value -> list of extracted metrics.
    """
    groups = {}
    for key, data in results.items():
        if "run_info" not in data:
            continue
        value = data["run_info"].get(dimension, "unknown")
        if value not in groups:
            groups[value] = []
        groups[value].append(extract_metrics(data))

    # Print summary
    print(f"\n  BY {dimension.upper()}")
    print("  " + "-" * 50)
    for value, metrics_list in groups.items():
        bert_avg = np.mean([m.get("bert", 0) for m in metrics_list])
        rouge_avg = np.mean([m.get("rouge", 0) for m in metrics_list])
        kir_avg = np.mean([m.get("kir", 0) for m in metrics_list])
        print(f"  {value}: BERT={bert_avg:.4f} ROUGE={rouge_avg:.4f} KIR={kir_avg:.2%}")

    return groups


def find_best_configs(
    results: Dict[str, dict],
) -> Dict[str, Tuple[str, float]]:
    """Find best configurations for each metric.

    Args:
        results: Dict of result key -> data.

    Returns:
        Dict mapping metric name -> (best_config_key, best_value).
    """
    best = {}

    metrics_funcs = {
        "bert": lambda d: d.get("metrics", {}).get("bert_score", d.get("metrics", {}).get("bert", {})).get("mean", 0),
        "rouge": lambda d: d.get("metrics", {}).get("rouge1", d.get("metrics", {}).get("rouge", {})).get("mean", 0),
        "kir": lambda d: d.get("metrics", {}).get("kir", {}).get("mean", 0),
    }

    print("\n  BEST CONFIGURATIONS")
    print("  " + "-" * 50)

    for metric_name, extract_fn in metrics_funcs.items():
        best_key = max(results.keys(), key=lambda k: extract_fn(results[k]))
        best_value = extract_fn(results[best_key])
        best[metric_name] = (best_key, best_value)
        print(f"  Best {metric_name.upper()}: {best_key} = {best_value:.4f}")

    return best
