"""
Visualization utilities for results comparison.

Generates matplotlib plots for metric comparisons across
different experimental dimensions (config, quantization, etc.).
"""

from typing import Dict, List, Optional
import numpy as np


def plot_comparison(
    groups: Dict[str, List[Dict[str, float]]],
    metric: str = "bert",
    title: str = "Comparison",
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Create a bar chart comparing a metric across groups.

    Args:
        groups: Dict from group_by(), mapping group name → list of metrics dicts.
        metric: Metric key to plot (e.g. "bert", "rouge", "kir").
        title: Plot title.
        ylabel: Y-axis label (defaults to metric name).
        save_path: If provided, save figure to this path.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    group_names = list(groups.keys())
    means = []
    stds = []

    for name in group_names:
        values = [m.get(metric, 0) for m in groups[name]]
        means.append(np.mean(values))
        stds.append(np.std(values))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(group_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, edgecolor="black")

    ax.set_xlabel("Configuration")
    ax.set_ylabel(ylabel or metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    return fig


def plot_multi_metric(
    groups: Dict[str, List[Dict[str, float]]],
    metrics: List[str] = None,
    title: str = "Multi-Metric Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """Create a grouped bar chart comparing multiple metrics across groups.

    Args:
        groups: Dict from group_by().
        metrics: List of metric keys to plot.
        title: Plot title.
        save_path: If provided, save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["bert", "rouge", "kir"]

    group_names = list(groups.keys())
    n_groups = len(group_names)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_groups)
    width = 0.8 / n_metrics

    for i, metric in enumerate(metrics):
        means = [np.mean([m.get(metric, 0) for m in groups[name]]) for name in group_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=metric.upper(), alpha=0.8, edgecolor="black")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    return fig
