"""
JSON I/O utilities for saving and loading results.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def save_results(
    data: Dict[str, Any],
    output_path: str,
    pretty: bool = True,
) -> str:
    """Save results to a JSON file.

    Creates parent directories if needed. Adds a timestamp
    if not already present in the data.

    Args:
        data: Results dict to save.
        output_path: Output file path.
        pretty: Whether to format JSON with indentation.

    Returns:
        Absolute path to the saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if "run_info" in data and "timestamp" not in data["run_info"]:
        data["run_info"]["timestamp"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(data, f, indent=2 if pretty else None, ensure_ascii=False)

    print(f"  Results saved: {path}")
    return str(path.absolute())


def load_results(filepath: str) -> Optional[Dict]:
    """Load results from a JSON file.

    Args:
        filepath: Path to JSON file.

    Returns:
        Parsed JSON data, or None if file doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"  Results file not found: {filepath}")
        return None

    with open(path) as f:
        data = json.load(f)

    return data
