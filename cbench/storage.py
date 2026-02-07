from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def save_results(benchmark_name: str, results: list, output_dir: str = "results") -> Path:
    """Save results to JSON and CSV in results/{benchmark_name}/{timestamp}/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(output_dir) / benchmark_name / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to serializable dicts
    serializable = []
    for r in results:
        d = asdict(r) if hasattr(r, "__dataclass_fields__") else r
        # Convert any non-serializable values
        serializable.append(_make_serializable(d))

    # Save JSON
    json_path = result_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Save CSV
    if serializable:
        csv_path = result_dir / "results.csv"
        # Flatten nested dicts for CSV
        flat_results = [_flatten_dict(d) for d in serializable]
        fieldnames = list(flat_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)

    return result_dir


def load_results(path: str | Path) -> list[dict]:
    """Load results from a JSON file or directory containing results.json."""
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    with open(p) as f:
        return json.load(f)


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return _make_serializable(asdict(obj))
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items
