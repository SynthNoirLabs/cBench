from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev

from cbench.config import LONG_CONTEXT_THRESHOLD, PRICING, ModelID


def calculate_cost(model_id: ModelID, usage: dict) -> float:
    """Calculate the API cost for a given model and usage.

    Args:
        model_id: The model that was used.
        usage: Dict with keys input_tokens, output_tokens, and optionally
               cache_creation_input_tokens and cache_read_input_tokens.

    Returns:
        Total cost in USD as a float.
    """
    pricing = PRICING[model_id]

    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
    cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_mtok"]

    cache_write_cost = (
        (cache_creation_input_tokens / 1_000_000)
        * pricing["input_per_mtok"]
        * pricing["cache_write_multiplier"]
    )
    cache_read_cost = (
        (cache_read_input_tokens / 1_000_000)
        * pricing["input_per_mtok"]
        * pricing["cache_read_multiplier"]
    )

    total = input_cost + output_cost + cache_write_cost + cache_read_cost

    if input_tokens > LONG_CONTEXT_THRESHOLD:
        long_input_mult = pricing["long_context_input_multiplier"]
        long_output_mult = pricing["long_context_output_multiplier"]
        total = (
            input_cost + cache_write_cost + cache_read_cost
        ) * long_input_mult + output_cost * long_output_mult

    return total


def compute_reliability_metrics(results: list[dict]) -> dict[tuple[str, str], dict]:
    """Compute reliability metrics grouped by (variant_name, task_name).

    Returns a dict keyed by (variant, task) with:
        - mean: average score
        - stddev: standard deviation of scores
        - pass_k: fraction of runs with score >= 0.5
        - num_runs: number of runs
        - scores: list of individual scores
        - mean_cost: average cost per call
        - mean_latency: average wall-clock seconds
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for r in results:
        # Support both dict and dataclass
        if hasattr(r, "__dataclass_fields__"):
            from dataclasses import asdict

            r = asdict(r)
        key = (r.get("variant_name", ""), r.get("task_name", ""))
        groups[key].append(r)

    metrics = {}
    for key, group in groups.items():
        scores = [r.get("score", 0.0) for r in group]
        costs = [r.get("call_result", {}).get("cost_usd", 0.0) for r in group]
        latencies = [r.get("call_result", {}).get("wall_clock_seconds", 0.0) for r in group]

        k = len(scores)
        metrics[key] = {
            "mean": mean(scores),
            "stddev": stdev(scores) if k > 1 else 0.0,
            "pass_k": sum(1 for s in scores if s >= 0.5) / k,
            "num_runs": k,
            "scores": scores,
            "mean_cost": mean(costs),
            "mean_latency": mean(latencies),
        }

    return metrics
