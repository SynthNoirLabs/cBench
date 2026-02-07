from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_charts(results: list[dict], output_path: str | Path) -> Path:
    output_dir = Path(output_path)
    if output_dir.is_file():
        output_dir = output_dir.parent
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    _score_by_variant(results, charts_dir)
    _cost_vs_score(results, charts_dir)
    _latency_by_variant(results, charts_dir)
    _tokens_by_variant(results, charts_dir)

    return charts_dir


def _score_by_variant(results: list[dict], output_dir: Path) -> None:
    scores = defaultdict(list)
    for r in results:
        scores[r.get("variant_name", "unknown")].append(r.get("score", 0))

    if not scores:
        return

    variants = list(scores.keys())
    means = [np.mean(v) for v in scores.values()]
    stds = [np.std(v) for v in scores.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(variants, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Average Score by Variant")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "score_by_variant.png", dpi=150)
    plt.close(fig)


def _cost_vs_score(results: list[dict], output_dir: Path) -> None:
    costs = defaultdict(list)
    scores = defaultdict(list)
    for r in results:
        variant = r.get("variant_name", "unknown")
        call = r.get("call_result", {})
        costs[variant].append(call.get("cost_usd", 0))
        scores[variant].append(r.get("score", 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in costs:
        ax.scatter(
            np.mean(costs[variant]),
            np.mean(scores[variant]),
            s=100,
            label=variant,
        )
    ax.set_xlabel("Average Cost ($)")
    ax.set_ylabel("Average Score")
    ax.set_title("Cost vs Accuracy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "cost_vs_score.png", dpi=150)
    plt.close(fig)


def _latency_by_variant(results: list[dict], output_dir: Path) -> None:
    latencies = defaultdict(list)
    for r in results:
        call = r.get("call_result", {})
        latencies[r.get("variant_name", "unknown")].append(call.get("wall_clock_seconds", 0))

    if not latencies:
        return

    variants = list(latencies.keys())
    means = [np.mean(v) for v in latencies.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(variants, means, color="coral", alpha=0.8)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Average Latency by Variant")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "latency_by_variant.png", dpi=150)
    plt.close(fig)


def _tokens_by_variant(results: list[dict], output_dir: Path) -> None:
    input_tok = defaultdict(list)
    output_tok = defaultdict(list)
    for r in results:
        call = r.get("call_result", {})
        variant = r.get("variant_name", "unknown")
        input_tok[variant].append(call.get("input_tokens", 0))
        output_tok[variant].append(call.get("output_tokens", 0))

    if not input_tok:
        return

    variants = list(input_tok.keys())
    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        [np.mean(input_tok[v]) for v in variants],
        width,
        label="Input",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        [np.mean(output_tok[v]) for v in variants],
        width,
        label="Output",
        color="coral",
    )
    ax.set_ylabel("Tokens")
    ax.set_title("Average Token Usage by Variant")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "tokens_by_variant.png", dpi=150)
    plt.close(fig)
