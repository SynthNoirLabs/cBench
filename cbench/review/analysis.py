"""Review-specific analysis: markdown report and comparison charts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_review_report(results: list[dict], output_path: str | Path) -> Path:
    """Generate a markdown report from review results."""
    output_dir = Path(output_path)
    if output_dir.is_file():
        output_dir = output_dir.parent
    report_path = output_dir / "review_report.md"

    lines = [
        "# Repo Review Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total reviews: {len(results)}",
        "",
        "## Summary",
        "",
        "| Variant | Runs | Avg Score | Avg Turns | Avg Tool Calls | Avg Files | Avg Cost ($) | Avg Time (s) |",
        "|---------|------|-----------|-----------|----------------|-----------|-------------|-------------|",
    ]

    by_variant = defaultdict(list)
    for r in results:
        by_variant[r.get("variant_name", "unknown")].append(r)

    for variant, vresults in by_variant.items():
        scores = [r.get("score", 0) for r in vresults]
        metrics_list = [r.get("agent_metrics", {}) for r in vresults]
        turns = [m.get("total_turns", 0) for m in metrics_list]
        tool_calls = [m.get("tool_calls", 0) for m in metrics_list]
        files_read = [m.get("files_read", 0) for m in metrics_list]
        calls = [r.get("call_result", {}) for r in vresults]
        costs = [c.get("cost_usd", 0) for c in calls]
        times = [r.get("wall_clock_seconds", 0) for r in vresults]
        n = len(vresults)

        lines.append(
            f"| {variant} | {n} "
            f"| {sum(scores) / n:.3f} "
            f"| {sum(turns) / n:.1f} "
            f"| {sum(tool_calls) / n:.1f} "
            f"| {sum(files_read) / n:.1f} "
            f"| {sum(costs) / n:.4f} "
            f"| {sum(times) / n:.1f} |"
        )

    lines.append("")

    # Dimension breakdown
    lines.append("## Score Dimensions")
    lines.append("")
    lines.append(
        "| Variant | Completeness | Accuracy | Depth | Actionability | Prioritization | Composite |"
    )
    lines.append(
        "|---------|-------------|----------|-------|--------------|---------------|-----------|"
    )

    for variant, vresults in by_variant.items():
        judge_list = [r.get("judge_scores", {}) for r in vresults]
        n = len(vresults)
        dims = {}
        for dim in ["completeness", "accuracy", "depth", "actionability", "prioritization"]:
            dims[dim] = sum(j.get(dim, 0) for j in judge_list) / n
        composite = sum(r.get("score", 0) for r in vresults) / n

        lines.append(
            f"| {variant} "
            f"| {dims['completeness']:.2f} "
            f"| {dims['accuracy']:.2f} "
            f"| {dims['depth']:.2f} "
            f"| {dims['actionability']:.2f} "
            f"| {dims['prioritization']:.2f} "
            f"| {composite:.2f} |"
        )

    lines.append("")

    # Per-run details
    lines.append("## Per-Run Details")
    lines.append("")

    for r in results:
        variant = r.get("variant_name", "unknown")
        run_idx = r.get("run_index", 0)
        score = r.get("score", 0)
        metrics = r.get("agent_metrics", {})
        judge = r.get("judge_scores", {})

        lines.append(f"### {variant} (run {run_idx})")
        lines.append("")
        lines.append(f"- **Score**: {score:.2f}")
        lines.append(f"- **Turns**: {metrics.get('total_turns', 0)}")
        lines.append(f"- **Tool calls**: {metrics.get('tool_calls', 0)}")
        lines.append(f"- **Files read**: {metrics.get('files_read', 0)}")
        lines.append(f"- **Judge reasoning**: {judge.get('judge_reasoning', 'N/A')}")
        lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def generate_review_charts(results: list[dict], output_path: str | Path) -> Path:
    """Generate comparison charts from review results."""
    output_dir = Path(output_path)
    if output_dir.is_file():
        output_dir = output_dir.parent
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    by_variant = defaultdict(list)
    for r in results:
        by_variant[r.get("variant_name", "unknown")].append(r)

    _score_radar(by_variant, charts_dir)
    _score_comparison(by_variant, charts_dir)
    _cost_vs_score(by_variant, charts_dir)
    _exploration_depth(by_variant, charts_dir)

    return charts_dir


def _score_radar(by_variant: dict, output_dir: Path) -> None:
    """Radar chart comparing dimension scores across variants."""
    dimensions = ["completeness", "accuracy", "depth", "actionability", "prioritization"]
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for variant, vresults in by_variant.items():
        judge_list = [r.get("judge_scores", {}) for r in vresults]
        n = len(vresults)
        values = [sum(j.get(d, 0) for j in judge_list) / n for d in dimensions]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=variant)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.capitalize() for d in dimensions])
    ax.set_ylim(0, 1)
    ax.set_title("Review Quality by Dimension", size=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(output_dir / "review_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _score_comparison(by_variant: dict, output_dir: Path) -> None:
    """Bar chart of composite scores by variant."""
    variants = list(by_variant.keys())
    scores = []
    stds = []
    for v in variants:
        s = [r.get("score", 0) for r in by_variant[v]]
        scores.append(np.mean(s))
        stds.append(np.std(s))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))
    ax.bar(variants, scores, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel("Composite Score (0-1)")
    ax.set_title("Review Composite Score by Variant")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "review_scores.png", dpi=150)
    plt.close(fig)


def _cost_vs_score(by_variant: dict, output_dir: Path) -> None:
    """Scatter plot: cost vs composite score."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, vresults in by_variant.items():
        costs = [r.get("call_result", {}).get("cost_usd", 0) for r in vresults]
        scores = [r.get("score", 0) for r in vresults]
        ax.scatter(np.mean(costs), np.mean(scores), s=150, label=variant, zorder=5)

    ax.set_xlabel("Average Cost ($)")
    ax.set_ylabel("Average Composite Score")
    ax.set_title("Cost vs Review Quality")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "review_cost_vs_score.png", dpi=150)
    plt.close(fig)


def _exploration_depth(by_variant: dict, output_dir: Path) -> None:
    """Bar chart of exploration metrics (turns, tool calls, files read) by variant."""
    variants = list(by_variant.keys())
    metric_names = ["total_turns", "tool_calls", "files_read"]
    labels = ["Turns", "Tool Calls", "Files Read"]

    x = np.arange(len(variants))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, label) in enumerate(zip(metric_names, labels)):
        values = []
        for v in variants:
            metrics_list = [r.get("agent_metrics", {}) for r in by_variant[v]]
            values.append(np.mean([m.get(metric, 0) for m in metrics_list]))
        ax.bar(x + i * width, values, width, label=label)

    ax.set_ylabel("Count")
    ax.set_title("Exploration Depth by Variant")
    ax.set_xticks(x + width)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "review_exploration.png", dpi=150)
    plt.close(fig)
