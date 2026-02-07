"""Review-specific display: results table and cost estimate panel."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import Any

from rich.panel import Panel
from rich.table import Table

from cbench.display import console


def print_review_results_table(results: list[Any]) -> None:
    """Print a Rich table summarizing review benchmark results."""
    table = Table(title="Repo Review Results", show_lines=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Agent", style="magenta")
    table.add_column("Run", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Tool Calls", justify="right")
    table.add_column("Files Read", justify="right")
    table.add_column("Compl.", justify="right")
    table.add_column("Accur.", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Action.", justify="right")
    table.add_column("Prior.", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Cost ($)", justify="right", style="yellow")
    table.add_column("Score/$", justify="right", style="magenta")
    table.add_column("Score/1Ktok", justify="right", style="magenta")
    table.add_column("Time (s)", justify="right")

    for r in results:
        # Support both dataclass and dict
        if hasattr(r, "__dataclass_fields__"):
            variant = r.variant_name
            agent_type = getattr(r, "agent_type", "raw")
            run_idx = r.run_index
            metrics = r.agent_metrics
            scores = r.judge_scores
            score = r.score
            cost = r.call_result.cost_usd
            elapsed = r.wall_clock_seconds
            input_tok = r.call_result.input_tokens
            output_tok = r.call_result.output_tokens
        else:
            variant = r.get("variant_name", "")
            agent_type = r.get("agent_type", "raw")
            run_idx = r.get("run_index", 0)
            metrics = r.get("agent_metrics", {})
            scores = r.get("judge_scores", {})
            score = r.get("score", 0)
            call = r.get("call_result", {})
            cost = call.get("cost_usd", 0) if isinstance(call, dict) else 0
            elapsed = r.get("wall_clock_seconds", 0)
            input_tok = call.get("input_tokens", 0) if isinstance(call, dict) else 0
            output_tok = call.get("output_tokens", 0) if isinstance(call, dict) else 0

        # Extract metrics
        if hasattr(metrics, "total_turns"):
            turns = metrics.total_turns
            tool_calls = metrics.tool_calls
            files_read = metrics.files_read
        elif isinstance(metrics, dict):
            turns = metrics.get("total_turns", 0)
            tool_calls = metrics.get("tool_calls", 0)
            files_read = metrics.get("files_read", 0)
        else:
            turns = tool_calls = files_read = 0

        # Extract scores
        if hasattr(scores, "completeness"):
            compl = scores.completeness
            accur = scores.accuracy
            depth = scores.depth
            action = scores.actionability
            prior = scores.prioritization
        elif isinstance(scores, dict):
            compl = scores.get("completeness", 0)
            accur = scores.get("accuracy", 0)
            depth = scores.get("depth", 0)
            action = scores.get("actionability", 0)
            prior = scores.get("prioritization", 0)
        else:
            compl = accur = depth = action = prior = 0

        score_style = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"

        total_tok = input_tok + output_tok
        score_per_dollar = f"{score / cost:.1f}" if cost > 0 else "-"
        score_per_1k_tok = f"{score / (total_tok / 1000):.3f}" if total_tok > 0 else "-"

        table.add_row(
            str(variant),
            str(agent_type),
            str(run_idx),
            str(turns),
            str(tool_calls),
            str(files_read),
            f"{compl:.2f}",
            f"{accur:.2f}",
            f"{depth:.2f}",
            f"{action:.2f}",
            f"{prior:.2f}",
            f"[{score_style}]{score:.2f}[/{score_style}]",
            f"{cost:.4f}",
            score_per_dollar,
            score_per_1k_tok,
            f"{elapsed:.1f}",
        )

    console.print(table)


def print_review_summary_table(results: list[Any]) -> None:
    """Print aggregated summary table for review results when num_runs > 1."""
    rows = []
    for r in results:
        if hasattr(r, "__dataclass_fields__"):
            from dataclasses import asdict

            r = asdict(r)
        rows.append(r)

    groups: dict[str, list[float]] = defaultdict(list)
    cost_groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        key = r.get("variant_name", "")
        groups[key].append(r.get("score", 0.0))
        call = r.get("call_result", {})
        cost_groups[key].append(call.get("cost_usd", 0.0))

    table = Table(title="Review Summary (aggregated over runs)", show_lines=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Mean", justify="right", style="bold")
    table.add_column("Stddev", justify="right")
    table.add_column("Pass^k", justify="right")
    table.add_column("Avg Cost ($)", justify="right", style="yellow")
    table.add_column("Mean Score/$", justify="right", style="magenta")

    for variant, scores in sorted(groups.items()):
        k = len(scores)
        avg = mean(scores)
        sd = stdev(scores) if k > 1 else 0.0
        pass_k = sum(1 for s in scores if s >= 0.5) / k
        avg_cost = mean(cost_groups[variant])
        mean_score_dollar = f"{avg / avg_cost:.1f}" if avg_cost > 0 else "-"

        score_style = "green" if avg >= 0.7 else "yellow" if avg >= 0.4 else "red"
        pass_style = "green" if pass_k >= 0.8 else "yellow" if pass_k >= 0.5 else "red"

        table.add_row(
            variant,
            str(k),
            f"[{score_style}]{avg:.3f}[/{score_style}]",
            f"{sd:.3f}",
            f"[{pass_style}]{pass_k:.2f}[/{pass_style}]",
            f"{avg_cost:.4f}",
            mean_score_dollar,
        )

    console.print(table)


def print_review_cost_estimate(estimate: dict[str, Any]) -> None:
    """Print a Rich panel with review cost estimate details."""
    repo = estimate.get("repo_stats", {})
    size_cat = repo.get("size_category", "unknown")
    size_colors = {"small": "green", "medium": "yellow", "large": "red", "huge": "bold red"}
    size_style = size_colors.get(size_cat, "white")

    lines = [
        f"[bold]Benchmark:[/bold] {estimate['benchmark']}",
        f"[bold]Repository:[/bold] {repo.get('source_files', '?')} source files, "
        f"{repo.get('total_loc', '?'):,} LOC "
        f"({repo.get('total_files', '?')} total files) "
        f"[{size_style}][{size_cat}][/{size_style}]",
        f"[bold]Variants:[/bold] {estimate['variants']}",
        f"[bold]Runs per variant:[/bold] {estimate['num_runs']}",
        f"[bold]Max turns:[/bold] {estimate['max_turns']}",
        "",
    ]

    table = Table(show_header=True, show_lines=False, padding=(0, 1))
    table.add_column("Variant", style="cyan")
    table.add_column("Agent", style="magenta")
    table.add_column("Model")
    table.add_column("Est. Turns", justify="right")
    table.add_column("Agent ($)", justify="right", style="yellow")
    table.add_column("Judge ($)", justify="right", style="yellow")
    table.add_column("Total ($)", justify="right", style="bold yellow")

    for d in estimate.get("variant_details", []):
        agent_type = d.get("agent_type", "raw_api")
        agent_label = "sdk" if agent_type == "sdk" else "raw"
        table.add_row(
            d["variant"],
            agent_label,
            d["model"],
            str(d["est_turns"]),
            f"{d['agent_cost']:.4f}",
            f"{d['judge_cost']:.4f}",
            f"{d['total_cost']:.4f}",
        )

    console.print(
        Panel(
            "\n".join(lines),
            title="Review Cost Estimate",
        )
    )
    console.print(table)
    console.print(
        f"\n[bold yellow]Estimated total cost: ${estimate['estimated_cost']:.4f}[/bold yellow]"
    )
