from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


def get_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_results_table(results: list) -> None:
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Run", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Input Tok", justify="right")
    table.add_column("Output Tok", justify="right")
    table.add_column("Cost ($)", justify="right", style="yellow")
    table.add_column("Score/$", justify="right", style="magenta")
    table.add_column("Score/1Ktok", justify="right", style="magenta")
    table.add_column("Latency (s)", justify="right")
    table.add_column("TTFB (s)", justify="right")

    for r in results:
        # Support both dict and dataclass
        if hasattr(r, "__dataclass_fields__"):
            from dataclasses import asdict

            r = asdict(r)

        call = r.get("call_result", {})
        score_val = r.get("score", 0)
        score_style = "green" if score_val >= 0.8 else "yellow" if score_val >= 0.5 else "red"

        cost = call.get("cost_usd", 0)
        input_tok = call.get("input_tokens", 0)
        output_tok = call.get("output_tokens", 0)
        total_tok = input_tok + output_tok

        score_per_dollar = f"{score_val / cost:.1f}" if cost > 0 else "-"
        score_per_1k_tok = f"{score_val / (total_tok / 1000):.3f}" if total_tok > 0 else "-"

        table.add_row(
            str(r.get("variant_name", "")),
            str(r.get("task_name", "")),
            str(r.get("run_index", 0)),
            f"[{score_style}]{score_val:.2f}[/{score_style}]",
            str(input_tok),
            str(output_tok),
            f"{cost:.6f}",
            score_per_dollar,
            score_per_1k_tok,
            f"{call.get('wall_clock_seconds', 0):.2f}",
            f"{call.get('time_to_first_token_seconds', '-')}",
        )

    console.print(table)


def print_summary_table(results: list) -> None:
    """Print aggregated summary table when num_runs > 1.

    Groups results by (variant, task) and shows mean, stddev, pass^k.
    """
    # Convert dataclasses to dicts
    rows = []
    for r in results:
        if hasattr(r, "__dataclass_fields__"):
            from dataclasses import asdict

            r = asdict(r)
        rows.append(r)

    # Group by (variant, task)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    cost_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        key = (r.get("variant_name", ""), r.get("task_name", ""))
        groups[key].append(r.get("score", 0.0))
        call = r.get("call_result", {})
        cost_groups[key].append(call.get("cost_usd", 0.0))

    table = Table(title="Summary (aggregated over runs)", show_lines=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Runs", justify="right")
    table.add_column("Mean", justify="right", style="bold")
    table.add_column("Stddev", justify="right")
    table.add_column("Pass^k", justify="right")
    table.add_column("Avg Cost ($)", justify="right", style="yellow")
    table.add_column("Mean Score/$", justify="right", style="magenta")

    for (variant, task), scores in sorted(groups.items()):
        k = len(scores)
        avg = mean(scores)
        sd = stdev(scores) if k > 1 else 0.0
        pass_k = sum(1 for s in scores if s >= 0.5) / k
        avg_cost = mean(cost_groups[(variant, task)])
        mean_score_dollar = f"{avg / avg_cost:.1f}" if avg_cost > 0 else "-"

        score_style = "green" if avg >= 0.8 else "yellow" if avg >= 0.5 else "red"
        pass_style = "green" if pass_k >= 0.8 else "yellow" if pass_k >= 0.5 else "red"

        table.add_row(
            variant,
            task,
            str(k),
            f"[{score_style}]{avg:.3f}[/{score_style}]",
            f"{sd:.3f}",
            f"[{pass_style}]{pass_k:.2f}[/{pass_style}]",
            f"{avg_cost:.6f}",
            mean_score_dollar,
        )

    console.print(table)


def print_cost_estimate(estimates: list[dict]) -> None:
    table = Table(title="Cost Estimates (Dry Run)", show_lines=True)
    table.add_column("Benchmark", style="cyan")
    table.add_column("Variants", justify="right")
    table.add_column("Tasks", justify="right")
    table.add_column("API Calls", justify="right")
    table.add_column("Est. Cost ($)", justify="right", style="yellow")

    total = 0.0
    for e in estimates:
        table.add_row(
            e["benchmark"],
            str(e["variants"]),
            str(e["tasks"]),
            str(e["api_calls"]),
            f"{e['estimated_cost']:.4f}",
        )
        total += e["estimated_cost"]

    table.add_row("[bold]TOTAL[/bold]", "", "", "", f"[bold]{total:.4f}[/bold]")
    console.print(table)


def print_benchmark_list(benchmarks: list[dict]) -> None:
    table = Table(title="Available Benchmarks")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Variants", justify="right")

    for b in benchmarks:
        table.add_row(b["name"], b["description"], str(b["variants"]))

    console.print(table)


def confirm_run(total_cost: float) -> bool:
    console.print(
        Panel(
            f"[yellow]Estimated total cost: ${total_cost:.4f}[/yellow]\n"
            "This will make real API calls.",
            title="Confirm Run",
        )
    )
    return console.input("[bold]Proceed? [y/N]: [/bold]").strip().lower() in ("y", "yes")
