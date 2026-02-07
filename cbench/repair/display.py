"""Display for code repair benchmark results."""

from __future__ import annotations

from typing import Any

from rich.table import Table

from cbench.display import console


def print_repair_results_table(results: list[Any]) -> None:
    """Print a Rich table summarizing repair benchmark results."""
    table = Table(title="Code Repair Results", show_lines=True)
    table.add_column("Variant", style="cyan")
    table.add_column("Fixture", style="green")
    table.add_column("Difficulty", style="magenta")
    table.add_column("Run", justify="right")
    table.add_column("Tests Before", justify="right")
    table.add_column("Tests After", justify="right")
    table.add_column("Total Tests", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Turns", justify="right")
    table.add_column("Cost ($)", justify="right", style="yellow")
    table.add_column("Time (s)", justify="right")

    for r in results:
        if hasattr(r, "__dataclass_fields__"):
            variant = r.variant_name
            fixture = r.fixture_name
            difficulty = r.difficulty
            run_idx = r.run_index
            before = r.tests_passed_before
            after = r.tests_passed_after
            total = r.tests_total
            score = r.score
            turns = r.agent_turns
            cost = r.call_result.cost_usd
            elapsed = r.wall_clock_seconds
        else:
            variant = r.get("variant_name", "")
            fixture = r.get("fixture_name", "")
            difficulty = r.get("difficulty", "")
            run_idx = r.get("run_index", 0)
            before = r.get("tests_passed_before", 0)
            after = r.get("tests_passed_after", 0)
            total = r.get("tests_total", 0)
            score = r.get("score", 0)
            turns = r.get("agent_turns", 0)
            call = r.get("call_result", {})
            cost = call.get("cost_usd", 0) if isinstance(call, dict) else 0
            elapsed = r.get("wall_clock_seconds", 0)

        score_style = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
        diff_style = {"easy": "green", "medium": "yellow", "hard": "red"}.get(difficulty, "white")

        table.add_row(
            variant,
            fixture,
            f"[{diff_style}]{difficulty}[/{diff_style}]",
            str(run_idx),
            f"{before}/{total}",
            f"{after}/{total}",
            str(total),
            f"[{score_style}]{score:.2f}[/{score_style}]",
            str(turns),
            f"{cost:.4f}",
            f"{elapsed:.1f}",
        )

    console.print(table)
