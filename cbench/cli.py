from __future__ import annotations

import argparse
import sys

from cbench.benchmarks import BENCHMARK_REGISTRY, get_benchmark, list_benchmarks
from cbench.client import BenchClient
from cbench.display import (
    confirm_run,
    console,
    print_benchmark_list,
    print_cost_estimate,
    print_results_table,
    print_summary_table,
)
from cbench.review.benchmark import ReviewBenchmark
from cbench.review.display import (
    print_review_cost_estimate,
    print_review_results_table,
    print_review_summary_table,
)
from cbench.review.runner import ReviewRunner
from cbench.runner import BenchmarkRunner
from cbench.storage import load_results
from cbench.tasks import get_all_tasks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cbench", description="Multi-provider AI Model Benchmark Suite"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("benchmarks", nargs="+", help="Benchmark names or 'all'")
    run_parser.add_argument(
        "--num-runs", type=int, default=1, help="Number of runs per task per variant"
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Estimate cost without running")
    run_parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    run_parser.add_argument("--output-dir", default="results", help="Output directory for results")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze saved results")
    analyze_parser.add_argument("path", help="Path to results directory or JSON file")
    analyze_parser.add_argument(
        "--format", choices=["charts", "report", "both"], default="both", help="Output format"
    )

    # list
    subparsers.add_parser("list", help="List available benchmarks")

    # review
    review_parser = subparsers.add_parser(
        "review", help="Review a local repository with LLM-as-Judge scoring"
    )
    review_parser.add_argument("repo_path", help="Path to the repository to review")
    review_parser.add_argument("--num-runs", type=int, default=1, help="Number of runs per variant")
    review_parser.add_argument(
        "--max-turns", type=int, default=30, help="Max agent exploration turns"
    )
    review_parser.add_argument(
        "--dry-run", action="store_true", help="Estimate cost without running"
    )
    review_parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    review_parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    review_parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated list of variant names to run (e.g. 'opus_adaptive,haiku_no_thinking')",
    )

    # estimate
    estimate_parser = subparsers.add_parser("estimate", help="Estimate cost for benchmarks")
    estimate_parser.add_argument(
        "benchmarks", nargs="*", default=["all"], help="Benchmark names or 'all'"
    )
    estimate_parser.add_argument("--num-runs", type=int, default=1)

    # repair
    repair_parser = subparsers.add_parser(
        "repair", help="Run code repair benchmark (agentic, multi-turn)"
    )
    repair_parser.add_argument(
        "fixtures",
        nargs="*",
        default=["all"],
        help="Fixture names or 'all' (e.g. calculator_bug, api_handler_bug)",
    )
    repair_parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant names (e.g. 'sdk_opus,sdk_haiku')",
    )
    repair_parser.add_argument("--num-runs", type=int, default=1, help="Runs per fixture per variant")
    repair_parser.add_argument("--max-turns", type=int, default=15, help="Max agent turns")
    repair_parser.add_argument("--dry-run", action="store_true", help="Estimate cost without running")
    repair_parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    repair_parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args(argv)

    if args.command == "list":
        cmd_list()
    elif args.command == "estimate":
        cmd_estimate(args.benchmarks, args.num_runs)
    elif args.command == "run":
        cmd_run(args.benchmarks, args.num_runs, args.dry_run, args.no_confirm, args.output_dir)
    elif args.command == "analyze":
        cmd_analyze(args.path, args.format)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "repair":
        cmd_repair(args)


def cmd_list() -> None:
    benchmarks = list_benchmarks()
    print_benchmark_list(benchmarks)


def cmd_estimate(benchmark_names: list[str], num_runs: int) -> None:
    benchmarks = _resolve_benchmarks(benchmark_names)
    tasks = get_all_tasks()
    estimates = [BenchmarkRunner.estimate_cost(b, tasks, num_runs) for b in benchmarks]
    print_cost_estimate(estimates)


def cmd_run(
    benchmark_names: list[str], num_runs: int, dry_run: bool, no_confirm: bool, output_dir: str
) -> None:
    benchmarks = _resolve_benchmarks(benchmark_names)
    tasks = get_all_tasks()

    # Always show estimates first
    estimates = [BenchmarkRunner.estimate_cost(b, tasks, num_runs) for b in benchmarks]
    print_cost_estimate(estimates)

    if dry_run:
        console.print("[yellow]Dry run complete. No API calls made.[/yellow]")
        return

    total_cost = sum(e["estimated_cost"] for e in estimates)
    if not no_confirm and not confirm_run(total_cost):
        console.print("[red]Aborted.[/red]")
        return

    client = BenchClient()
    runner = BenchmarkRunner(client, output_dir)

    all_results = []
    for benchmark in benchmarks:
        console.print(f"\n[bold cyan]Running: {benchmark.name}[/bold cyan]")
        results = runner.run_benchmark(benchmark, tasks, num_runs)
        all_results.extend(results)
        print_results_table(results)
        if num_runs > 1:
            print_summary_table(results)

    console.print(
        f"\n[bold green]All benchmarks complete. Results saved to {output_dir}/[/bold green]"
    )


def cmd_analyze(path: str, fmt: str) -> None:
    results = load_results(path)

    # Detect if these are review results (have judge_scores field)
    is_review = results and "judge_scores" in results[0]

    if is_review:
        from cbench.review.analysis import generate_review_charts, generate_review_report

        if fmt in ("report", "both"):
            report_path = generate_review_report(results, path)
            console.print(f"[green]Review report saved to {report_path}[/green]")
        if fmt in ("charts", "both"):
            chart_dir = generate_review_charts(results, path)
            console.print(f"[green]Review charts saved to {chart_dir}[/green]")
    else:
        if fmt in ("report", "both"):
            from cbench.analysis.report import generate_report

            report_path = generate_report(results, path)
            console.print(f"[green]Report saved to {report_path}[/green]")
        if fmt in ("charts", "both"):
            from cbench.analysis.charts import generate_charts

            chart_dir = generate_charts(results, path)
            console.print(f"[green]Charts saved to {chart_dir}[/green]")


def cmd_review(args) -> None:
    from pathlib import Path

    import anthropic

    repo_path = Path(args.repo_path).resolve()
    if not repo_path.is_dir():
        console.print(f"[red]Error: {args.repo_path} is not a directory[/red]")
        sys.exit(1)

    benchmark = ReviewBenchmark()

    # Filter variants if specified
    if args.variants:
        requested = [v.strip() for v in args.variants.split(",")]
        all_variants = benchmark.get_variants()
        available_names = [v["name"] for v in all_variants]
        filtered = [v for v in all_variants if v["name"] in requested]
        unknown = [n for n in requested if n not in available_names]
        if unknown:
            console.print(f"[red]Unknown variants: {', '.join(unknown)}[/red]")
            console.print(f"[dim]Available: {', '.join(available_names)}[/dim]")
            sys.exit(1)
        benchmark = ReviewBenchmark(variants=filtered)

    # Show cost estimate
    estimate = ReviewRunner.estimate_cost(benchmark, repo_path, args.num_runs, args.max_turns)
    print_review_cost_estimate(estimate)

    if args.dry_run:
        console.print("[yellow]Dry run complete. No API calls made.[/yellow]")
        return

    if not args.no_confirm and not confirm_run(estimate["estimated_cost"]):
        console.print("[red]Aborted.[/red]")
        return

    client = anthropic.Anthropic(max_retries=3)
    runner = ReviewRunner(client, args.output_dir)

    console.print(f"\n[bold cyan]Reviewing: {repo_path}[/bold cyan]")
    results = runner.run_review(benchmark, repo_path, args.num_runs, args.max_turns)
    print_review_results_table(results)
    if args.num_runs > 1:
        print_review_summary_table(results)
    console.print(
        f"\n[bold green]Review complete. Results saved to {args.output_dir}/[/bold green]"
    )


def cmd_repair(args) -> None:
    from cbench.repair.benchmark import RepairBenchmark
    from cbench.repair.runner import RepairRunner

    benchmark = RepairBenchmark()

    # Filter fixtures
    if "all" not in args.fixtures:
        benchmark = RepairBenchmark(fixture_names=args.fixtures)

    # Filter variants if specified
    if args.variants:
        requested = [v.strip() for v in args.variants.split(",")]
        all_variants = benchmark.get_variants()
        available_names = [v["name"] for v in all_variants]
        filtered = [v for v in all_variants if v["name"] in requested]
        unknown = [n for n in requested if n not in available_names]
        if unknown:
            console.print(f"[red]Unknown variants: {', '.join(unknown)}[/red]")
            console.print(f"[dim]Available: {', '.join(available_names)}[/dim]")
            sys.exit(1)
        benchmark = RepairBenchmark(
            fixture_names=None if "all" in args.fixtures else args.fixtures,
            variants=filtered,
        )

    # Show cost estimate
    estimate = RepairRunner.estimate_cost(benchmark, args.num_runs, args.max_turns)
    from cbench.display import print_cost_estimate as _print_est

    console.print(f"\n[bold]Repair Benchmark[/bold]")
    console.print(f"  Fixtures: {len(benchmark.get_fixtures())}")
    console.print(f"  Variants: {len(benchmark.get_variants())}")
    console.print(f"  Runs: {args.num_runs}")
    console.print(f"  Max turns: {args.max_turns}")
    console.print(f"  [yellow]Estimated cost: ${estimate['estimated_cost']:.4f}[/yellow]")

    if args.dry_run:
        console.print("[yellow]Dry run complete. No API calls made.[/yellow]")
        return

    if not args.no_confirm and not confirm_run(estimate["estimated_cost"]):
        console.print("[red]Aborted.[/red]")
        return

    runner = RepairRunner(args.output_dir)
    console.print(f"\n[bold cyan]Running code repair benchmark[/bold cyan]")
    results = runner.run(benchmark, args.num_runs, args.max_turns)

    # Display results
    from cbench.repair.display import print_repair_results_table

    print_repair_results_table(results)
    console.print(
        f"\n[bold green]Repair benchmark complete. Results saved to {args.output_dir}/[/bold green]"
    )


def _resolve_benchmarks(names: list[str]) -> list:
    if "all" in names:
        return list(BENCHMARK_REGISTRY.values())
    return [get_benchmark(name) for name in names]
