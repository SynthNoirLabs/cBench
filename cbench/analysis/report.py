from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path


def generate_report(results: list[dict], output_path: str | Path) -> Path:
    output_dir = Path(output_path)
    if output_dir.is_file():
        output_dir = output_dir.parent
    report_path = output_dir / "report.md"

    lines = [
        "# cBench Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"Total results: {len(results)}",
        "",
    ]

    # Group by benchmark
    by_benchmark = defaultdict(list)
    for r in results:
        by_benchmark[r.get("benchmark_name", "unknown")].append(r)

    for bench_name, bench_results in by_benchmark.items():
        lines.append(f"## {bench_name}")
        lines.append("")

        # Group by variant
        by_variant = defaultdict(list)
        for r in bench_results:
            by_variant[r.get("variant_name", "unknown")].append(r)

        # Table header
        lines.append("| Variant | Avg Score | Avg Latency (s) | Avg Cost ($) | Calls |")
        lines.append("|---------|-----------|-----------------|-------------|-------|")

        for variant, vresults in by_variant.items():
            scores = [r.get("score", 0) for r in vresults]
            calls = [r.get("call_result", {}) for r in vresults]
            latencies = [c.get("wall_clock_seconds", 0) for c in calls]
            costs = [c.get("cost_usd", 0) for c in calls]

            avg_score = sum(scores) / len(scores) if scores else 0
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            avg_cost = sum(costs) / len(costs) if costs else 0

            lines.append(
                f"| {variant} | {avg_score:.3f} | {avg_lat:.2f} | {avg_cost:.6f} | {len(vresults)} |"
            )

        lines.append("")

        # Per-task breakdown
        lines.append("### Task Breakdown")
        lines.append("")
        lines.append("| Task | Variant | Score | Latency (s) | Cost ($) |")
        lines.append("|------|---------|-------|-------------|----------|")

        for r in bench_results:
            call = r.get("call_result", {})
            lines.append(
                f"| {r.get('task_name', '')} "
                f"| {r.get('variant_name', '')} "
                f"| {r.get('score', 0):.2f} "
                f"| {call.get('wall_clock_seconds', 0):.2f} "
                f"| {call.get('cost_usd', 0):.6f} |"
            )

        lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path
