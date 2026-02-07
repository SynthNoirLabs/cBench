"""ReviewRunner: orchestrates agent loops + judge scoring for repo review benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

from cbench.client import CallResult
from cbench.config import PRICING, ModelID
from cbench.review.agent import AgentLoop, AgentMetrics
from cbench.review.benchmark import ReviewBenchmark
from cbench.review.context import build_repo_tree, gather_key_files, scan_repo
from cbench.review.judge import JudgeScorer, JudgeScores
from cbench.review.tools import RepoSandbox
from cbench.storage import save_results


@dataclass
class ReviewResult:
    benchmark_name: str
    variant_name: str
    review_text: str
    agent_metrics: AgentMetrics
    judge_scores: JudgeScores
    score: float  # = judge composite
    call_result: CallResult  # aggregated tokens/cost
    run_index: int = 0
    wall_clock_seconds: float = 0.0
    agent_type: str = "raw"  # "raw" or "sdk"


class ReviewRunner:
    """Runs review agent + judge for all variants, saves results."""

    def __init__(self, client: anthropic.Anthropic, output_dir: str = "results") -> None:
        self.client = client
        self.output_dir = output_dir

    def run_review(
        self,
        benchmark: ReviewBenchmark,
        repo_path: str | Path,
        num_runs: int = 1,
        max_turns: int = 30,
    ) -> list[ReviewResult]:
        repo_path = Path(repo_path).resolve()
        sandbox = RepoSandbox(repo_path)

        # Build repo context once for the judge
        repo_tree = build_repo_tree(repo_path)
        key_files = gather_key_files(repo_path)

        judge = JudgeScorer(self.client)
        variants = benchmark.get_variants()
        results: list[ReviewResult] = []

        from cbench.display import console

        for variant in variants:
            for run_idx in range(num_runs):
                agent_type = variant.get("agent_type", "raw_api")
                agent_label = "sdk" if agent_type == "sdk" else "raw"
                console.print(
                    f"  [cyan]{variant['name']}[/cyan] [{agent_label}] "
                    f"run {run_idx + 1}/{num_runs}..."
                )
                start = time.monotonic()

                # Run agent loop (factory dispatch based on agent_type)
                agent = self._create_agent(variant, sandbox, repo_path, max_turns)
                agent_result = agent.run()

                if agent_result.error:
                    console.print(f"    [red]Agent error: {agent_result.error}[/red]")
                    # Create a result with zero scores
                    elapsed = time.monotonic() - start
                    results.append(
                        ReviewResult(
                            benchmark_name=benchmark.name,
                            variant_name=variant["name"],
                            review_text="",
                            agent_metrics=agent_result.metrics,
                            judge_scores=JudgeScores(),
                            score=0.0,
                            call_result=CallResult(
                                input_tokens=agent_result.metrics.total_input_tokens,
                                output_tokens=agent_result.metrics.total_output_tokens,
                                cost_usd=agent_result.metrics.total_cost_usd,
                                wall_clock_seconds=elapsed,
                                error=agent_result.error,
                            ),
                            run_index=run_idx,
                            wall_clock_seconds=elapsed,
                            agent_type=agent_label,
                        )
                    )
                    continue

                # Judge the review
                console.print("    [dim]Judging review...[/dim]")
                judge_scores = judge.score(agent_result.review_text, repo_tree, key_files)

                elapsed = time.monotonic() - start

                # Aggregate costs
                total_input = (
                    agent_result.metrics.total_input_tokens + judge_scores.judge_input_tokens
                )
                total_output = (
                    agent_result.metrics.total_output_tokens + judge_scores.judge_output_tokens
                )
                total_cost = agent_result.metrics.total_cost_usd + judge_scores.judge_cost_usd

                results.append(
                    ReviewResult(
                        benchmark_name=benchmark.name,
                        variant_name=variant["name"],
                        review_text=agent_result.review_text,
                        agent_metrics=agent_result.metrics,
                        judge_scores=judge_scores,
                        score=judge_scores.composite_score,
                        call_result=CallResult(
                            input_tokens=total_input,
                            output_tokens=total_output,
                            cost_usd=total_cost,
                            wall_clock_seconds=elapsed,
                            response_text=agent_result.review_text,
                            thinking_text=agent_result.thinking_text,
                        ),
                        run_index=run_idx,
                        wall_clock_seconds=elapsed,
                        agent_type=agent_label,
                    )
                )

                console.print(
                    f"    [green]Score: {judge_scores.composite_score:.2f}[/green] "
                    f"({agent_result.metrics.total_turns} turns, "
                    f"{agent_result.metrics.tool_calls} tool calls, "
                    f"${total_cost:.4f})"
                )

                time.sleep(1)  # Rate limiting between runs

        save_results(benchmark.name, results, self.output_dir)
        return results

    def _create_agent(self, variant: dict, sandbox: RepoSandbox, repo_path, max_turns: int):
        """Factory: dispatch to raw API or SDK agent based on variant config."""
        agent_type = variant.get("agent_type", "raw_api")
        if agent_type == "sdk":
            from cbench.review.agent_sdk import SDKAgentLoop

            return SDKAgentLoop(
                model=variant["model"],
                repo_path=repo_path,
                max_turns=max_turns,
                max_tokens=variant.get("max_tokens", 16384),
                thinking=variant.get("thinking"),
                effort=variant.get("effort"),
            )
        return AgentLoop(
            client=self.client,
            sandbox=sandbox,
            model=variant["model"],
            thinking=variant.get("thinking"),
            effort=variant.get("effort"),
            max_tokens=variant.get("max_tokens", 16384),
            max_turns=max_turns,
        )

    @staticmethod
    def estimate_cost(
        benchmark: ReviewBenchmark,
        repo_path: str | Path,
        num_runs: int = 1,
        max_turns: int = 30,
    ) -> dict:
        """Repo-aware cost estimate: scans file count and LOC to calibrate."""
        repo_stats = scan_repo(repo_path)
        variants = benchmark.get_variants()

        avg_turns = min(repo_stats.est_turns, max_turns)
        avg_input_per_turn = repo_stats.est_input_per_turn
        avg_output_per_turn = repo_stats.est_output_per_turn

        # Judge overhead per review (scales slightly with repo size)
        judge_input = 5000 + min(repo_stats.total_loc // 10, 10_000)
        judge_output = 500

        total_cost = 0.0
        details: list[dict] = []

        for variant in variants:
            model = variant["model"]
            pricing = PRICING[model]
            agent_type = variant.get("agent_type", "raw_api")

            # SDK variants have higher input overhead from Claude Code's system prompt
            sdk_overhead = 1.15 if agent_type == "sdk" else 1.0

            agent_input = avg_turns * avg_input_per_turn * num_runs * sdk_overhead
            agent_output = avg_turns * avg_output_per_turn * num_runs
            agent_cost = (agent_input / 1_000_000) * pricing["input_per_mtok"] + (
                agent_output / 1_000_000
            ) * pricing["output_per_mtok"]

            # Judge always uses Opus
            judge_pricing = PRICING[ModelID.OPUS_4_6]
            judge_cost_per_run = (judge_input / 1_000_000) * judge_pricing["input_per_mtok"] + (
                judge_output / 1_000_000
            ) * judge_pricing["output_per_mtok"]
            judge_total = judge_cost_per_run * num_runs

            variant_cost = agent_cost + judge_total
            total_cost += variant_cost

            details.append(
                {
                    "variant": variant["name"],
                    "model": model.value,
                    "agent_type": agent_type,
                    "est_turns": avg_turns,
                    "agent_cost": agent_cost,
                    "judge_cost": judge_total,
                    "total_cost": variant_cost,
                }
            )

        return {
            "benchmark": benchmark.name,
            "variants": len(variants),
            "num_runs": num_runs,
            "max_turns": max_turns,
            "estimated_cost": total_cost,
            "variant_details": details,
            "repo_stats": {
                "total_files": repo_stats.total_files,
                "source_files": repo_stats.source_files,
                "total_loc": repo_stats.total_loc,
                "size_category": repo_stats.size_category,
            },
        }
