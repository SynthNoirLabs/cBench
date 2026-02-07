"""Agent teams benchmark — tests multi-agent coordination via Claude Agent SDK.

Uses the claude-agent-sdk to spawn sub-agents that work on parallel subtasks,
then combines results. Compares single-agent vs multi-agent performance.
"""

import time
from typing import Any

from cbench.benchmarks.base import Benchmark, BenchmarkResult
from cbench.client import CallResult
from cbench.config import ModelID
from cbench.display import get_progress
from cbench.metrics import calculate_cost
from cbench.storage import save_results
from cbench.tasks.base import ScoringMethod, TaskDefinition

# Tasks designed to benefit from parallelization
_AGENT_TASKS: list[dict[str, Any]] = [
    {
        "name": "parallel_research",
        "prompt": (
            "Research and provide detailed summaries of three topics:\n"
            "1. The history and current state of quantum computing\n"
            "2. Recent advances in CRISPR gene editing technology\n"
            "3. The development of fusion energy reactors\n\n"
            "For each topic, provide at least 3 key facts and recent developments."
        ),
        "check_terms": ["quantum", "CRISPR", "fusion"],
    },
    {
        "name": "multi_perspective_analysis",
        "prompt": (
            "Analyze the impact of artificial intelligence on employment from "
            "three perspectives:\n"
            "1. An economist focusing on labor market data\n"
            "2. A technologist focusing on automation capabilities\n"
            "3. A policy maker focusing on regulatory frameworks\n\n"
            "Each perspective should be clearly labeled and contain substantive analysis."
        ),
        "check_terms": ["economist", "technolog", "policy"],
    },
]


class AgentTeamsBenchmark(Benchmark):
    name = "agent_teams"
    description = "Multi-agent coordination via Agent SDK — single vs team performance"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "single_agent",
                "use_team": False,
                "model": ModelID.OPUS_4_6,
            },
            {
                "name": "agent_team_3",
                "use_team": True,
                "num_agents": 3,
                "model": ModelID.OPUS_4_6,
            },
        ]

    def get_tasks(self) -> list[TaskDefinition]:
        tasks: list[TaskDefinition] = []
        for t in _AGENT_TASKS:
            tasks.append(
                TaskDefinition(
                    name=str(t["name"]),
                    category="agent_teams",
                    prompt=str(t["prompt"]),
                    expected_answer=t["check_terms"][0],
                    scoring_method=ScoringMethod.CONTAINS,
                    difficulty="hard",
                )
            )
        return tasks

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        return {}


class AgentTeamsRunner:
    """Custom runner for agent teams benchmark.

    Single-agent variant: one API call with the full prompt.
    Team variant: spawns sub-agents via Agent SDK, each handling a subtask.
    """

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir

    def run(self, benchmark: AgentTeamsBenchmark, num_runs: int = 1) -> list[BenchmarkResult]:
        variants = benchmark.get_variants()
        total = len(variants) * len(_AGENT_TASKS) * num_runs
        results: list[BenchmarkResult] = []

        with get_progress() as progress:
            prog_task = progress.add_task("[cyan]agent_teams[/cyan]", total=total)

            for variant in variants:
                model = ModelID.OPUS_4_6

                for task_info in _AGENT_TASKS:
                    for run_idx in range(num_runs):
                        start = time.monotonic()

                        if variant["use_team"]:
                            call_result = self._run_team(
                                model, task_info, variant["num_agents"]
                            )
                        else:
                            call_result = self._run_single(model, task_info)

                        elapsed = time.monotonic() - start
                        call_result.wall_clock_seconds = elapsed

                        # Score: check if all key terms appear in response
                        check_terms = task_info["check_terms"]
                        response_lower = call_result.response_text.lower()
                        hits = sum(
                            1 for term in check_terms if term.lower() in response_lower
                        )
                        score = hits / len(check_terms)

                        task_def = TaskDefinition(
                            name=str(task_info["name"]),
                            category="agent_teams",
                            prompt=str(task_info["prompt"]),
                            expected_answer=check_terms[0],
                            scoring_method=ScoringMethod.CONTAINS,
                            difficulty="hard",
                        )

                        results.append(
                            BenchmarkResult(
                                benchmark_name="agent_teams",
                                variant_name=variant["name"],
                                task_name=task_def.name,
                                run_index=run_idx,
                                call_result=call_result,
                                score=score,
                                variant_config=variant,
                            )
                        )
                        progress.update(prog_task, advance=1)
                        time.sleep(0.5)

        save_results("agent_teams", results, self.output_dir)
        return results

    def _run_single(self, model: ModelID, task_info: dict[str, Any]) -> CallResult:
        """Run with a single API call."""
        from cbench.client import BenchClient

        client = BenchClient()
        return client.call_sync(
            model=model,
            messages=[{"role": "user", "content": str(task_info["prompt"])}],
            max_tokens=4096,
        )

    def _run_team(self, model: ModelID, task_info: dict[str, Any], num_agents: int) -> CallResult:
        """Run with multiple sub-agents via Agent SDK."""
        try:
            return self._run_team_sdk(model, task_info, num_agents)
        except ImportError:
            # Fallback: simulate team with parallel single calls
            return self._run_team_fallback(model, task_info, num_agents)

    def _run_team_sdk(self, model: ModelID, task_info: dict[str, Any], num_agents: int) -> CallResult:
        """Use Claude Agent SDK to spawn sub-agents."""
        from claude_agent_sdk import ClaudeSDKClient

        sdk = ClaudeSDKClient()

        # Split the prompt into subtasks (one per numbered section)
        lines = str(task_info["prompt"]).split("\n")
        subtasks: list[str] = []
        current = ""
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:3]:
                if current:
                    subtasks.append(current.strip())
                current = stripped
            elif current:
                current += " " + stripped
        if current:
            subtasks.append(current.strip())

        # Ensure we have enough subtasks for agents
        while len(subtasks) < num_agents:
            subtasks.append(subtasks[-1] if subtasks else str(task_info["prompt"]))

        # Spawn sub-agents
        agent_results: list[str] = []
        total_input = 0
        total_output = 0
        _total_cost = 0.0

        for i in range(min(num_agents, len(subtasks))):
            sub_prompt = (
                f"You are agent {i + 1} of {num_agents} working on a team task. "
                f"Your specific subtask is:\n{subtasks[i]}\n\n"
                f"Provide a thorough, detailed response covering this subtask."
            )

            try:
                result = sdk.create_message(  # type: ignore[attr-defined]
                    model=model.value,
                    messages=[{"role": "user", "content": sub_prompt}],
                    max_tokens=2048,
                )
                text = ""
                for block in result.content:
                    if block.type == "text":
                        text += block.text

                agent_results.append(text)
                total_input += result.usage.input_tokens
                total_output += result.usage.output_tokens
            except Exception as e:
                agent_results.append(f"Agent {i + 1} error: {e}")

        # Combine results
        combined = "\n\n---\n\n".join(
            f"## Agent {i + 1} Response\n{text}"
            for i, text in enumerate(agent_results)
        )

        usage_dict: dict[str, Any] = {"input_tokens": total_input, "output_tokens": total_output}
        cost = calculate_cost(model, usage_dict)

        return CallResult(
            input_tokens=total_input,
            output_tokens=total_output,
            response_text=combined,
            response_length=len(combined),
            cost_usd=cost,
        )

    def _run_team_fallback(self, model: ModelID, task_info: dict[str, Any], num_agents: int) -> CallResult:
        """Fallback: simulate team using multiple BenchClient calls."""
        from cbench.client import BenchClient

        client = BenchClient()

        # Split prompt into subtasks
        lines = str(task_info["prompt"]).split("\n")
        subtasks: list[str] = []
        current = ""
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:3]:
                if current:
                    subtasks.append(current.strip())
                current = stripped
            elif current:
                current += " " + stripped
        if current:
            subtasks.append(current.strip())

        while len(subtasks) < num_agents:
            subtasks.append(subtasks[-1] if subtasks else str(task_info["prompt"]))

        agent_results: list[str] = []
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for i in range(min(num_agents, len(subtasks))):
            sub_prompt = (
                f"You are agent {i + 1} of {num_agents}. "
                f"Your specific subtask:\n{subtasks[i]}\n\n"
                f"Provide a thorough response."
            )
            result = client.call_sync(
                model=model,
                messages=[{"role": "user", "content": sub_prompt}],
                max_tokens=2048,
            )
            agent_results.append(result.response_text)
            total_input += result.input_tokens
            total_output += result.output_tokens
            total_cost += result.cost_usd

        combined = "\n\n---\n\n".join(
            f"## Agent {i + 1} Response\n{text}"
            for i, text in enumerate(agent_results)
        )

        return CallResult(
            input_tokens=total_input,
            output_tokens=total_output,
            response_text=combined,
            response_length=len(combined),
            cost_usd=total_cost,
        )

    @staticmethod
    def estimate_cost(num_runs: int = 1) -> dict[str, Any]:
        num_tasks = len(_AGENT_TASKS)
        # Single: 1 call per task; Team: 3 calls per task
        calls_single = num_tasks * num_runs
        calls_team = num_tasks * 3 * num_runs
        total_calls = calls_single + calls_team

        avg_input = 500
        avg_output = 1000
        pricing_input = 5.00
        pricing_output = 25.00

        cost = (
            (avg_input / 1_000_000) * pricing_input * total_calls
            + (avg_output / 1_000_000) * pricing_output * total_calls
        )

        return {
            "benchmark": "agent_teams",
            "variants": 2,
            "tasks": num_tasks,
            "api_calls": total_calls,
            "estimated_cost": cost,
        }
