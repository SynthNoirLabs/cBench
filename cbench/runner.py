from __future__ import annotations

import time

from cbench.benchmarks.base import Benchmark, BenchmarkResult
from cbench.client import BenchClient, CallResult
from cbench.config import PRICING, ModelID, is_claude_model
from cbench.display import get_progress
from cbench.providers import create_client
from cbench.providers.base import ProviderClient
from cbench.scorer import score_response
from cbench.storage import save_results
from cbench.tasks.base import TaskDefinition


class BenchmarkRunner:
    def __init__(
        self,
        client: BenchClient | ProviderClient | None = None,
        output_dir: str = "results",
    ):
        self._legacy_client = client if isinstance(client, BenchClient) else None
        self._provider_client = client if isinstance(client, ProviderClient) else None
        self.output_dir = output_dir
        # Cache provider clients per provider to avoid re-creating
        self._client_cache: dict[ModelID, ProviderClient] = {}

    def _get_client(self, model: ModelID) -> BenchClient | ProviderClient:
        """Get the appropriate client for a model.

        If a legacy BenchClient was passed and the model is Claude, use it.
        If a specific ProviderClient was passed, use it.
        Otherwise, create one via the factory.
        """
        if self._legacy_client and is_claude_model(model):
            return self._legacy_client
        if self._provider_client:
            return self._provider_client

        # Create client via factory, caching per model
        if model not in self._client_cache:
            self._client_cache[model] = create_client(model)
        return self._client_cache[model]

    def run_benchmark(
        self,
        benchmark: Benchmark,
        tasks: list[TaskDefinition],
        num_runs: int = 1,
    ) -> list[BenchmarkResult]:
        variants = benchmark.get_variants()
        total_calls = len(variants) * len(tasks) * num_runs
        results: list[BenchmarkResult] = []

        with get_progress() as progress:
            task_id = progress.add_task(
                f"[cyan]{benchmark.name}[/cyan]",
                total=total_calls,
            )

            for variant in variants:
                call_kwargs = benchmark.build_call_kwargs(variant)
                model = benchmark.get_model(variant)
                use_streaming = benchmark.uses_streaming(variant)
                client = self._get_client(model)

                # Strip Claude-only kwargs for non-Claude models
                if not is_claude_model(model):
                    call_kwargs.pop("thinking", None)
                    call_kwargs.pop("effort", None)

                for task_def in tasks:
                    for run_idx in range(num_runs):
                        messages = [{"role": "user", "content": task_def.prompt}]

                        try:
                            if use_streaming:
                                call_result = client.call_streaming(
                                    model=model,
                                    messages=messages,
                                    **call_kwargs,
                                )
                            else:
                                call_result = client.call_sync(
                                    model=model,
                                    messages=messages,
                                    **call_kwargs,
                                )
                        except Exception as e:
                            call_result = CallResult(error=str(e))

                        if call_result.error:
                            score = 0.0
                        else:
                            score = score_response(task_def, call_result.response_text)

                        results.append(
                            BenchmarkResult(
                                benchmark_name=benchmark.name,
                                variant_name=variant["name"],
                                task_name=task_def.name,
                                run_index=run_idx,
                                call_result=call_result,
                                score=score,
                                variant_config=variant,
                            )
                        )

                        progress.update(task_id, advance=1)
                        time.sleep(0.5)  # Rate limiting

        save_results(benchmark.name, results, self.output_dir)
        return results

    @staticmethod
    def estimate_cost(benchmark: Benchmark, tasks: list[TaskDefinition], num_runs: int) -> dict:
        variants = benchmark.get_variants()
        num_calls = len(variants) * len(tasks) * num_runs

        # Estimate ~500 input tokens and ~200 output tokens per call as baseline
        avg_input = 500
        avg_output = 200

        # Collect unique models used
        models_used = set()
        for v in variants:
            models_used.add(v.get("model", ModelID.OPUS_4_6))

        # Use the most expensive model for conservative estimate
        total_cost = 0.0
        calls_per_model = num_calls / len(models_used) if models_used else num_calls
        for model in models_used:
            pricing = PRICING[model]
            input_cost = (avg_input / 1_000_000) * pricing["input_per_mtok"] * calls_per_model
            output_cost = (avg_output / 1_000_000) * pricing["output_per_mtok"] * calls_per_model
            total_cost += input_cost + output_cost

        return {
            "benchmark": benchmark.name,
            "variants": len(variants),
            "tasks": len(tasks),
            "api_calls": num_calls,
            "estimated_cost": total_cost,
        }
