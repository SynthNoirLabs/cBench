from cbench.benchmarks.base import Benchmark, BenchmarkResult
from cbench.benchmarks.budget_sweep import BudgetSweepBenchmark
from cbench.benchmarks.caching import CachingBenchmark
from cbench.benchmarks.effort_levels import EffortLevelsBenchmark
from cbench.benchmarks.model_comparison import ModelComparisonBenchmark
from cbench.benchmarks.streaming import StreamingBenchmark
from cbench.benchmarks.temperature import TemperatureBenchmark
from cbench.benchmarks.thinking_modes import ThinkingModesBenchmark

BENCHMARK_REGISTRY: dict[str, Benchmark] = {
    "thinking_modes": ThinkingModesBenchmark(),
    "effort_levels": EffortLevelsBenchmark(),
    "budget_sweep": BudgetSweepBenchmark(),
    "model_comparison": ModelComparisonBenchmark(),
    "temperature": TemperatureBenchmark(),
    "streaming": StreamingBenchmark(),
    "caching": CachingBenchmark(),
}


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    return BENCHMARK_REGISTRY[name]


def list_benchmarks() -> list[dict]:
    return [
        {"name": b.name, "description": b.description, "variants": len(b.get_variants())}
        for b in BENCHMARK_REGISTRY.values()
    ]
