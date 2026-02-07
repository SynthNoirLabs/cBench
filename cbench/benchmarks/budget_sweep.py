from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID


class BudgetSweepBenchmark(Benchmark):
    name = "budget_sweep"
    description = "Manual thinking budget_tokens sweep: 1024 to 32768"

    def get_variants(self) -> list[dict[str, Any]]:
        budgets = [1024, 2048, 4096, 8192, 16384, 32768]
        return [{"name": f"budget_{b}", "budget": b, "model": ModelID.OPUS_4_6} for b in budgets]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        return {
            "thinking": {"type": "enabled", "budget_tokens": variant["budget"]},
        }
