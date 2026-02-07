from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID


class EffortLevelsBenchmark(Benchmark):
    name = "effort_levels"
    description = "Compare effort levels with adaptive thinking"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {"name": f"effort_{level}", "effort": level, "model": ModelID.OPUS_4_6}
            for level in ("low", "medium", "high", "max")
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        return {
            "thinking": {"type": "adaptive"},
            "effort": variant["effort"],
        }
