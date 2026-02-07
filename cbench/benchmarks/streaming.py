from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID


class StreamingBenchmark(Benchmark):
    name = "streaming"
    description = "Streaming vs non-streaming latency comparison"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {"name": "sync", "streaming": False, "model": ModelID.OPUS_4_6},
            {"name": "streaming", "streaming": True, "model": ModelID.OPUS_4_6},
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        return {}

    def uses_streaming(self, variant: dict[str, Any]) -> bool:
        return bool(variant.get("streaming", False))
