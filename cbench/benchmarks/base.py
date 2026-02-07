from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cbench.client import CallResult


@dataclass
class BenchmarkResult:
    benchmark_name: str
    variant_name: str
    task_name: str
    run_index: int
    call_result: CallResult
    score: float
    variant_config: dict = field(default_factory=dict)


class Benchmark(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def get_variants(self) -> list[dict]:
        """Return list of variant configs. Each has at least 'name' key."""
        ...

    @abstractmethod
    def build_call_kwargs(self, variant: dict) -> dict:
        """Return kwargs for BenchClient.call_sync/call_streaming (thinking, effort, temperature, etc.)."""
        ...

    def get_model(self, variant: dict):
        from cbench.config import ModelID

        return variant.get("model", ModelID.OPUS_4_6)

    def uses_streaming(self, variant: dict) -> bool:
        return False
