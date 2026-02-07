from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cbench.client import CallResult
from cbench.config import ModelID
from cbench.tasks.base import TaskDefinition


@dataclass
class BenchmarkResult:
    benchmark_name: str
    variant_name: str
    task_name: str
    run_index: int
    call_result: CallResult
    score: float
    variant_config: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def get_variants(self) -> list[dict[str, Any]]:
        """Return list of variant configs. Each has at least 'name' key."""
        ...

    @abstractmethod
    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        """Return kwargs for BenchClient.call_sync/call_streaming (thinking, effort, temperature, etc.)."""
        ...

    def get_model(self, variant: dict[str, Any]) -> ModelID:
        model: ModelID = variant.get("model", ModelID.OPUS_4_6)
        return model

    def get_tasks(self) -> list[TaskDefinition] | None:
        """Override to provide benchmark-specific tasks instead of using global tasks."""
        return None

    def uses_streaming(self, variant: dict[str, Any]) -> bool:
        return False
