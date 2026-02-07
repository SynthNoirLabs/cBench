"""Large output benchmark — tests output capacity at 16K, 64K, and 128K tokens."""

from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import BETA_EXTENDED_OUTPUT, ModelID
from cbench.tasks.base import ScoringMethod, TaskDefinition


class LargeOutputBenchmark(Benchmark):
    name = "large_output"
    description = "Output capacity test at 16K / 64K / 128K max output tokens"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "16k_output",
                "max_tokens": 16_384,
                "model": ModelID.OPUS_4_6,
            },
            {
                "name": "64k_output",
                "max_tokens": 65_536,
                "model": ModelID.OPUS_4_6,
            },
            {
                "name": "128k_output",
                "max_tokens": 131_072,
                "model": ModelID.OPUS_4_6,
                "betas": [BETA_EXTENDED_OUTPUT],
            },
        ]

    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="exhaustive_enumeration",
                category="large_output",
                prompt=(
                    "Generate a comprehensive numbered list of historical events, scientific "
                    "discoveries, and technological inventions from 1 CE to 2025 CE. For each "
                    "entry include: the number, the year, a title, and a 2-3 sentence description. "
                    "Start at entry 1 and continue sequentially. Do NOT stop early, do NOT "
                    "summarize, do NOT skip entries. Generate as many entries as possible. "
                    "Format each as:\n"
                    "N. [YEAR] Title — Description.\n\n"
                    "Begin now with entry 1."
                ),
                expected_answer="100.",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="hard",
            ),
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_tokens": variant["max_tokens"],
        }
        if "betas" in variant:
            kwargs["betas"] = variant["betas"]
        return kwargs
