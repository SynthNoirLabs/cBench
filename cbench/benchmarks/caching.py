from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID

LARGE_SYSTEM_PROMPT = (
    "You are an expert assistant. " * 200 + "Always answer concisely and accurately."
)


class CachingBenchmark(Benchmark):
    name = "caching"
    description = "Prompt caching effectiveness (3 rapid calls per task)"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {"name": "call_1_cold", "call_index": 0, "model": ModelID.OPUS_4_6},
            {"name": "call_2_warm", "call_index": 1, "model": ModelID.OPUS_4_6},
            {"name": "call_3_hot", "call_index": 2, "model": ModelID.OPUS_4_6},
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        return {
            "system": [
                {
                    "type": "text",
                    "text": LARGE_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
