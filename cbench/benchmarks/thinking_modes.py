from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID


class ThinkingModesBenchmark(Benchmark):
    name = "thinking_modes"
    description = "Compare adaptive vs manual vs disabled thinking"

    def get_variants(self) -> list[dict]:
        return [
            {"name": "adaptive", "thinking": {"type": "adaptive"}, "model": ModelID.OPUS_4_6},
            {
                "name": "manual_8192",
                "thinking": {"type": "enabled", "budget_tokens": 8192},
                "model": ModelID.OPUS_4_6,
            },
            {"name": "disabled", "model": ModelID.OPUS_4_6},
        ]

    def build_call_kwargs(self, variant: dict) -> dict:
        kwargs = {}
        if "thinking" in variant:
            kwargs["thinking"] = variant["thinking"]
        return kwargs
