from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID


class ModelComparisonBenchmark(Benchmark):
    name = "model_comparison"
    description = "Compare Claude, GPT, Gemini, and open models"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            # Claude models
            {"name": "opus_adaptive", "model": ModelID.OPUS_4_6, "thinking": {"type": "adaptive"}},
            {
                "name": "sonnet_manual",
                "model": ModelID.SONNET_4_5,
                "thinking": {"type": "enabled", "budget_tokens": 8192},
            },
            {"name": "haiku_no_thinking", "model": ModelID.HAIKU_4_5},
            # OpenAI models (all GPT-5 family — reasoning with thinking)
            {"name": "gpt5_2", "model": ModelID.GPT_5_2},
            {"name": "gpt5_1", "model": ModelID.GPT_5_1},
            {"name": "gpt5_mini", "model": ModelID.GPT_5_MINI},
            # Google models
            {"name": "gemini3_pro", "model": ModelID.GEMINI_3_PRO},
            {"name": "gemini3_flash", "model": ModelID.GEMINI_3_FLASH},
            {"name": "gemini2_5_pro", "model": ModelID.GEMINI_2_5_PRO},
            {"name": "gemini2_5_flash", "model": ModelID.GEMINI_2_5_FLASH},
            # Open models (via OpenRouter)
            {"name": "qwen3_coder", "model": ModelID.QWEN3_CODER},
            {"name": "deepseek_v3", "model": ModelID.DEEPSEEK_V3},
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if "thinking" in variant:
            kwargs["thinking"] = variant["thinking"]
        return kwargs
