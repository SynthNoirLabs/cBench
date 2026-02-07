from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID


class ModelComparisonBenchmark(Benchmark):
    name = "model_comparison"
    description = "Compare Claude, GPT, Gemini, and open models"

    def get_variants(self) -> list[dict]:
        return [
            # Claude models
            {"name": "opus_adaptive", "model": ModelID.OPUS_4_6, "thinking": {"type": "adaptive"}},
            {
                "name": "sonnet_manual",
                "model": ModelID.SONNET_4_5,
                "thinking": {"type": "enabled", "budget_tokens": 8192},
            },
            {"name": "haiku_no_thinking", "model": ModelID.HAIKU_4_5},
            # OpenAI models
            {"name": "gpt5", "model": ModelID.GPT_5},
            {"name": "gpt4o", "model": ModelID.GPT_4O},
            {"name": "o3", "model": ModelID.O3},
            # Google models
            {"name": "gemini3_pro", "model": ModelID.GEMINI_3_PRO},
            {"name": "gemini3_flash", "model": ModelID.GEMINI_3_FLASH},
            # Open models (via OpenRouter)
            {"name": "qwen3_coder", "model": ModelID.QWEN3_CODER},
            {"name": "deepseek_v3", "model": ModelID.DEEPSEEK_V3},
        ]

    def build_call_kwargs(self, variant: dict) -> dict:
        kwargs = {}
        if "thinking" in variant:
            kwargs["thinking"] = variant["thinking"]
        return kwargs
