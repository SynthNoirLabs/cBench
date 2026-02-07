from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class ModelID(str, Enum):
    # Claude (Anthropic)
    OPUS_4_6 = "claude-opus-4-6"
    SONNET_4_5 = "claude-sonnet-4-5-20250929"
    HAIKU_4_5 = "claude-haiku-4-5-20251001"
    # OpenAI
    GPT_5 = "gpt-5"
    GPT_4O = "gpt-4o"
    O3 = "o3"
    # Google
    GEMINI_3_PRO = "gemini-3.0-pro"
    GEMINI_3_FLASH = "gemini-3.0-flash"
    # Open (via OpenRouter)
    QWEN3_CODER = "qwen/qwen3-coder-next"
    DEEPSEEK_V3 = "deepseek/deepseek-chat-v3"


PROVIDER_MAP: dict[ModelID, Provider] = {
    ModelID.OPUS_4_6: Provider.ANTHROPIC,
    ModelID.SONNET_4_5: Provider.ANTHROPIC,
    ModelID.HAIKU_4_5: Provider.ANTHROPIC,
    ModelID.GPT_5: Provider.OPENAI,
    ModelID.GPT_4O: Provider.OPENAI,
    ModelID.O3: Provider.OPENAI,
    ModelID.GEMINI_3_PRO: Provider.GOOGLE,
    ModelID.GEMINI_3_FLASH: Provider.GOOGLE,
    ModelID.QWEN3_CODER: Provider.OPENROUTER,
    ModelID.DEEPSEEK_V3: Provider.OPENROUTER,
}


LONG_CONTEXT_THRESHOLD = 200_000

PRICING: dict[ModelID, dict] = {
    # Anthropic
    ModelID.OPUS_4_6: {
        "input_per_mtok": 5.00,
        "output_per_mtok": 25.00,
        "batch_input_per_mtok": 2.50,
        "batch_output_per_mtok": 12.50,
        "cache_write_multiplier": 1.25,
        "cache_read_multiplier": 0.1,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    ModelID.SONNET_4_5: {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
        "batch_input_per_mtok": 1.50,
        "batch_output_per_mtok": 7.50,
        "cache_write_multiplier": 1.25,
        "cache_read_multiplier": 0.1,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    ModelID.HAIKU_4_5: {
        "input_per_mtok": 1.00,
        "output_per_mtok": 5.00,
        "batch_input_per_mtok": 0.50,
        "batch_output_per_mtok": 2.50,
        "cache_write_multiplier": 1.25,
        "cache_read_multiplier": 0.1,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    # OpenAI
    ModelID.GPT_5: {
        "input_per_mtok": 10.00,
        "output_per_mtok": 30.00,
        "batch_input_per_mtok": 5.00,
        "batch_output_per_mtok": 15.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.GPT_4O: {
        "input_per_mtok": 2.50,
        "output_per_mtok": 10.00,
        "batch_input_per_mtok": 1.25,
        "batch_output_per_mtok": 5.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.O3: {
        "input_per_mtok": 10.00,
        "output_per_mtok": 40.00,
        "batch_input_per_mtok": 5.00,
        "batch_output_per_mtok": 20.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    # Google
    ModelID.GEMINI_3_PRO: {
        "input_per_mtok": 1.25,
        "output_per_mtok": 5.00,
        "batch_input_per_mtok": 0.625,
        "batch_output_per_mtok": 2.50,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.GEMINI_3_FLASH: {
        "input_per_mtok": 0.075,
        "output_per_mtok": 0.30,
        "batch_input_per_mtok": 0.0375,
        "batch_output_per_mtok": 0.15,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    # OpenRouter
    ModelID.QWEN3_CODER: {
        "input_per_mtok": 0.80,
        "output_per_mtok": 3.20,
        "batch_input_per_mtok": 0.40,
        "batch_output_per_mtok": 1.60,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.DEEPSEEK_V3: {
        "input_per_mtok": 0.27,
        "output_per_mtok": 1.10,
        "batch_input_per_mtok": 0.135,
        "batch_output_per_mtok": 0.55,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
}


def is_claude_model(model: ModelID) -> bool:
    """Check if a model is a Claude (Anthropic) model."""
    return PROVIDER_MAP.get(model) == Provider.ANTHROPIC


@dataclass
class BenchmarkConfig:
    model: ModelID = ModelID.OPUS_4_6
    num_runs: int = 1
    benchmarks: list[str] = field(default_factory=lambda: ["all"])
    dry_run: bool = False
    no_confirm: bool = False
    output_dir: str = "results"
