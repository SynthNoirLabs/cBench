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
    # OpenAI (all GPT-5 family — reasoning models with thinking)
    GPT_5_2 = "gpt-5.2"
    GPT_5_1 = "gpt-5.1"
    GPT_5_MINI = "gpt-5-mini"
    # Google
    GEMINI_3_PRO = "gemini-3-pro-preview"
    GEMINI_3_FLASH = "gemini-3-flash-preview"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    # Open (via OpenRouter)
    QWEN3_CODER = "qwen/qwen3-coder-480b-a35b"
    DEEPSEEK_V3 = "deepseek/deepseek-chat"


PROVIDER_MAP: dict[ModelID, Provider] = {
    ModelID.OPUS_4_6: Provider.ANTHROPIC,
    ModelID.SONNET_4_5: Provider.ANTHROPIC,
    ModelID.HAIKU_4_5: Provider.ANTHROPIC,
    ModelID.GPT_5_2: Provider.OPENAI,
    ModelID.GPT_5_1: Provider.OPENAI,
    ModelID.GPT_5_MINI: Provider.OPENAI,
    ModelID.GEMINI_3_PRO: Provider.GOOGLE,
    ModelID.GEMINI_3_FLASH: Provider.GOOGLE,
    ModelID.GEMINI_2_5_PRO: Provider.GOOGLE,
    ModelID.GEMINI_2_5_FLASH: Provider.GOOGLE,
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
    ModelID.GPT_5_2: {
        "input_per_mtok": 1.75,
        "output_per_mtok": 14.00,
        "batch_input_per_mtok": 0.875,
        "batch_output_per_mtok": 7.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.1,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.GPT_5_1: {
        "input_per_mtok": 1.25,
        "output_per_mtok": 10.00,
        "batch_input_per_mtok": 0.625,
        "batch_output_per_mtok": 5.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.1,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.GPT_5_MINI: {
        "input_per_mtok": 0.25,
        "output_per_mtok": 2.00,
        "batch_input_per_mtok": 0.125,
        "batch_output_per_mtok": 1.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.025,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    # Google
    ModelID.GEMINI_3_PRO: {
        "input_per_mtok": 2.00,
        "output_per_mtok": 12.00,
        "batch_input_per_mtok": 1.00,
        "batch_output_per_mtok": 6.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.25,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    ModelID.GEMINI_3_FLASH: {
        "input_per_mtok": 0.50,
        "output_per_mtok": 3.00,
        "batch_input_per_mtok": 0.25,
        "batch_output_per_mtok": 1.50,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.1,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.GEMINI_2_5_PRO: {
        "input_per_mtok": 1.25,
        "output_per_mtok": 10.00,
        "batch_input_per_mtok": 0.625,
        "batch_output_per_mtok": 5.00,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.25,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
    },
    ModelID.GEMINI_2_5_FLASH: {
        "input_per_mtok": 0.30,
        "output_per_mtok": 2.50,
        "batch_input_per_mtok": 0.15,
        "batch_output_per_mtok": 1.25,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.25,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    # OpenRouter
    ModelID.QWEN3_CODER: {
        "input_per_mtok": 0.22,
        "output_per_mtok": 0.95,
        "batch_input_per_mtok": 0.11,
        "batch_output_per_mtok": 0.475,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.0,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
    ModelID.DEEPSEEK_V3: {
        "input_per_mtok": 0.28,
        "output_per_mtok": 0.42,
        "batch_input_per_mtok": 0.14,
        "batch_output_per_mtok": 0.21,
        "cache_write_multiplier": 1.0,
        "cache_read_multiplier": 0.1,
        "long_context_input_multiplier": 1.0,
        "long_context_output_multiplier": 1.0,
    },
}


def is_claude_model(model: ModelID) -> bool:
    """Check if a model is a Claude (Anthropic) model."""
    return PROVIDER_MAP.get(model) == Provider.ANTHROPIC


def is_openai_reasoning_model(model: ModelID) -> bool:
    """Check if a model is an OpenAI GPT-5 family reasoning model."""
    return model in (ModelID.GPT_5_2, ModelID.GPT_5_1, ModelID.GPT_5_MINI)


@dataclass
class BenchmarkConfig:
    model: ModelID = ModelID.OPUS_4_6
    num_runs: int = 1
    benchmarks: list[str] = field(default_factory=lambda: ["all"])
    dry_run: bool = False
    no_confirm: bool = False
    output_dir: str = "results"
