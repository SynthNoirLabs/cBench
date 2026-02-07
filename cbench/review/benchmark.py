"""Review benchmark definition with default model/thinking variants."""

from __future__ import annotations

from cbench.config import ModelID


class ReviewBenchmark:
    """Defines the set of model+config variants to benchmark for repo review."""

    name = "repo_review"
    description = "Multi-turn repo review with LLM-as-Judge scoring"

    DEFAULT_VARIANTS = [
        # Raw API variants (hand-rolled agent loop with anthropic SDK)
        {
            "name": "opus_adaptive",
            "model": ModelID.OPUS_4_6,
            "agent_type": "raw_api",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "effort": None,
            "max_tokens": 16384,
            "description": "Opus 4.6 with adaptive thinking",
        },
        {
            "name": "opus_effort_high",
            "model": ModelID.OPUS_4_6,
            "agent_type": "raw_api",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "effort": "high",
            "max_tokens": 16384,
            "description": "Opus 4.6 with adaptive thinking + effort=high",
        },
        {
            "name": "sonnet_adaptive",
            "model": ModelID.SONNET_4_5,
            "agent_type": "raw_api",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "effort": None,
            "max_tokens": 16384,
            "description": "Sonnet 4.5 with adaptive thinking",
        },
        {
            "name": "haiku_no_thinking",
            "model": ModelID.HAIKU_4_5,
            "agent_type": "raw_api",
            "thinking": None,
            "effort": None,
            "max_tokens": 8192,
            "description": "Haiku 4.5 without thinking",
        },
        # Agent SDK variants (Claude Code Agent SDK with built-in tools)
        {
            "name": "sdk_opus",
            "model": ModelID.OPUS_4_6,
            "agent_type": "sdk",
            "thinking": None,
            "effort": None,
            "max_tokens": 16384,
            "description": "Opus 4.6 via Claude Code Agent SDK",
        },
        {
            "name": "sdk_sonnet",
            "model": ModelID.SONNET_4_5,
            "agent_type": "sdk",
            "thinking": None,
            "effort": None,
            "max_tokens": 16384,
            "description": "Sonnet 4.5 via Claude Code Agent SDK",
        },
        {
            "name": "sdk_haiku",
            "model": ModelID.HAIKU_4_5,
            "agent_type": "sdk",
            "thinking": None,
            "effort": None,
            "max_tokens": 8192,
            "description": "Haiku 4.5 via Claude Code Agent SDK",
        },
    ]

    def __init__(self, variants: list[dict] | None = None) -> None:
        self.variants = variants or self.DEFAULT_VARIANTS

    def get_variants(self) -> list[dict]:
        return self.variants
