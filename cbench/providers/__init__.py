"""Multi-provider support for cBench."""

from __future__ import annotations

from cbench.config import PROVIDER_MAP, ModelID, Provider
from cbench.providers.base import ProviderClient


def create_client(model: ModelID) -> ProviderClient:
    """Create the appropriate provider client for a given model."""
    provider = PROVIDER_MAP[model]

    if provider == Provider.ANTHROPIC:
        from cbench.providers.anthropic import AnthropicClient

        return AnthropicClient()

    if provider == Provider.OPENAI:
        from cbench.providers.openai import OpenAIClient

        return OpenAIClient()

    if provider == Provider.GOOGLE:
        from cbench.providers.google import GoogleClient

        return GoogleClient()

    if provider == Provider.OPENROUTER:
        from cbench.providers.openrouter import OpenRouterClient

        return OpenRouterClient()

    raise ValueError(f"Unknown provider for model {model}: {provider}")
