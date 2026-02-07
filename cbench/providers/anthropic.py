"""Anthropic provider — wraps the existing BenchClient logic."""

from __future__ import annotations

from cbench.client import BenchClient, CallResult
from cbench.config import ModelID
from cbench.providers.base import ProviderClient


class AnthropicClient(ProviderClient):
    """Anthropic API client (Claude models)."""

    def __init__(self) -> None:
        self._client = BenchClient()

    def call_sync(
        self,
        model: ModelID,
        messages: list[dict],
        *,
        system: str | list[dict] | None = None,
        thinking: dict | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CallResult:
        return self._client.call_sync(
            model=model,
            messages=messages,
            system=system,
            thinking=thinking,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def call_streaming(
        self,
        model: ModelID,
        messages: list[dict],
        *,
        system: str | list[dict] | None = None,
        thinking: dict | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CallResult:
        return self._client.call_streaming(
            model=model,
            messages=messages,
            system=system,
            thinking=thinking,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )
