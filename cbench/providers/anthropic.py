"""Anthropic provider — wraps the existing BenchClient logic."""

from __future__ import annotations

from typing import Any

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
        messages: list[dict[str, Any]],
        *,
        system: str | list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        betas: list[str] | None = None,
    ) -> CallResult:
        return self._client.call_sync(
            model=model,
            messages=messages,
            system=system,
            thinking=thinking,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            betas=betas,
        )

    def call_streaming(
        self,
        model: ModelID,
        messages: list[dict[str, Any]],
        *,
        system: str | list[dict[str, Any]] | None = None,
        thinking: dict[str, Any] | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        betas: list[str] | None = None,
    ) -> CallResult:
        return self._client.call_streaming(
            model=model,
            messages=messages,
            system=system,
            thinking=thinking,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            betas=betas,
        )
