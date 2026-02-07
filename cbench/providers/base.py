"""Abstract base for provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cbench.client import CallResult
from cbench.config import ModelID


class ProviderClient(ABC):
    """Unified interface for calling LLM APIs across providers."""

    @abstractmethod
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
        """Make a synchronous API call."""
        ...

    @abstractmethod
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
        """Make a streaming API call."""
        ...
