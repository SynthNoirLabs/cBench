"""OpenRouter provider client — uses OpenAI SDK with custom base URL."""

from __future__ import annotations

import os
import time
from typing import Any

from cbench.client import CallResult
from cbench.config import ModelID
from cbench.metrics import calculate_cost
from cbench.providers.base import ProviderClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ProviderClient):
    """OpenRouter API client (Qwen, DeepSeek, etc. via OpenAI-compatible API)."""

    def __init__(self) -> None:
        try:
            import openai

            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            self.client = openai.OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                max_retries=3,
            )
        except ImportError as err:
            raise ImportError(
                "openai package required for OpenRouter models. "
                "Install with: pip install 'cbench[openai]' or pip install openai"
            ) from err

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
        oai_messages = self._build_messages(system, messages)
        kwargs = self._build_kwargs(model, oai_messages, temperature, max_tokens)

        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            elapsed = time.monotonic() - start
            return CallResult(wall_clock_seconds=elapsed, error=str(e))
        elapsed = time.monotonic() - start

        return self._parse_response(response, model, elapsed)

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
        oai_messages = self._build_messages(system, messages)
        kwargs = self._build_kwargs(model, oai_messages, temperature, max_tokens)
        kwargs["stream"] = True

        text_parts: list[str] = []
        ttfb: float | None = None
        start = time.monotonic()
        input_tokens = 0
        output_tokens = 0

        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if ttfb is None and chunk.choices and chunk.choices[0].delta.content:
                    ttfb = time.monotonic() - start
                if chunk.choices and chunk.choices[0].delta.content:
                    text_parts.append(chunk.choices[0].delta.content)
                if chunk.usage:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
        except Exception as e:
            elapsed = time.monotonic() - start
            return CallResult(wall_clock_seconds=elapsed, error=str(e))

        elapsed = time.monotonic() - start
        response_text = "".join(text_parts)

        usage_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        cost = calculate_cost(model, usage_dict)

        return CallResult(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_clock_seconds=elapsed,
            time_to_first_token_seconds=ttfb,
            response_text=response_text,
            response_length=len(response_text),
            cost_usd=cost,
        )

    def _build_messages(
        self, system: str | list[dict[str, Any]] | None, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        oai_messages: list[dict[str, Any]] = []
        if system is not None:
            if isinstance(system, str):
                oai_messages.append({"role": "system", "content": system})
            else:
                text = " ".join(
                    b["text"] for b in system if "text" in b
                )
                if text:
                    oai_messages.append({"role": "system", "content": text})
        oai_messages.extend(messages)
        return oai_messages

    def _build_kwargs(
        self,
        model: ModelID,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model.value,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    def _parse_response(
        self, response: Any, model: ModelID, elapsed: float
    ) -> CallResult:
        choice = response.choices[0] if response.choices else None
        response_text = choice.message.content or "" if choice else ""

        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        usage_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        cost = calculate_cost(model, usage_dict)

        return CallResult(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_clock_seconds=elapsed,
            response_text=response_text,
            response_length=len(response_text),
            cost_usd=cost,
        )
