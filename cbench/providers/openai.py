"""OpenAI provider client."""

from __future__ import annotations

import time

from cbench.client import CallResult
from cbench.config import ModelID
from cbench.metrics import calculate_cost
from cbench.providers.base import ProviderClient


class OpenAIClient(ProviderClient):
    """OpenAI API client (GPT-5.2, GPT-5.1, GPT-5-mini — all reasoning models)."""

    def __init__(self) -> None:
        try:
            import openai

            self.client = openai.OpenAI(max_retries=3)
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI models. "
                "Install with: pip install 'cbench[openai]' or pip install openai"
            )

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
        oai_messages = self._build_messages(system, messages)
        kwargs = self._build_kwargs(model, oai_messages, temperature, max_tokens, effort=effort)

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
        messages: list[dict],
        *,
        system: str | list[dict] | None = None,
        thinking: dict | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CallResult:
        oai_messages = self._build_messages(system, messages)
        kwargs = self._build_kwargs(model, oai_messages, temperature, max_tokens, effort=effort)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        text_parts = []
        ttfb = None
        usage = None
        start = time.monotonic()

        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if ttfb is None and chunk.choices and chunk.choices[0].delta.content:
                    ttfb = time.monotonic() - start
                if chunk.choices and chunk.choices[0].delta.content:
                    text_parts.append(chunk.choices[0].delta.content)
                if chunk.usage:
                    usage = chunk.usage
        except Exception as e:
            elapsed = time.monotonic() - start
            return CallResult(wall_clock_seconds=elapsed, error=str(e))

        elapsed = time.monotonic() - start
        response_text = "".join(text_parts)

        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

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
        self, system: str | list[dict] | None, messages: list[dict]
    ) -> list[dict]:
        oai_messages = []
        if system is not None:
            if isinstance(system, str):
                oai_messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                # Convert Anthropic-style system blocks to plain string
                text = " ".join(
                    b["text"] for b in system if isinstance(b, dict) and "text" in b
                )
                if text:
                    oai_messages.append({"role": "system", "content": text})
        oai_messages.extend(messages)
        return oai_messages

    # Map cBench effort names to OpenAI reasoning_effort values
    _EFFORT_MAP = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": "high",  # OpenAI has no "max", map to high
        "minimal": "minimal",  # GPT-5 family only
        "xhigh": "xhigh",  # GPT-5.2 only
    }

    def _build_kwargs(
        self,
        model: ModelID,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int,
        *,
        effort: str | None = None,
    ) -> dict:
        kwargs: dict = {
            "model": model.value,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if effort is not None:
            kwargs["reasoning_effort"] = self._EFFORT_MAP.get(effort, effort)
        return kwargs

    def _parse_response(
        self, response, model: ModelID, elapsed: float
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
