from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from cbench.config import ModelID
from cbench.metrics import calculate_cost


@dataclass
class CallResult:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    wall_clock_seconds: float = 0.0
    time_to_first_token_seconds: float | None = None
    response_text: str = ""
    thinking_text: str = ""
    has_thinking: bool = False
    response_length: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class BenchClient:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(max_retries=3)

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
        kwargs = self._build_kwargs(
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

        start = time.monotonic()
        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
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
        kwargs = self._build_kwargs(
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

        ttfb = None
        start = time.monotonic()

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if ttfb is None and hasattr(event, "type") and event.type in ("content_block_delta",):
                        ttfb = time.monotonic() - start
                    # Collect text from the stream
                response = stream.get_final_message()
        except anthropic.APIStatusError as e:
            elapsed = time.monotonic() - start
            return CallResult(wall_clock_seconds=elapsed, error=str(e))

        elapsed = time.monotonic() - start
        result = self._parse_response(response, model, elapsed)
        result.time_to_first_token_seconds = ttfb
        return result

    def _build_kwargs(
        self,
        model: ModelID,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None,
        thinking: dict[str, Any] | None,
        effort: str | None,
        temperature: float | None,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        betas: list[str] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model.value,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if system is not None:
            kwargs["system"] = system

        if thinking is not None:
            kwargs["thinking"] = thinking
            # When thinking is enabled, must not set temperature
            # and max_tokens might need to be higher to accommodate thinking
        elif temperature is not None:
            kwargs["temperature"] = temperature

        if effort is not None:
            kwargs["output_config"] = {"effort": effort}

        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        if betas is not None:
            kwargs["extra_headers"] = {"anthropic-beta": ",".join(betas)}

        return kwargs

    def _parse_response(self, response: Any, model: ModelID, elapsed: float) -> CallResult:
        text_parts = []
        thinking_parts = []
        tool_calls = []
        has_thinking = False

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
                has_thinking = True
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        response_text = "\n".join(text_parts)
        thinking_text = "\n".join(thinking_parts)

        usage = response.usage
        usage_dict = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        }

        cost = calculate_cost(model, usage_dict)

        return CallResult(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage_dict["cache_creation_input_tokens"],
            cache_read_input_tokens=usage_dict["cache_read_input_tokens"],
            wall_clock_seconds=elapsed,
            response_text=response_text,
            thinking_text=thinking_text,
            has_thinking=has_thinking,
            response_length=len(response_text),
            cost_usd=cost,
            tool_calls=tool_calls,
        )
