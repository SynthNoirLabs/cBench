from __future__ import annotations

import time
from dataclasses import dataclass

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


class BenchClient:
    def __init__(self):
        self.client = anthropic.Anthropic(max_retries=3)

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
        kwargs = self._build_kwargs(
            model=model,
            messages=messages,
            system=system,
            thinking=thinking,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
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
        messages: list[dict],
        *,
        system: str | list[dict] | None = None,
        thinking: dict | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CallResult:
        kwargs = self._build_kwargs(
            model=model,
            messages=messages,
            system=system,
            thinking=thinking,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text_parts = []
        thinking_parts = []
        ttfb = None
        start = time.monotonic()

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if ttfb is None and hasattr(event, "type"):
                        if event.type in ("content_block_delta",):
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
        messages: list[dict],
        system: str | list[dict] | None,
        thinking: dict | None,
        effort: str | None,
        temperature: float | None,
        max_tokens: int,
    ) -> dict:
        kwargs: dict = {
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

        return kwargs

    def _parse_response(self, response, model: ModelID, elapsed: float) -> CallResult:
        text_parts = []
        thinking_parts = []
        has_thinking = False

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
                has_thinking = True

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
        )
