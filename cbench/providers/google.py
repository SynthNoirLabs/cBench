"""Google (Gemini) provider client."""

from __future__ import annotations

import time
from typing import Any

from cbench.client import CallResult
from cbench.config import ModelID
from cbench.metrics import calculate_cost
from cbench.providers.base import ProviderClient


class GoogleClient(ProviderClient):
    """Google Gemini API client."""

    def __init__(self) -> None:
        try:
            from google import genai

            self.client = genai.Client()
        except ImportError as err:
            raise ImportError(
                "google-genai package required for Gemini models. "
                "Install with: pip install 'cbench[google]' or pip install google-genai"
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
        contents = self._build_contents(messages)
        config = self._build_config(system, temperature, max_tokens)

        start = time.monotonic()
        try:
            response = self.client.models.generate_content(
                model=model.value,
                contents=contents,
                config=config,
            )
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
        contents = self._build_contents(messages)
        config = self._build_config(system, temperature, max_tokens)

        text_parts: list[str] = []
        ttfb: float | None = None
        last_response: Any = None
        start = time.monotonic()

        try:
            stream = self.client.models.generate_content_stream(
                model=model.value,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                if ttfb is None and chunk.text:
                    ttfb = time.monotonic() - start
                if chunk.text:
                    text_parts.append(chunk.text)
                last_response = chunk
        except Exception as e:
            elapsed = time.monotonic() - start
            return CallResult(wall_clock_seconds=elapsed, error=str(e))

        elapsed = time.monotonic() - start
        response_text = "".join(text_parts)

        # Extract usage from the last chunk
        input_tokens = 0
        output_tokens = 0
        if last_response and hasattr(last_response, "usage_metadata"):
            meta = last_response.usage_metadata
            input_tokens = getattr(meta, "prompt_token_count", 0) or 0
            output_tokens = getattr(meta, "candidates_token_count", 0) or 0

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

    def _build_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert standard messages to Gemini format."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return contents

    def _build_config(
        self,
        system: str | list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int,
    ) -> Any:
        from google.genai import types

        config: dict[str, Any] = {"max_output_tokens": max_tokens}

        if system is not None:
            if isinstance(system, str):
                config["system_instruction"] = system
            else:
                text = " ".join(
                    b["text"] for b in system if "text" in b
                )
                if text:
                    config["system_instruction"] = text

        if temperature is not None:
            config["temperature"] = temperature

        return types.GenerateContentConfig(**config)

    def _parse_response(
        self, response: Any, model: ModelID, elapsed: float
    ) -> CallResult:
        response_text = response.text or ""

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            input_tokens = getattr(meta, "prompt_token_count", 0) or 0
            output_tokens = getattr(meta, "candidates_token_count", 0) or 0

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
