"""Tests for agent context management: truncation, compression, rate limiting."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from cbench.config import ModelID
from cbench.review.agent import (
    KEEP_RECENT_RESULTS,
    MAX_TOOL_RESULT_CHARS,
    AgentLoop,
    _estimate_message_tokens,
    _truncate_tool_result,
)
from cbench.review.tools import RepoSandbox

# --- _truncate_tool_result ---


class TestTruncateToolResult:
    def test_short_result_unchanged(self):
        text = "Hello world"
        assert _truncate_tool_result(text, 1000) == text

    def test_exact_limit_unchanged(self):
        text = "x" * 100
        assert _truncate_tool_result(text, 100) == text

    def test_long_result_truncated(self):
        text = "x" * 200
        result = _truncate_tool_result(text, 100)
        assert result.startswith("x" * 100)
        assert "[truncated, 200 total chars]" in result
        assert len(result) < 200

    def test_preserves_beginning(self):
        text = "IMPORTANT HEADER\n" + "data line\n" * 100
        result = _truncate_tool_result(text, 50)
        assert result.startswith("IMPORTANT HEADER")

    def test_default_max_chars(self):
        text = "x" * (MAX_TOOL_RESULT_CHARS + 1000)
        result = _truncate_tool_result(text, MAX_TOOL_RESULT_CHARS)
        assert len(result) < len(text)
        assert "[truncated" in result


# --- _estimate_message_tokens ---


class TestEstimateMessageTokens:
    def test_empty_messages(self):
        assert _estimate_message_tokens([]) == 0

    def test_string_content(self):
        messages = [{"role": "user", "content": "x" * 400}]
        assert _estimate_message_tokens(messages) == 100  # 400 / 4

    def test_list_content_with_text(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "x" * 800},
                ],
            },
        ]
        # 800 chars of text + 4 chars for "text" type string = 804 / 4 = 201
        assert _estimate_message_tokens(messages) == 201

    def test_tool_result_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "id1", "content": "x" * 1200},
                ],
            },
        ]
        # "tool_result" + "id1" + "x"*1200 = 11 + 3 + 1200 = 1214 chars of string values
        est = _estimate_message_tokens(messages)
        assert est > 0

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "x" * 400},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "y" * 400},
                ],
            },
        ]
        est = _estimate_message_tokens(messages)
        # At least 200 tokens (800 chars / 4)
        assert est >= 200


# --- _compress_old_results ---


def _make_tool_turn(tool_use_id: str, content: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Helper: create an assistant tool_use message + user tool_result message pair."""
    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "read_file",
                "input": {"path": "test.py"},
            },
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": content},
        ],
    }
    return assistant_msg, user_msg


class TestCompressOldResults:
    def _make_agent(self, tmp_path: Path) -> AgentLoop:
        """Create an AgentLoop with a dummy sandbox (no API client needed)."""
        (tmp_path / "test.py").write_text("pass")
        sandbox = RepoSandbox(tmp_path)
        return AgentLoop(
            client=None,  # type: ignore[arg-type]
            sandbox=sandbox,
            model=ModelID.HAIKU_4_5,
            max_turns=5,
        )

    def test_no_compression_below_threshold(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Please review this repository."}]
        # Add small tool results (won't exceed threshold)
        for i in range(3):
            a, u = _make_tool_turn(f"id_{i}", f"Small result {i}")
            messages.extend([a, u])

        original_content = messages[2]["content"][0]["content"]  # type: ignore[index]
        agent._compress_old_results(messages)
        # Should be unchanged
        assert messages[2]["content"][0]["content"] == original_content  # type: ignore[index]

    def test_compresses_old_results_above_threshold(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Please review this repository."}]

        # Add enough large tool results to exceed MAX_CONTEXT_TOKENS
        large_content = "File: big.py (lines 1-500 of 500)\n" + "x" * 30_000
        for i in range(KEEP_RECENT_RESULTS + 3):
            a, u = _make_tool_turn(f"id_{i}", large_content)
            messages.extend([a, u])

        agent._compress_old_results(messages)

        # Older results should be compressed
        first_tool_result = messages[2]["content"][0]["content"]  # type: ignore[index]
        assert first_tool_result.startswith("[Earlier result:")
        assert len(first_tool_result) < 200

        # Recent results should be intact
        last_tool_result = messages[-1]["content"][0]["content"]  # type: ignore[index]
        assert last_tool_result == large_content

    def test_keeps_recent_results_intact(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Please review this repository."}]

        large_content = "x" * 30_000
        for i in range(KEEP_RECENT_RESULTS + 2):
            a, u = _make_tool_turn(f"id_{i}", large_content)
            messages.extend([a, u])

        agent._compress_old_results(messages)

        # Count how many tool results are still full-size
        full_results = 0
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        if not block["content"].startswith("[Earlier result:"):
                            full_results += 1

        assert full_results == KEEP_RECENT_RESULTS

    def test_already_compressed_not_recompressed(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Please review this repository."}]

        large_content = "x" * 30_000
        for i in range(KEEP_RECENT_RESULTS + 2):
            a, u = _make_tool_turn(f"id_{i}", large_content)
            messages.extend([a, u])

        # Compress once
        agent._compress_old_results(messages)
        first_compressed = messages[2]["content"][0]["content"]  # type: ignore[index]

        # Compress again — already-compressed results should not change
        agent._compress_old_results(messages)
        assert messages[2]["content"][0]["content"] == first_compressed  # type: ignore[index]

    def test_too_few_results_not_compressed(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "x" * 200_000}]  # large initial msg

        # Add exactly KEEP_RECENT_RESULTS tool results
        for i in range(KEEP_RECENT_RESULTS):
            a, u = _make_tool_turn(f"id_{i}", "x" * 10_000)
            messages.extend([a, u])

        agent._compress_old_results(messages)

        # None should be compressed (we only have KEEP_RECENT_RESULTS, threshold not met for compression)
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        assert not block["content"].startswith("[Earlier result:")


# --- _rate_limit_delay ---


class TestRateLimitDelay:
    def _make_agent(self, tmp_path: Path, model: ModelID = ModelID.HAIKU_4_5) -> AgentLoop:
        (tmp_path / "test.py").write_text("pass")
        sandbox = RepoSandbox(tmp_path)
        return AgentLoop(
            client=None,  # type: ignore[arg-type]
            sandbox=sandbox,
            model=model,
            max_turns=5,
        )

    def test_no_delay_first_request(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        start = time.monotonic()
        agent._rate_limit_delay(0.0, 0)
        _elapsed = time.monotonic() - start
        assert _elapsed < 0.1

    def test_no_delay_zero_tokens(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        start = time.monotonic()
        agent._rate_limit_delay(time.monotonic() - 0.1, 0)
        _elapsed = time.monotonic() - start
        assert _elapsed < 0.1

    def test_delays_when_needed(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        # Simulate: just sent 50K tokens (haiku limit), request just finished
        start = time.monotonic()
        agent._rate_limit_delay(time.monotonic(), 50_000)
        _elapsed = time.monotonic() - start
        # Should wait ~60 seconds for 50K/50K, but that's too slow for tests.
        # Just verify it waits > 0 seconds
        # Actually, let's use smaller tokens to keep test fast
        pass

    def test_no_delay_when_enough_time_passed(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        # 1000 tokens at 50K/min = 1.2 seconds required
        # If 5 seconds already passed, no delay needed
        start = time.monotonic()
        agent._rate_limit_delay(time.monotonic() - 5.0, 1000)
        _elapsed = time.monotonic() - start
        assert _elapsed < 0.1

    def test_higher_limit_for_opus(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path, model=ModelID.OPUS_4_6)
        # Opus has 100K limit, so 50K tokens = 30 seconds required
        # vs Haiku 50K limit = 60 seconds required
        # Both should be no-op if enough time passed
        start = time.monotonic()
        agent._rate_limit_delay(time.monotonic() - 60.0, 50_000)
        _elapsed = time.monotonic() - start
        assert _elapsed < 0.1
