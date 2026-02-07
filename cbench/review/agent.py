"""Multi-turn agent loop: model explores a repo using tools, produces a review."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

from cbench.config import PRICING, ModelID
from cbench.review.tools import TOOL_DEFINITIONS, RepoSandbox, dispatch_tool

# Context management constants
MAX_TOOL_RESULT_CHARS = 6000  # Max chars per individual tool result (~1500 tokens)
MAX_CONTEXT_TOKENS = 30_000  # Compress old results when estimated context exceeds this
KEEP_RECENT_RESULTS = 4  # Number of recent tool-result turns to keep uncompressed

SYSTEM_PROMPT = """\
You are an expert code reviewer. You have access to tools that let you explore a code repository. \
Your task is to produce a thorough, actionable code review of the repository.

Explore the repo systematically:
1. Start by listing the root directory to understand the project structure.
2. Read key files (README, config files, entry points) to understand the purpose and architecture.
3. Dive into source code to assess quality, patterns, and potential issues.
4. Search for common anti-patterns, security issues, or code smells.

Your final review should cover:
- **Architecture & Design**: Overall structure, separation of concerns, design patterns
- **Code Quality**: Readability, naming, DRY principle, complexity
- **Error Handling**: Robustness, edge cases, failure modes
- **Security**: Input validation, injection risks, secrets management
- **Testing**: Test coverage adequacy, test quality
- **Performance**: Obvious bottlenecks, resource management
- **Documentation**: README quality, code comments, API docs
- **Dependencies**: Dependency health, version pinning, licensing concerns
- **Actionable Recommendations**: Prioritized list of improvements

Be specific: reference file names, line numbers, and code snippets. Prioritize findings by severity.\
"""


@dataclass
class TurnMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AgentMetrics:
    total_turns: int = 0
    tool_calls: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    files_read: int = 0
    directories_listed: int = 0
    searches_performed: int = 0
    per_turn: list[TurnMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0


@dataclass
class AgentResult:
    review_text: str = ""
    thinking_text: str = ""
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    conversation_history: list[dict] = field(default_factory=list)
    error: str | None = None


class AgentLoop:
    """Multi-turn agent: sends messages with tools, executes tool calls, loops until done."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        sandbox: RepoSandbox,
        model: ModelID,
        thinking: dict | None = None,
        effort: str | None = None,
        max_tokens: int = 16384,
        max_turns: int = 30,
    ) -> None:
        self.client = client
        self.sandbox = sandbox
        self.model = model
        self.thinking = thinking
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_turns = max_turns

    def run(self) -> AgentResult:
        metrics = AgentMetrics()
        messages: list[dict] = [
            {"role": "user", "content": "Please review this repository."},
        ]
        review_parts: list[str] = []
        thinking_parts: list[str] = []
        start_time = time.monotonic()
        last_request_time = 0.0
        last_input_tokens = 0

        for turn in range(self.max_turns):
            # Compress old tool results if context is getting large
            self._compress_old_results(messages)

            # Rate-limit-aware delay: avoid exceeding input tokens/minute
            self._rate_limit_delay(last_request_time, last_input_tokens)

            kwargs = self._build_kwargs(messages)
            last_request_time = time.monotonic()
            try:
                response = self.client.messages.create(**kwargs)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                return AgentResult(
                    metrics=metrics,
                    conversation_history=messages,
                    error=str(e),
                )
            except Exception as e:
                return AgentResult(
                    metrics=metrics,
                    conversation_history=messages,
                    error=f"{type(e).__name__}: {e}",
                )

            # Track token usage
            usage = response.usage
            last_input_tokens = usage.input_tokens
            turn_cost = self._calculate_turn_cost(usage)
            turn_metrics = TurnMetrics(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=turn_cost,
            )
            metrics.per_turn.append(turn_metrics)
            metrics.total_turns += 1
            metrics.total_input_tokens += usage.input_tokens
            metrics.total_output_tokens += usage.output_tokens
            metrics.total_cost_usd += turn_cost

            # Parse response content blocks
            assistant_content = []
            turn_text_parts = []
            turn_thinking_parts = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
                    turn_text_parts.append(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "thinking":
                    turn_thinking_parts.append(block.thinking)
                    assistant_content.append(
                        {
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": block.signature,
                        }
                    )
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_content})

            # Accumulate text
            if turn_text_parts:
                review_parts.append("\n".join(turn_text_parts))
            if turn_thinking_parts:
                thinking_parts.append("\n".join(turn_thinking_parts))

            # If stop_reason is end_turn (no more tool calls), we're done
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                break

            # Execute tool calls and build tool_result messages
            tool_results = []
            for tool_block in tool_use_blocks:
                metrics.tool_calls += 1
                name = tool_block.name
                metrics.tool_calls_by_name[name] = metrics.tool_calls_by_name.get(name, 0) + 1
                if name == "read_file":
                    metrics.files_read += 1
                elif name == "list_directory":
                    metrics.directories_listed += 1
                elif name == "search_code":
                    metrics.searches_performed += 1

                result_text = dispatch_tool(self.sandbox, name, tool_block.input)
                # Truncate individual results to prevent any single tool call
                # from adding too much context
                result_text = _truncate_tool_result(result_text, MAX_TOOL_RESULT_CHARS)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result_text,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        metrics.wall_clock_seconds = time.monotonic() - start_time

        return AgentResult(
            review_text="\n\n".join(review_parts),
            thinking_text="\n\n".join(thinking_parts),
            metrics=metrics,
            conversation_history=messages,
        )

    def _build_kwargs(self, messages: list[dict]) -> dict:
        kwargs: dict = {
            "model": self.model.value,
            "messages": messages,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "max_tokens": self.max_tokens,
            "tools": TOOL_DEFINITIONS,
        }
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def _calculate_turn_cost(self, usage) -> float:
        pricing = PRICING[self.model]
        input_cost = (usage.input_tokens / 1_000_000) * pricing["input_per_mtok"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output_per_mtok"]
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write_cost = (
            (cache_write / 1_000_000)
            * pricing["input_per_mtok"]
            * pricing["cache_write_multiplier"]
        )
        cache_read_cost = (
            (cache_read / 1_000_000) * pricing["input_per_mtok"] * pricing["cache_read_multiplier"]
        )
        return input_cost + output_cost + cache_write_cost + cache_read_cost

    def _compress_old_results(self, messages: list[dict]) -> None:
        """In-place compression: replace old tool result contents with short summaries.

        When estimated context tokens exceed MAX_CONTEXT_TOKENS, this replaces the
        content of old tool_result blocks with a one-line summary, keeping only the
        most recent KEEP_RECENT_RESULTS tool-result turns intact.
        """
        est_tokens = _estimate_message_tokens(messages)
        if est_tokens < MAX_CONTEXT_TOKENS:
            return

        # Find indices of user messages that contain tool_results
        result_indices = []
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                if any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in msg["content"]
                ):
                    result_indices.append(i)

        if len(result_indices) <= KEEP_RECENT_RESULTS:
            return  # Not enough history to compress

        # Compress all but the most recent tool-result turns
        to_compress = result_indices[:-KEEP_RECENT_RESULTS]
        for idx in to_compress:
            for block in messages[idx]["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    old_content = block.get("content", "")
                    if old_content.startswith("[Earlier result:"):
                        continue  # Already compressed
                    # Keep just the first line as a summary
                    first_line = old_content.split("\n")[0][:150] if old_content else ""
                    block["content"] = f"[Earlier result: {first_line}]"

    def _rate_limit_delay(self, last_request_time: float, last_input_tokens: int) -> None:
        """Sleep if needed to stay under input token rate limits.

        Haiku has a 50K tokens/minute limit. If we just sent a large request,
        we need to wait before sending another to avoid 429 errors.
        """
        if last_request_time == 0.0 or last_input_tokens == 0:
            return

        # Conservative: assume 50K tokens/min for Haiku, higher for other models
        rate_limit = 50_000  # tokens per minute (lowest tier)
        if self.model in (ModelID.SONNET_4_5,):
            rate_limit = 80_000
        elif self.model in (ModelID.OPUS_4_6,):
            rate_limit = 100_000

        elapsed = time.monotonic() - last_request_time
        # How long should we wait to not exceed rate limit?
        # If we used N tokens and limit is L/min, we need N/L minutes of budget
        required_seconds = (last_input_tokens / rate_limit) * 60.0
        wait = required_seconds - elapsed
        if wait > 0:
            time.sleep(wait)


def _truncate_tool_result(text: str, max_chars: int) -> str:
    """Truncate a tool result to max_chars, preserving the beginning."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text)} total chars]"


def _estimate_message_tokens(messages: list[dict]) -> int:
    """Rough token estimate from message content (~4 chars per token)."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for v in block.values():
                        if isinstance(v, str):
                            total_chars += len(v)
    return total_chars // 4
