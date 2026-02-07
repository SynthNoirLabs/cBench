"""Agent SDK-based review loop: uses claude-agent-sdk to explore a repo and produce a review."""

from __future__ import annotations

import asyncio
from pathlib import Path

from cbench.config import ModelID
from cbench.review.agent import AgentMetrics, AgentResult

# Review instructions appended to the Claude Code system prompt
REVIEW_INSTRUCTIONS = """\
You are reviewing a code repository. Explore it systematically:
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


class SDKAgentLoop:
    """Review agent using the Claude Code Agent SDK (claude-agent-sdk).

    Same interface as AgentLoop: call .run() to get an AgentResult.
    The SDK manages its own tools (Read, Glob, Grep), context compaction,
    and session management.
    """

    def __init__(
        self,
        model: ModelID,
        repo_path: Path,
        max_turns: int = 30,
        max_tokens: int = 16384,
        thinking: dict | None = None,
        effort: str | None = None,
    ) -> None:
        self.model = model
        self.repo_path = Path(repo_path).resolve()
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.effort = effort

    def run(self) -> AgentResult:
        """Synchronous entry point — bridges to the async SDK via asyncio.run()."""
        try:
            return asyncio.run(self._run_async())
        except Exception as e:
            return AgentResult(
                error=f"{type(e).__name__}: {e}",
                metrics=AgentMetrics(),
            )

    async def _run_async(self) -> AgentResult:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
            from claude_agent_sdk.types import AssistantMessage, ResultMessage
        except ImportError as e:
            return AgentResult(
                error=f"claude-agent-sdk not installed: {e}",
                metrics=AgentMetrics(),
            )

        # Track tool calls via allowed_tools filtering (we count from messages)
        tool_counts: dict[str, int] = {}

        options = ClaudeAgentOptions(
            model=self.model.value,
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="bypassPermissions",
            cwd=str(self.repo_path),
            max_turns=self.max_turns,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": REVIEW_INSTRUCTIONS,
            },
        )

        review_text = ""
        result_message = None

        try:
            async for message in query(
                prompt="Please review this repository thoroughly.",
                options=options,
            ):
                if isinstance(message, ResultMessage):
                    result_message = message
                    review_text = message.result or ""
                elif isinstance(message, AssistantMessage):
                    # Count tool uses from assistant messages
                    from claude_agent_sdk.types import ToolUseBlock

                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            tool_counts[block.name] = tool_counts.get(block.name, 0) + 1
        except Exception as e:
            error_type = type(e).__name__
            return AgentResult(
                error=f"{error_type}: {e}",
                metrics=AgentMetrics(),
            )

        total_tool_calls = sum(tool_counts.values())

        # Build metrics from result message
        metrics = AgentMetrics(
            total_turns=result_message.num_turns if result_message else 0,
            tool_calls=total_tool_calls,
            tool_calls_by_name=tool_counts,
            files_read=tool_counts.get("Read", 0),
            directories_listed=tool_counts.get("Glob", 0),
            searches_performed=tool_counts.get("Grep", 0),
            total_cost_usd=result_message.total_cost_usd or 0.0 if result_message else 0.0,
            wall_clock_seconds=(result_message.duration_ms / 1000.0) if result_message else 0.0,
        )

        # Extract token usage if available
        if result_message and result_message.usage:
            usage = result_message.usage
            metrics.total_input_tokens = usage.get("input_tokens", 0)
            metrics.total_output_tokens = usage.get("output_tokens", 0)

        return AgentResult(
            review_text=review_text,
            metrics=metrics,
        )
