"""Programmatic Tool Calling benchmark — tests the advanced tool use beta.

Beta header: advanced-tool-use-2025-11-20
Tools with allowed_callers=["code_execution_20250825"] let Claude write Python code
that orchestrates tool calls in a sandbox. Only print() output enters the context,
reducing token usage by ~37% (up to 98.7% combined with tool search).
"""

from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import BETA_PROGRAMMATIC_TOOLS, ModelID
from cbench.tasks.base import ScoringMethod, TaskDefinition

# Tools that support programmatic calling (have allowed_callers)
_PROGRAMMATIC_TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
            },
            "required": ["ticker"],
        },
        "allowed_callers": ["code_execution_20250825"],
    },
    {
        "name": "get_exchange_rate",
        "description": "Get exchange rate between two currencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["from_currency", "to_currency"],
        },
        "allowed_callers": ["code_execution_20250825"],
    },
    {
        "name": "calculate_tax",
        "description": "Calculate tax amount for a given income and jurisdiction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "income": {"type": "number"},
                "jurisdiction": {"type": "string"},
            },
            "required": ["income", "jurisdiction"],
        },
        "allowed_callers": ["code_execution_20250825"],
    },
    {
        "name": "send_notification",
        "description": "Send a notification message to a user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "message": {"type": "string"},
                "channel": {"type": "string", "enum": ["email", "sms", "push"]},
            },
            "required": ["user_id", "message"],
        },
        "allowed_callers": ["code_execution_20250825"],
    },
]

# Same tools without allowed_callers (baseline)
_STANDARD_TOOLS = [
    {k: v for k, v in tool.items() if k != "allowed_callers"}
    for tool in _PROGRAMMATIC_TOOLS
]


class ProgrammaticToolsBenchmark(Benchmark):
    name = "programmatic_tools"
    description = "Programmatic tool calling — code execution for tool orchestration vs standard"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "with_programmatic",
                "use_programmatic": True,
                "model": ModelID.OPUS_4_6,
                "betas": [BETA_PROGRAMMATIC_TOOLS],
            },
            {
                "name": "without_programmatic",
                "use_programmatic": False,
                "model": ModelID.OPUS_4_6,
            },
        ]

    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="multi_tool_stock_analysis",
                category="programmatic_tools",
                prompt=(
                    "I need to know the current stock prices for AAPL, GOOGL, and MSFT, "
                    "then convert each price from USD to EUR. Use the available tools to "
                    "get the stock prices and exchange rate, then present the results."
                ),
                expected_answer="AAPL",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="hard",
            ),
            TaskDefinition(
                name="conditional_notification",
                category="programmatic_tools",
                prompt=(
                    "Check the stock price of TSLA. If it's mentioned in your response, "
                    "send a push notification to user 'trader_42' with the price. "
                    "Use the available tools."
                ),
                expected_answer="TSLA",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="medium",
            ),
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        tools = _PROGRAMMATIC_TOOLS if variant["use_programmatic"] else _STANDARD_TOOLS
        kwargs: dict[str, Any] = {
            "tools": tools,
        }
        if "betas" in variant:
            kwargs["betas"] = variant["betas"]
        return kwargs
