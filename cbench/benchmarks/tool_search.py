"""Tool Search benchmark — tests the Tool Search Tool beta feature.

Beta header: tool-search-2025-04-15
When tools are marked with defer_loading=True, Claude discovers them on demand
instead of loading all tool definitions upfront, saving ~85% of tool definition tokens.
"""

from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import BETA_TOOL_SEARCH, ModelID
from cbench.tasks.base import ScoringMethod, TaskDefinition

# 30 dummy tools + 5 real target tools = 35 total
# The dummy tools create noise; the model must find the right one via tool search

_DUMMY_TOOLS = [
    {
        "name": f"legacy_tool_{i}",
        "description": f"Legacy internal tool #{i} for deprecated system operations.",
        "input_schema": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
    }
    for i in range(30)
]

_TARGET_TOOLS = [
    {
        "name": "weather_lookup",
        "description": "Get current weather conditions for a city. Returns temperature, humidity, and conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
    {
        "name": "currency_convert",
        "description": "Convert an amount from one currency to another using live exchange rates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "ISO 4217 code"},
                "to_currency": {"type": "string", "description": "ISO 4217 code"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate_bmi",
        "description": "Calculate Body Mass Index from height and weight measurements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "height_cm": {"type": "number"},
                "weight_kg": {"type": "number"},
            },
            "required": ["height_cm", "weight_kg"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text between languages using machine translation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_lang": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
    {
        "name": "search_database",
        "description": "Query a structured database with SQL-like filters and return matching records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "filters": {"type": "object"},
                "limit": {"type": "integer"},
            },
            "required": ["table"],
        },
    },
]


def _all_tools(defer_loading: bool) -> list[dict[str, Any]]:
    """Return all tools, optionally marking them for deferred loading."""
    tools: list[dict[str, Any]] = _DUMMY_TOOLS + _TARGET_TOOLS
    if defer_loading:
        return [{**t, "defer_loading": True} for t in tools]
    return [dict(t) for t in tools]


class ToolSearchBenchmark(Benchmark):
    name = "tool_search"
    description = "Tool Search Tool — deferred tool loading vs full tool context"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "with_tool_search",
                "defer_loading": True,
                "model": ModelID.OPUS_4_6,
                "betas": [BETA_TOOL_SEARCH],
            },
            {
                "name": "without_tool_search",
                "defer_loading": False,
                "model": ModelID.OPUS_4_6,
            },
        ]

    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="select_weather_tool",
                category="tool_search",
                prompt="What is the current weather in Tokyo? Use the appropriate tool.",
                expected_answer="weather_lookup",
                scoring_method=ScoringMethod.TOOL_MATCH,
                difficulty="medium",
            ),
            TaskDefinition(
                name="select_currency_tool",
                category="tool_search",
                prompt="Convert 500 USD to EUR. Use the appropriate tool.",
                expected_answer="currency_convert",
                scoring_method=ScoringMethod.TOOL_MATCH,
                difficulty="medium",
            ),
            TaskDefinition(
                name="select_translate_tool",
                category="tool_search",
                prompt="Translate 'Hello, how are you?' to Japanese. Use the appropriate tool.",
                expected_answer="translate_text",
                scoring_method=ScoringMethod.TOOL_MATCH,
                difficulty="medium",
            ),
            TaskDefinition(
                name="select_database_tool",
                category="tool_search",
                prompt="Find all users in the 'customers' table who signed up in 2025. Use the appropriate tool.",
                expected_answer="search_database",
                scoring_method=ScoringMethod.TOOL_MATCH,
                difficulty="medium",
            ),
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "tools": _all_tools(variant["defer_loading"]),
            "tool_choice": {"type": "any"},
        }
        if "betas" in variant:
            kwargs["betas"] = variant["betas"]
        return kwargs
