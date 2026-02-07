"""Repair benchmark: model × agent_type variants over fixture repos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cbench.config import ModelID


FIXTURES_DIR = Path(__file__).parent / "fixtures"

ALL_FIXTURE_NAMES = ["calculator_bug", "api_handler_bug", "data_pipeline_bug"]

FIXTURE_METADATA = {
    "calculator_bug": {
        "difficulty": "easy",
        "description": "Off-by-one bug in factorial function",
        "source_file": "calculator.py",
        "test_file": "test_calculator.py",
    },
    "api_handler_bug": {
        "difficulty": "medium",
        "description": "Missing error handling in REST API handler",
        "source_file": "handler.py",
        "test_file": "test_handler.py",
    },
    "data_pipeline_bug": {
        "difficulty": "hard",
        "description": "Multiple bugs in data pipeline (CSV parsing, filtering, dedup)",
        "source_file": "pipeline.py",
        "test_file": "test_pipeline.py",
    },
}


@dataclass
class FixtureInfo:
    name: str
    path: Path
    difficulty: str
    description: str
    source_file: str
    test_file: str


class RepairBenchmark:
    name = "code_repair"
    description = "Agentic code repair: fix bugs to make failing tests pass"

    def __init__(
        self,
        fixture_names: list[str] | None = None,
        variants: list[dict] | None = None,
    ) -> None:
        self._fixture_names = fixture_names or ALL_FIXTURE_NAMES
        self._variants = variants

    def get_fixtures(self) -> list[FixtureInfo]:
        fixtures = []
        for name in self._fixture_names:
            meta = FIXTURE_METADATA.get(name, {})
            path = FIXTURES_DIR / name
            if path.is_dir():
                fixtures.append(
                    FixtureInfo(
                        name=name,
                        path=path,
                        difficulty=meta.get("difficulty", "unknown"),
                        description=meta.get("description", ""),
                        source_file=meta.get("source_file", ""),
                        test_file=meta.get("test_file", ""),
                    )
                )
        return fixtures

    def get_variants(self) -> list[dict]:
        if self._variants:
            return self._variants
        return [
            {
                "name": "opus_agent",
                "model": ModelID.OPUS_4_6,
                "thinking": {"type": "adaptive"},
            },
            {
                "name": "sonnet_agent",
                "model": ModelID.SONNET_4_5,
                "thinking": {"type": "enabled", "budget_tokens": 8192},
            },
            {
                "name": "haiku_agent",
                "model": ModelID.HAIKU_4_5,
            },
        ]
