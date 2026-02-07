from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ScoringMethod(StrEnum):
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    CODE_EXECUTION = "code_execution"
    LLM_JUDGE = "llm_judge"
    TOOL_MATCH = "tool_match"


@dataclass
class TaskDefinition:
    name: str
    category: str
    prompt: str
    expected_answer: str
    scoring_method: ScoringMethod
    difficulty: str = "medium"  # low / medium / hard
    test_cases: list[dict[str, Any]] | None = None  # For code_execution tasks
    judge_rubric: str = ""  # For LLM_JUDGE tasks


class BenchmarkTask(ABC):
    @abstractmethod
    def get_tasks(self) -> list[TaskDefinition]: ...
