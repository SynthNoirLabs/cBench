from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ScoringMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    CODE_EXECUTION = "code_execution"
    LLM_JUDGE = "llm_judge"


@dataclass
class TaskDefinition:
    name: str
    category: str
    prompt: str
    expected_answer: str
    scoring_method: ScoringMethod
    difficulty: str = "medium"  # low / medium / hard
    test_cases: list[dict] | None = None  # For code_execution tasks
    judge_rubric: str = ""  # For LLM_JUDGE tasks


class BenchmarkTask(ABC):
    @abstractmethod
    def get_tasks(self) -> list[TaskDefinition]: ...
