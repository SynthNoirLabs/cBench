from __future__ import annotations

from cbench.tasks.api_design import ApiDesignTasks
from cbench.tasks.base import BenchmarkTask, TaskDefinition
from cbench.tasks.base import ScoringMethod as ScoringMethod  # re-export
from cbench.tasks.code_generation import CodeGenerationTasks
from cbench.tasks.complex_analysis import ComplexAnalysisTasks
from cbench.tasks.math_reasoning import MathReasoningTasks

TASK_REGISTRY: dict[str, type[BenchmarkTask]] = {
    "math_reasoning": MathReasoningTasks,
    "code_generation": CodeGenerationTasks,
    "complex_analysis": ComplexAnalysisTasks,
    "api_design": ApiDesignTasks,
}


def get_all_tasks() -> list[TaskDefinition]:
    tasks = []
    for task_class in TASK_REGISTRY.values():
        tasks.extend(task_class().get_tasks())
    return tasks
