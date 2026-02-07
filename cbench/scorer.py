from __future__ import annotations

import json
import re
import string
import subprocess

from cbench.tasks.base import ScoringMethod, TaskDefinition


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace, and remove trailing punctuation."""
    text = text.strip().lower()
    text = text.rstrip(string.punctuation)
    return text


def _extract_code(response_text: str) -> str:
    """Extract Python code from markdown code blocks.

    Looks for ```python ... ``` first, then plain ``` ... ```.
    Falls back to the entire response if no block is found.
    """
    # Try ```python blocks first
    match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try plain ``` blocks
    match = re.search(r"```\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response_text.strip()


def _score_exact_match(expected: str, response: str) -> float:
    return 1.0 if _normalize(expected) == _normalize(response) else 0.0


def _score_contains(expected: str, response: str) -> float:
    return 1.0 if _normalize(expected) in _normalize(response) else 0.0


def _score_code_execution(task_def: TaskDefinition, response_text: str) -> float:
    if not task_def.test_cases:
        return 0.0

    code = _extract_code(response_text)
    passed = 0

    for test_case in task_def.test_cases:
        try:
            test_input = test_case["input"]
            expected = test_case["expected"]

            # Build the test invocation based on input type
            if isinstance(test_input, dict):
                # Keyword arguments: merge_sorted(a=[1,3,5], b=[2,4,6])
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in test_input.items())
                # Find the function name from the code
                func_name = _extract_func_name(code)
                script = f"{code}\n\nprint(repr({func_name}({args_str})))"
            elif isinstance(test_input, str):
                # String input
                func_name = _extract_func_name(code)
                script = f"{code}\n\nprint(repr({func_name}({repr(test_input)})))"
            else:
                # List/positional input (original format)
                func_name = _extract_func_name(code)
                script = f"{code}\n\nprint(repr({func_name}({repr(test_input)})))"

            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=10,
            )

            actual = result.stdout.strip()
            if repr(expected) == actual or str(expected) == actual:
                passed += 1
        except Exception:
            continue

    return passed / len(task_def.test_cases)


def _extract_func_name(code: str) -> str:
    """Extract the first function name from Python code."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else "solution"


def _score_llm_judge(task_def: TaskDefinition, response_text: str) -> float:
    """Score using LLM-as-Judge (Opus 4.6).

    Returns a float between 0.0 and 1.0 (composite of rubric dimensions).
    """
    try:
        import anthropic

        from cbench.config import PRICING, ModelID
    except ImportError:
        return 0.0

    judge_model = ModelID.OPUS_4_6

    system_prompt = (
        "You are an expert judge evaluating the quality of an AI model's response. "
        "You will be given:\n"
        "1. The original task prompt\n"
        "2. The model's response\n"
        "3. A scoring rubric\n\n"
        "Score the response according to the rubric. "
        "Respond with ONLY a JSON object (no markdown fences) matching the rubric's format."
    )

    user_message = (
        f"## Task Prompt\n{task_def.prompt}\n\n"
        f"## Model Response\n{response_text}\n\n"
        f"## Scoring Rubric\n{task_def.judge_rubric}"
    )

    try:
        client = anthropic.Anthropic(max_retries=3)
        response = client.messages.create(
            model=judge_model.value,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2048,
        )
    except Exception:
        return 0.0

    # Extract text from response
    judge_text = ""
    for block in response.content:
        if block.type == "text":
            judge_text += block.text

    # Parse JSON scores
    scores = _parse_judge_scores(judge_text)
    if not scores:
        return 0.0

    # Normalize all numeric scores from 1-5 to 0-1, compute average
    normalized = []
    for key, val in scores.items():
        if key == "reasoning":
            continue
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            normalized.append((val - 1) / 4.0)

    return sum(normalized) / len(normalized) if normalized else 0.0


def _parse_judge_scores(text: str) -> dict | None:
    """Parse JSON scores from judge response."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code blocks
    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue

    # Try bare braces
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def score_response(task_def: TaskDefinition, response_text: str) -> float:
    """Score a model response against a task definition.

    Returns a float between 0.0 and 1.0.
    """
    try:
        if task_def.scoring_method == ScoringMethod.EXACT_MATCH:
            return _score_exact_match(task_def.expected_answer, response_text)

        if task_def.scoring_method == ScoringMethod.CONTAINS:
            return _score_contains(task_def.expected_answer, response_text)

        if task_def.scoring_method == ScoringMethod.CODE_EXECUTION:
            return _score_code_execution(task_def, response_text)

        if task_def.scoring_method == ScoringMethod.LLM_JUDGE:
            return _score_llm_judge(task_def, response_text)

        return 0.0
    except Exception:
        return 0.0
