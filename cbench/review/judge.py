"""LLM-as-Judge: Opus 4.6 scores a code review on 5 dimensions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

from cbench.config import PRICING, ModelID

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating the quality of a code review. You will be given:
1. A code review produced by an AI model
2. The repository's directory tree
3. Key files from the repository (README, config, entry points)

Score the review on these 5 dimensions using a 1-5 scale:

## Scoring Rubric

### Completeness (1-5)
1 = Covers <20% of the codebase, misses entire modules
2 = Covers ~30-40%, misses several important areas
3 = Covers ~50-60%, addresses main components
4 = Covers ~70-80%, only minor areas missed
5 = Comprehensive coverage of all significant code areas

### Accuracy (1-5)
1 = Multiple incorrect claims, misunderstands the codebase
2 = Several inaccuracies, some mischaracterizations
3 = Mostly accurate with a few minor errors
4 = Highly accurate, at most 1-2 trivial errors
5 = Fully accurate, all claims verifiable against the code

### Depth (1-5)
1 = Surface-level observations only (e.g., "code looks clean")
2 = Some specific observations but mostly generic
3 = Mix of specific and generic, identifies real issues
4 = Deep analysis with specific file/line references and root-cause reasoning
5 = Expert-level analysis: identifies non-obvious issues, architectural implications, interaction effects

### Actionability (1-5)
1 = Vague suggestions ("improve error handling")
2 = Some specific suggestions but mostly vague
3 = Mix of specific and vague recommendations
4 = Most recommendations are concrete with clear steps
5 = All recommendations are specific, prioritized, with implementation guidance

### Prioritization (1-5)
1 = No distinction between critical and minor issues
2 = Some attempt at ordering but inconsistent
3 = Issues loosely grouped by importance
4 = Clear severity labels, critical issues highlighted first
5 = Expert triage: issues ranked by impact, effort, and risk with clear rationale

## Output Format
Respond with ONLY a JSON object (no markdown fences):
{
  "completeness": <1-5>,
  "accuracy": <1-5>,
  "depth": <1-5>,
  "actionability": <1-5>,
  "prioritization": <1-5>,
  "reasoning": "<2-3 sentences explaining your scoring>"
}\
"""

DIMENSION_WEIGHTS = {
    "accuracy": 0.25,
    "depth": 0.25,
    "completeness": 0.20,
    "actionability": 0.15,
    "prioritization": 0.15,
}


@dataclass
class JudgeScores:
    completeness: float = 0.0
    accuracy: float = 0.0
    depth: float = 0.0
    actionability: float = 0.0
    prioritization: float = 0.0
    composite_score: float = 0.0
    raw_scores: dict[str, int] = field(default_factory=dict)
    judge_reasoning: str = ""
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cost_usd: float = 0.0


class JudgeScorer:
    """Scores a code review using Opus 4.6 as judge."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client
        self.model = ModelID.OPUS_4_6

    def score(
        self,
        review_text: str,
        repo_tree: str,
        key_files_content: str,
    ) -> JudgeScores:
        user_message = (
            f"## Repository Structure\n```\n{repo_tree}\n```\n\n"
            f"## Key Files\n{key_files_content}\n\n"
            f"## Code Review to Judge\n{review_text}"
        )

        try:
            response = self.client.messages.create(
                model=self.model.value,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=2048,
            )
        except anthropic.APIStatusError as e:
            return JudgeScores(judge_reasoning=f"Judge API error: {e}")

        # Parse response
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        usage = response.usage
        pricing = PRICING[self.model]
        judge_cost = (usage.input_tokens / 1_000_000) * pricing["input_per_mtok"] + (
            usage.output_tokens / 1_000_000
        ) * pricing["output_per_mtok"]

        # Extract JSON from response
        scores = self._parse_scores(response_text)
        scores.judge_input_tokens = usage.input_tokens
        scores.judge_output_tokens = usage.output_tokens
        scores.judge_cost_usd = judge_cost
        return scores

    def _parse_scores(self, text: str) -> JudgeScores:
        """Parse JSON scores from judge response."""
        # Try direct JSON parse first, then try extracting from markdown
        for candidate in [text.strip(), *self._extract_json_blocks(text)]:
            try:
                data = json.loads(candidate)
                return self._build_scores(data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        return JudgeScores(judge_reasoning=f"Failed to parse judge response: {text[:200]}")

    def _extract_json_blocks(self, text: str) -> list[str]:
        """Extract potential JSON blocks from text."""
        blocks = []
        # Try markdown code fences
        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
            blocks.append(match.group(1).strip())
        # Try bare braces
        for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
            blocks.append(match.group(0))
        return blocks

    def _build_scores(self, data: dict) -> JudgeScores:
        """Build JudgeScores from parsed JSON data."""
        raw = {}
        normalized = {}
        for dim in DIMENSION_WEIGHTS:
            raw_val = int(data[dim])
            if not 1 <= raw_val <= 5:
                raise ValueError(f"Score out of range for {dim}: {raw_val}")
            raw[dim] = raw_val
            normalized[dim] = (raw_val - 1) / 4.0

        composite = sum(normalized[dim] * weight for dim, weight in DIMENSION_WEIGHTS.items())

        return JudgeScores(
            completeness=normalized["completeness"],
            accuracy=normalized["accuracy"],
            depth=normalized["depth"],
            actionability=normalized["actionability"],
            prioritization=normalized["prioritization"],
            composite_score=composite,
            raw_scores=raw,
            judge_reasoning=data.get("reasoning", ""),
        )
