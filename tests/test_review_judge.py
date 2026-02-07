"""Tests for cbench.review.judge — score parsing and normalization."""

from __future__ import annotations

import pytest

from cbench.review.judge import DIMENSION_WEIGHTS, JudgeScorer


class FakeClient:
    """Stub to avoid importing anthropic for parse-only tests."""

    pass


@pytest.fixture
def scorer():
    return JudgeScorer(FakeClient())


class TestParseScores:
    def test_parse_valid_json(self, scorer):
        text = '{"completeness": 4, "accuracy": 5, "depth": 3, "actionability": 4, "prioritization": 3, "reasoning": "Good review."}'
        scores = scorer._parse_scores(text)
        assert scores.raw_scores["completeness"] == 4
        assert scores.raw_scores["accuracy"] == 5
        assert scores.accuracy == 1.0  # (5-1)/4
        assert scores.completeness == 0.75  # (4-1)/4
        assert scores.depth == 0.5  # (3-1)/4
        assert scores.judge_reasoning == "Good review."

    def test_parse_json_in_code_fence(self, scorer):
        text = 'Here is my assessment:\n```json\n{"completeness": 3, "accuracy": 3, "depth": 3, "actionability": 3, "prioritization": 3, "reasoning": "Average."}\n```'
        scores = scorer._parse_scores(text)
        assert scores.raw_scores["completeness"] == 3
        assert scores.composite_score == 0.5  # All 3s -> (3-1)/4 = 0.5

    def test_parse_json_in_bare_fence(self, scorer):
        text = '```\n{"completeness": 5, "accuracy": 5, "depth": 5, "actionability": 5, "prioritization": 5, "reasoning": "Perfect."}\n```'
        scores = scorer._parse_scores(text)
        assert scores.composite_score == 1.0

    def test_parse_invalid_json(self, scorer):
        scores = scorer._parse_scores("This is not JSON at all.")
        assert scores.composite_score == 0.0
        assert "Failed to parse" in scores.judge_reasoning

    def test_parse_missing_dimension(self, scorer):
        text = '{"completeness": 4, "accuracy": 5, "depth": 3}'
        scores = scorer._parse_scores(text)
        # Should fail because actionability and prioritization are missing
        assert scores.composite_score == 0.0
        assert "Failed to parse" in scores.judge_reasoning

    def test_parse_out_of_range(self, scorer):
        text = '{"completeness": 6, "accuracy": 5, "depth": 3, "actionability": 4, "prioritization": 3, "reasoning": "Bad."}'
        scores = scorer._parse_scores(text)
        assert scores.composite_score == 0.0
        assert "Failed to parse" in scores.judge_reasoning

    def test_parse_zero_score(self, scorer):
        text = '{"completeness": 0, "accuracy": 5, "depth": 3, "actionability": 4, "prioritization": 3, "reasoning": "Bad."}'
        scores = scorer._parse_scores(text)
        assert scores.composite_score == 0.0
        assert "Failed to parse" in scores.judge_reasoning


class TestCompositeScore:
    def test_all_fives_gives_one(self, scorer):
        text = '{"completeness": 5, "accuracy": 5, "depth": 5, "actionability": 5, "prioritization": 5, "reasoning": "Perfect."}'
        scores = scorer._parse_scores(text)
        assert scores.composite_score == pytest.approx(1.0)

    def test_all_ones_gives_zero(self, scorer):
        text = '{"completeness": 1, "accuracy": 1, "depth": 1, "actionability": 1, "prioritization": 1, "reasoning": "Bad."}'
        scores = scorer._parse_scores(text)
        assert scores.composite_score == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_weighted_average(self, scorer):
        text = '{"completeness": 3, "accuracy": 5, "depth": 1, "actionability": 3, "prioritization": 3, "reasoning": "Mixed."}'
        scores = scorer._parse_scores(text)
        # Manual: compl=0.5*0.20, accur=1.0*0.25, depth=0.0*0.25, action=0.5*0.15, prior=0.5*0.15
        expected = 0.5 * 0.20 + 1.0 * 0.25 + 0.0 * 0.25 + 0.5 * 0.15 + 0.5 * 0.15
        assert scores.composite_score == pytest.approx(expected)


class TestExtractJsonBlocks:
    def test_extracts_from_code_fence(self, scorer):
        text = 'Some text\n```json\n{"key": 1}\n```\nMore text'
        blocks = scorer._extract_json_blocks(text)
        assert any('{"key": 1}' in b for b in blocks)

    def test_extracts_bare_braces(self, scorer):
        text = 'Result: {"a": 1}'
        blocks = scorer._extract_json_blocks(text)
        assert '{"a": 1}' in blocks

    def test_no_json(self, scorer):
        blocks = scorer._extract_json_blocks("No JSON here.")
        assert blocks == []
