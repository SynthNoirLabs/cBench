"""Tests for cbench.review.benchmark — variant definitions."""

from __future__ import annotations

from cbench.config import ModelID
from cbench.review.benchmark import ReviewBenchmark


class TestReviewBenchmark:
    def test_default_variants(self):
        b = ReviewBenchmark()
        variants = b.get_variants()
        assert len(variants) == 7
        names = [v["name"] for v in variants]
        # Raw API variants
        assert "opus_adaptive" in names
        assert "opus_effort_high" in names
        assert "sonnet_adaptive" in names
        assert "haiku_no_thinking" in names
        # SDK variants
        assert "sdk_opus" in names
        assert "sdk_sonnet" in names
        assert "sdk_haiku" in names

    def test_custom_variants(self):
        custom = [
            {
                "name": "test_variant",
                "model": ModelID.HAIKU_4_5,
                "thinking": None,
                "effort": None,
                "max_tokens": 4096,
            }
        ]
        b = ReviewBenchmark(variants=custom)
        assert len(b.get_variants()) == 1
        assert b.get_variants()[0]["name"] == "test_variant"

    def test_all_variants_have_required_keys(self):
        b = ReviewBenchmark()
        for v in b.get_variants():
            assert "name" in v
            assert "model" in v
            assert isinstance(v["model"], ModelID)

    def test_name_and_description(self):
        b = ReviewBenchmark()
        assert b.name == "repo_review"
        assert len(b.description) > 0
