"""Tests for the review CLI subcommand (argument parsing, dry-run, variant filtering)."""

from __future__ import annotations

import pytest

from cbench.cli import main


class TestReviewCLIParsing:
    def test_review_dry_run(self, capsys, tmp_path):
        """--dry-run should show cost estimate and exit without API calls."""
        (tmp_path / "README.md").write_text("# Test")
        main(["review", str(tmp_path), "--dry-run"])
        captured = capsys.readouterr()
        assert "Dry run complete" in captured.out

    def test_review_unknown_variant(self, tmp_path):
        """Unknown variant should produce an error and exit."""
        (tmp_path / "README.md").write_text("# Test")
        with pytest.raises(SystemExit):
            main(["review", str(tmp_path), "--dry-run", "--variants", "nonexistent_variant"])

    def test_review_valid_variant_filter(self, capsys, tmp_path):
        """Valid --variants should filter to just that variant."""
        (tmp_path / "README.md").write_text("# Test")
        main(["review", str(tmp_path), "--dry-run", "--variants", "haiku_no_thinking"])
        captured = capsys.readouterr()
        # Rich may truncate; check for prefix and that only 1 variant row appears
        assert "haiku_no" in captured.out
        # Should NOT have other variants
        assert "opus_adap" not in captured.out

    def test_review_nonexistent_path(self, tmp_path):
        """Non-existent path should produce an error."""
        with pytest.raises(SystemExit):
            main(["review", str(tmp_path / "no_such_dir"), "--dry-run"])

    def test_review_help(self):
        """--help should show review options."""
        with pytest.raises(SystemExit) as exc_info:
            main(["review", "--help"])
        assert exc_info.value.code == 0
