"""Tests for cbench.review.context — repo tree and key file gathering."""

from __future__ import annotations

from pathlib import Path

import pytest

from cbench.review.context import build_repo_tree, gather_key_files


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo structure for context tests."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "src" / "__init__.py").write_text("# init\n")
    (tmp_path / "src" / "utils").mkdir()
    (tmp_path / "src" / "utils" / "helpers.py").write_text("def helper(): pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass\n")
    (tmp_path / "README.md").write_text("# My Project\nThis is a test project.\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir()
    return tmp_path


class TestBuildRepoTree:
    def test_contains_root(self, tmp_repo: Path) -> None:
        tree = build_repo_tree(tmp_repo)
        assert tmp_repo.name + "/" in tree

    def test_contains_src_directory(self, tmp_repo: Path) -> None:
        tree = build_repo_tree(tmp_repo)
        assert "src/" in tree

    def test_contains_files(self, tmp_repo: Path) -> None:
        tree = build_repo_tree(tmp_repo)
        assert "main.py" in tree
        assert "README.md" in tree

    def test_skips_hidden_dirs(self, tmp_repo: Path) -> None:
        tree = build_repo_tree(tmp_repo)
        assert ".git" not in tree

    def test_skips_node_modules(self, tmp_repo: Path) -> None:
        tree = build_repo_tree(tmp_repo)
        # Check that node_modules/ doesn't appear as a tree entry
        # (the root dir name may contain "node_modules" as a pytest artifact)
        tree_lines = tree.splitlines()
        assert not any("node_modules/" in line for line in tree_lines[1:])

    def test_respects_max_depth(self, tmp_repo: Path) -> None:
        tree = build_repo_tree(tmp_repo, max_depth=1)
        # Should show top-level dirs but not their contents
        assert "src/" in tree
        assert "main.py" not in tree

    def test_tree_connectors(self, tmp_repo: Path) -> None:
        tree = build_repo_tree(tmp_repo)
        # Should use tree-drawing characters
        assert any(c in tree for c in ["├── ", "└── "])


class TestGatherKeyFiles:
    def test_finds_readme(self, tmp_repo: Path) -> None:
        content = gather_key_files(tmp_repo)
        assert "README.md" in content
        assert "My Project" in content

    def test_finds_pyproject(self, tmp_repo: Path) -> None:
        content = gather_key_files(tmp_repo)
        assert "pyproject.toml" in content

    def test_respects_char_budget(self, tmp_repo: Path) -> None:
        content = gather_key_files(tmp_repo, max_total_chars=50)
        assert len(content) < 200  # Some overhead from headers

    def test_no_key_files(self, tmp_path: Path) -> None:
        (tmp_path / "random.dat").write_text("data")
        content = gather_key_files(tmp_path)
        assert content == "(no key files found)"

    def test_truncates_large_files(self, tmp_repo: Path) -> None:
        # Write a large README
        (tmp_repo / "README.md").write_text("x" * 100_000)
        content = gather_key_files(tmp_repo, max_total_chars=1000)
        assert "truncated" in content
