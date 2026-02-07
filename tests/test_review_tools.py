"""Tests for cbench.review.tools — sandbox, tool execution, dispatch."""

from __future__ import annotations

import pytest

from cbench.review.tools import (
    RepoSandbox,
    dispatch_tool,
    execute_list_directory,
    execute_read_file,
    execute_search_code,
)


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal temp repo structure."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    print('hello')\n")
    (tmp_path / "README.md").write_text("# Test Repo\nA test repository.\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input.txt").write_text("line1\nline2\nline3\nline4\nline5\n")
    return tmp_path


@pytest.fixture
def sandbox(tmp_repo):
    return RepoSandbox(tmp_repo)


# --- RepoSandbox ---


class TestRepoSandbox:
    def test_resolve_valid_path(self, sandbox, tmp_repo):
        resolved = sandbox.resolve("src/main.py")
        assert resolved == tmp_repo / "src" / "main.py"

    def test_resolve_root(self, sandbox, tmp_repo):
        resolved = sandbox.resolve(".")
        assert resolved == tmp_repo

    def test_resolve_blocks_traversal(self, sandbox):
        with pytest.raises(PermissionError, match="escapes sandbox"):
            sandbox.resolve("../../etc/passwd")

    def test_resolve_blocks_absolute_path_traversal(self, sandbox):
        # Absolute paths get joined but resolve may escape
        with pytest.raises(PermissionError, match="escapes sandbox"):
            sandbox.resolve("../../../tmp")

    def test_resolve_blocks_symlink_escape(self, sandbox, tmp_repo):
        # Create a symlink pointing outside the sandbox
        outside_dir = tmp_repo.parent / "outside"
        outside_dir.mkdir(exist_ok=True)
        (outside_dir / "secret.txt").write_text("secret")
        link = tmp_repo / "evil_link"
        link.symlink_to(outside_dir)
        with pytest.raises(PermissionError, match="escapes sandbox"):
            sandbox.resolve("evil_link")

    def test_resolve_allows_internal_symlink(self, sandbox, tmp_repo):
        link = tmp_repo / "src_link"
        link.symlink_to(tmp_repo / "src")
        resolved = sandbox.resolve("src_link/main.py")
        assert resolved.is_file()

    def test_init_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            RepoSandbox(tmp_path / "nonexistent")

    def test_prefix_collision_blocked(self, tmp_path):
        """Regression: /tmp/sandbox vs /tmp/sandbox-evil must be blocked."""
        real_dir = tmp_path / "sandbox"
        real_dir.mkdir()
        evil_dir = tmp_path / "sandbox-evil"
        evil_dir.mkdir()
        (evil_dir / "secret.txt").write_text("secret")

        sb = RepoSandbox(real_dir)
        with pytest.raises(PermissionError):
            sb.resolve("../sandbox-evil/secret.txt")


# --- execute_list_directory ---


class TestListDirectory:
    def test_list_root(self, sandbox):
        result = execute_list_directory(sandbox, ".")
        assert "README.md" in result
        assert "src/" in result

    def test_skips_hidden_dirs(self, sandbox):
        result = execute_list_directory(sandbox, ".")
        assert ".git" not in result

    def test_list_subdirectory(self, sandbox):
        result = execute_list_directory(sandbox, "src")
        assert "main.py" in result

    def test_list_nonexistent(self, sandbox):
        result = execute_list_directory(sandbox, "nonexistent")
        assert "Error" in result

    def test_list_file_not_dir(self, sandbox):
        result = execute_list_directory(sandbox, "README.md")
        assert "Error" in result


# --- execute_read_file ---


class TestReadFile:
    def test_read_whole_file(self, sandbox):
        result = execute_read_file(sandbox, "src/main.py")
        assert "def hello():" in result
        assert "File: src/main.py" in result

    def test_read_with_offset(self, sandbox):
        result = execute_read_file(sandbox, "data/input.txt", offset=3, limit=2)
        assert "line3" in result
        assert "line1" not in result

    def test_read_nonexistent(self, sandbox):
        result = execute_read_file(sandbox, "no_such_file.py")
        assert "Error" in result

    def test_read_directory(self, sandbox):
        result = execute_read_file(sandbox, "src")
        assert "Error" in result

    def test_line_numbers(self, sandbox):
        result = execute_read_file(sandbox, "data/input.txt")
        assert "    1 | line1" in result
        assert "    5 | line5" in result


# --- execute_search_code ---


class TestSearchCode:
    def test_search_finds_match(self, sandbox):
        result = execute_search_code(sandbox, "def hello")
        assert "src/main.py" in result
        assert "def hello" in result

    def test_search_no_match(self, sandbox):
        result = execute_search_code(sandbox, "zzz_no_match_zzz")
        assert "No matches found" in result

    def test_search_with_file_glob(self, sandbox):
        result = execute_search_code(sandbox, "line", file_glob="*.txt")
        assert "input.txt" in result

    def test_search_in_subdirectory(self, sandbox):
        result = execute_search_code(sandbox, "print", path="src")
        assert "main.py" in result

    def test_search_invalid_regex(self, sandbox):
        result = execute_search_code(sandbox, "[invalid")
        assert "Error: invalid regex" in result

    def test_search_pattern_too_long(self, sandbox):
        result = execute_search_code(sandbox, "a" * 201)
        assert "Error: pattern too long" in result

    def test_search_respects_max_results(self, sandbox, tmp_repo):
        # Create a file with many matching lines
        many_lines = "\n".join(f"match_{i}" for i in range(100))
        (tmp_repo / "big.txt").write_text(many_lines)
        result = execute_search_code(sandbox, "match_", max_results=5)
        assert "truncated at 5" in result


# --- dispatch_tool ---


class TestDispatchTool:
    def test_dispatch_list_directory(self, sandbox):
        result = dispatch_tool(sandbox, "list_directory", {"path": "."})
        assert "src/" in result

    def test_dispatch_read_file(self, sandbox):
        result = dispatch_tool(sandbox, "read_file", {"path": "README.md"})
        assert "Test Repo" in result

    def test_dispatch_search_code(self, sandbox):
        result = dispatch_tool(sandbox, "search_code", {"pattern": "hello"})
        assert "main.py" in result

    def test_dispatch_unknown_tool(self, sandbox):
        result = dispatch_tool(sandbox, "unknown_tool", {})
        assert "Error: unknown tool" in result

    def test_dispatch_traversal_blocked(self, sandbox):
        result = dispatch_tool(sandbox, "read_file", {"path": "../../etc/passwd"})
        assert "Error" in result
