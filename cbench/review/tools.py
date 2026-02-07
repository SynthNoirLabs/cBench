"""Sandboxed repo exploration tools for the review agent."""

from __future__ import annotations

import os
import re
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".output",
    "target",
    ".gradle",
}

TOOL_DEFINITIONS = [
    {
        "name": "list_directory",
        "description": "List files and directories at the given path relative to the repository root. Returns entry names with '/' suffix for directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from repo root. Use '.' for the root directory.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file with line numbers. Returns up to 1000 lines starting from the given offset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file from repo root.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-based). Defaults to 1.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Defaults to 1000.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for a regex pattern in repository files. Returns matching lines with file paths and line numbers, limited to 50 results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in, relative to repo root. Defaults to '.'.",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files, e.g. '*.py'. Defaults to all files.",
                },
            },
            "required": ["pattern"],
        },
    },
]


class RepoSandbox:
    """Security boundary: resolves paths within the repo root, blocks traversal."""

    def __init__(self, repo_root: str | Path) -> None:
        self.root = Path(repo_root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repo root is not a directory: {self.root}")

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path within the sandbox. Raises on escape."""
        raw_path = self.root / relative_path
        # Check for symlinks before resolving — resolve() dereferences them,
        # so is_symlink() would always be False after resolve().
        if raw_path.is_symlink():
            target = raw_path.resolve()
            if not target.is_relative_to(self.root):
                raise PermissionError(f"Symlink escapes sandbox: {relative_path}")
        candidate = raw_path.resolve()
        # Use is_relative_to (not startswith) to avoid prefix-collision attacks
        # e.g. /tmp/sandbox vs /tmp/sandbox-evil
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"Path escapes sandbox: {relative_path}")
        return candidate


def execute_list_directory(sandbox: RepoSandbox, path: str) -> str:
    resolved = sandbox.resolve(path)
    if not resolved.is_dir():
        return f"Error: not a directory: {path}"
    entries = []
    try:
        for entry in sorted(resolved.iterdir()):
            name = entry.name
            if name.startswith(".") or name in SKIP_DIRS:
                continue
            suffix = "/" if entry.is_dir() else ""
            entries.append(f"{name}{suffix}")
    except PermissionError:
        return f"Error: permission denied: {path}"
    if not entries:
        return "(empty directory)"
    return "\n".join(entries)


def execute_read_file(sandbox: RepoSandbox, path: str, offset: int = 1, limit: int = 1000) -> str:
    resolved = sandbox.resolve(path)
    if not resolved.is_file():
        return f"Error: not a file: {path}"
    try:
        text = resolved.read_text(errors="replace")
    except PermissionError:
        return f"Error: permission denied: {path}"
    lines = text.splitlines()
    start = max(0, offset - 1)
    end = start + limit
    selected = lines[start:end]
    numbered = [f"{start + i + 1:>5} | {line}" for i, line in enumerate(selected)]
    header = f"File: {path} (lines {start + 1}-{min(end, len(lines))} of {len(lines)})"
    return header + "\n" + "\n".join(numbered)


def execute_search_code(
    sandbox: RepoSandbox,
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    max_results: int = 50,
) -> str:
    resolved = sandbox.resolve(path)
    if not resolved.is_dir():
        return f"Error: not a directory: {path}"
    if len(pattern) > 200:
        return "Error: pattern too long (max 200 chars)"
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"

    results: list[str] = []
    for root_dir, dirs, files in os.walk(resolved):
        # Prune skip dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        rel_root = Path(root_dir).relative_to(sandbox.root)
        for fname in files:
            if file_glob and not Path(fname).match(file_glob):
                continue
            fpath = Path(root_dir) / fname
            try:
                content = fpath.read_text(errors="replace")
            except (PermissionError, OSError):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{rel_root / fname}:{i}: {line.rstrip()}")
                    if len(results) >= max_results:
                        return (
                            f"Found {len(results)} matches (truncated at {max_results}):\n"
                            + "\n".join(results)
                        )
    if not results:
        return f"No matches found for pattern: {pattern}"
    return f"Found {len(results)} matches:\n" + "\n".join(results)


def dispatch_tool(sandbox: RepoSandbox, tool_name: str, tool_input: dict) -> str:
    """Route a tool call to the correct execution function. Returns result string."""
    try:
        if tool_name == "list_directory":
            return execute_list_directory(sandbox, tool_input["path"])
        elif tool_name == "read_file":
            return execute_read_file(
                sandbox,
                tool_input["path"],
                offset=tool_input.get("offset", 1),
                limit=tool_input.get("limit", 1000),
            )
        elif tool_name == "search_code":
            return execute_search_code(
                sandbox,
                tool_input["pattern"],
                path=tool_input.get("path", "."),
                file_glob=tool_input.get("file_glob"),
            )
        else:
            return f"Error: unknown tool: {tool_name}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error executing {tool_name}: {e}"
