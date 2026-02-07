"""Build repo context for the LLM judge (tree structure + key file contents)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cbench.review.tools import SKIP_DIRS

# Binary/non-text extensions to skip when counting LOC
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".o",
    ".a",
    ".lib",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".AppImage",
    ".deb",
    ".rpm",
}


@dataclass
class RepoStats:
    """Statistics about a repository for cost estimation."""

    total_files: int = 0
    source_files: int = 0
    total_loc: int = 0
    size_category: str = "small"  # small, medium, large, huge
    est_turns: int = 8
    est_input_per_turn: int = 2000
    est_output_per_turn: int = 600


def scan_repo(repo_root: str | Path) -> RepoStats:
    """Scan a repo to estimate size and calibrate cost parameters."""
    root = Path(repo_root).resolve()
    total_files = 0
    source_files = 0
    total_loc = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            if ext in BINARY_EXTENSIONS:
                continue
            total_files += 1
            # Count as source if it has a code-like extension
            if ext in {
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".rs",
                ".go",
                ".java",
                ".c",
                ".cpp",
                ".cc",
                ".cxx",
                ".h",
                ".hpp",
                ".hxx",
                ".swift",
                ".kt",
                ".rb",
                ".php",
                ".cs",
                ".scala",
                ".zig",
                ".lua",
                ".sh",
                ".bash",
                ".zsh",
                ".fish",
                ".sql",
                ".graphql",
                ".proto",
                ".vue",
                ".svelte",
                ".astro",
            }:
                source_files += 1
            # Count lines (cap per file to avoid massive generated files)
            try:
                text = fpath.read_text(errors="replace")
                total_loc += min(len(text.splitlines()), 10_000)
            except (PermissionError, OSError):
                continue

    # Categorize and set estimation parameters
    stats = RepoStats(total_files=total_files, source_files=source_files, total_loc=total_loc)

    if source_files <= 15:
        stats.size_category = "small"
        stats.est_turns = min(8, 30)
        stats.est_input_per_turn = 2000
        stats.est_output_per_turn = 600
    elif source_files <= 50:
        stats.size_category = "medium"
        stats.est_turns = min(15, 30)
        stats.est_input_per_turn = 3500
        stats.est_output_per_turn = 800
    elif source_files <= 200:
        stats.size_category = "large"
        stats.est_turns = min(25, 30)
        stats.est_input_per_turn = 5000
        stats.est_output_per_turn = 1000
    else:
        stats.size_category = "huge"
        stats.est_turns = 30
        stats.est_input_per_turn = 8000
        stats.est_output_per_turn = 1200

    return stats


def build_repo_tree(repo_root: str | Path, max_depth: int = 4) -> str:
    """Build an indented directory tree string for the repo."""
    root = Path(repo_root).resolve()
    lines: list[str] = [root.name + "/"]
    _walk_tree(root, "", max_depth, 0, lines)
    return "\n".join(lines)


def _walk_tree(
    directory: Path, prefix: str, max_depth: int, current_depth: int, lines: list[str]
) -> None:
    if current_depth >= max_depth:
        return
    try:
        entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return

    # Filter out hidden and noisy dirs
    entries = [e for e in entries if not e.name.startswith(".") and e.name not in SKIP_DIRS]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{connector}{entry.name}{suffix}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _walk_tree(entry, prefix + extension, max_depth, current_depth + 1, lines)


# Files to look for when gathering key context for the judge
KEY_FILE_PATTERNS = [
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".env.example",
]

# Source entry-point patterns (checked in order, first match per pattern)
SOURCE_ENTRY_PATTERNS = [
    "main.py",
    "app.py",
    "cli.py",
    "__main__.py",
    "index.ts",
    "index.js",
    "main.ts",
    "main.js",
    "main.go",
    "main.rs",
    "lib.rs",
    "src/main.py",
    "src/app.py",
    "src/index.ts",
    "src/index.js",
    "src/main.go",
    "src/main.rs",
    "src/lib.rs",
]


def gather_key_files(repo_root: str | Path, max_total_chars: int = 50_000) -> str:
    """Read key config/entry files from the repo, budget-capped at max_total_chars."""
    root = Path(repo_root).resolve()
    sections: list[str] = []
    total_chars = 0

    # Gather config/readme files
    for pattern in KEY_FILE_PATTERNS:
        fpath = root / pattern
        if fpath.is_file():
            content = _read_capped(fpath, max_total_chars - total_chars)
            if content is None:
                break
            sections.append(f"=== {pattern} ===\n{content}")
            total_chars += len(content) + len(pattern) + 10
            if total_chars >= max_total_chars:
                break

    # Gather source entry points
    if total_chars < max_total_chars:
        for pattern in SOURCE_ENTRY_PATTERNS:
            fpath = root / pattern
            if fpath.is_file():
                content = _read_capped(fpath, max_total_chars - total_chars)
                if content is None:
                    break
                sections.append(f"=== {pattern} ===\n{content}")
                total_chars += len(content) + len(pattern) + 10
                if total_chars >= max_total_chars:
                    break

    # If we found a src/ or lib/ directory, also grab __init__.py or mod.rs
    for subdir in ["src", "lib"]:
        if total_chars >= max_total_chars:
            break
        src_dir = root / subdir
        if src_dir.is_dir():
            for init_name in ["__init__.py", "mod.rs", "index.ts", "index.js"]:
                init_path = src_dir / init_name
                if init_path.is_file():
                    content = _read_capped(init_path, max_total_chars - total_chars)
                    if content is None:
                        break
                    rel = f"{subdir}/{init_name}"
                    sections.append(f"=== {rel} ===\n{content}")
                    total_chars += len(content) + len(rel) + 10

    if not sections:
        return "(no key files found)"
    return "\n\n".join(sections)


def _read_capped(path: Path, remaining_budget: int) -> str | None:
    """Read a file up to remaining_budget chars. Returns None if budget exhausted."""
    if remaining_budget <= 0:
        return None
    try:
        text = path.read_text(errors="replace")
    except (PermissionError, OSError):
        return None
    if len(text) > remaining_budget:
        text = text[:remaining_budget] + "\n... (truncated)"
    return text
