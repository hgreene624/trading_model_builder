#!/usr/bin/env python3
"""
Project directory tree printer (files shown by default).

Usage (from project root):
  python print_tree.py                       # files + dirs, depth=5
  python print_tree.py --max-depth 7         # deeper
  python print_tree.py --dirs-only           # hide files
  python print_tree.py --root /path --out tree.txt

Skips typical noise: .git, .venv, __pycache__, node_modules, etc.
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import Iterator, List, Tuple

DEFAULT_IGNORES = {
    ".git", ".svn", ".hg", ".idea", ".vscode",
    ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".ipynb_checkpoints", ".DS_Store",
}

def iter_entries(path: Path, show_files: bool) -> Iterator[Path]:
    try:
        with os.scandir(path) as it:
            for entry in it:
                name = entry.name
                if name in DEFAULT_IGNORES:
                    continue
                if name.startswith(".") and name not in {".env", ".envrc"}:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    yield Path(entry.path)
                elif show_files:
                    yield Path(entry.path)
    except PermissionError:
        return

def sort_key(p: Path) -> Tuple[int, str]:
    return (1 if p.is_file() else 0, p.name.lower())

def tree(root: Path, max_depth: int, show_files: bool) -> List[str]:
    lines: List[str] = []
    lines.append(f"{root.resolve().name}/")

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if max_depth >= 0 and depth > max_depth:
            return
        children = sorted(iter_entries(dir_path, show_files), key=sort_key)
        total = len(children)
        for i, child in enumerate(children):
            last = (i == total - 1)
            connector = "└── " if last else "├── "
            if child.is_dir():
                lines.append(f"{prefix}{connector}{child.name}/")
                extension = "    " if last else "│   "
                if max_depth < 0 or depth < max_depth:
                    walk(child, prefix + extension, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{child.name}")

    walk(root, "", 1)
    return lines

def main() -> int:
    ap = argparse.ArgumentParser(description="Print a clean directory tree (files shown by default).")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: CWD)")
    ap.add_argument("--max-depth", type=int, default=5, help="Max depth; -1 = unlimited (default: 5)")
    ap.add_argument("--dirs-only", action="store_true", help="Show directories only (hide files)")
    ap.add_argument("--ignore", nargs="*", default=[], help="Extra names to ignore")
    ap.add_argument("--out", type=Path, default=None, help="Write output to file")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"error: root path not found or not a directory: {root}", file=sys.stderr)
        return 2

    DEFAULT_IGNORES.update(set(args.ignore))
    lines = tree(root, max_depth=args.max_depth, show_files=not args.dirs_only)
    text = "\n".join(lines) + "\n"

    if args.out:
        try:
            args.out.write_text(text, encoding="utf-8")
            print(f"Wrote tree to {args.out}")
        except Exception as e:
            print(f"error: failed to write {args.out}: {e}", file=sys.stderr)
            return 3
    else:
        print(text, end="")
    return 0

if __name__ == "__main__":
    sys.exit(main())