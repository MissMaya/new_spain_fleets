"""
file_io.py

Shared filesystem and JSON utilities for the HTR cleaning pipeline.

This module centralises:

- consistent UTF-8 JSON and text reads
- automatic parent-directory creation
- safe (atomic) writes so output files are never left in a broken or
  partially-written state
- simple filesystem indexing helpers

Pipeline stages import these helpers rather than reimplementing them.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
from typing import Any, Iterable


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

def ensure_parent(path: Path) -> None:
    """
    Ensure parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def read_json(path: Path) -> Any:
    """
    Read UTF-8 JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path: Path, default: Any) -> Any:
    """
    Read JSON if file exists, otherwise return default.
    """
    return read_json(path) if path.exists() else default


def write_json(obj: Any, path: Path, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Write JSON (non-atomic).
    """
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def safe_write_json(obj: Any, path: Path, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Atomic JSON write.

    Writes to a temporary file first, then renames into place.
    Prevents partially-written JSON if execution is interrupted.
    """
    ensure_parent(path)

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=indent, ensure_ascii=ensure_ascii)
        tmp_path = tmp.name

    os.replace(tmp_path, path)


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------

def read_text(path: Path) -> str:
    """
    Read UTF-8 text file.
    """
    return Path(path).read_text(encoding="utf-8")


def safe_write_text(content: str, path: Path) -> None:
    """
    Atomic text write.

    Writes to a temporary file first, then renames into place.
    """
    path = Path(path)
    ensure_parent(path)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------

def index_txt_files(base_dir: Path) -> list[Path]:
    """
    Return sorted list of all .txt files under base_dir (recursive).
    """
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("*.txt"))


def index_htr_files_by_style(raw_dir: Path, styles: Iterable[str]) -> dict[str, list[Path]]:
    """
    Return mapping: style -> sorted list of HTR .txt files.
    """
    return {style: index_txt_files(raw_dir / style) for style in styles}
