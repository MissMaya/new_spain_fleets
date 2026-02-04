"""
file_io.py

Shared filesystem and JSON utilities for the HTR cleaning pipeline.

This module centralises:
- consistent UTF-8 JSON read/write
- automatic parent-directory creation
- safe JSON writes so output files are never left in a broken or partially-written state
- simple filesystem indexing helpers

Pipeline stages import these helpers rather than reimplementing them.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)


def load_json_if_exists(path: Path, default: Any) -> Any:
    return read_json(path) if path.exists() else default


def write_json(obj: Any, path: Path, indent: int = 2, ensure_ascii: bool = False) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def safe_write_json(obj: Any, path: Path, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Atomic JSON write: writes to a temporary file then renames into place.
    This prevents partially-written JSON if execution is interrupted.
    """
    ensure_parent(path)

    with tempfile.NamedTemporaryFile("w", delete = False, encoding = "utf-8") as tmp:
        json.dump(obj, tmp, indent = indent, ensure_ascii = ensure_ascii)
        tmp_path = tmp.name

    os.replace(tmp_path, path)


def index_txt_files(base_dir: Path) -> list[Path]:
    """
    Return a sorted list of all .txt files under base_dir (recursive).
    """
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("*.txt"))


def index_htr_files_by_style(raw_dir: Path, styles: list[str]) -> dict[str, list[Path]]:
    """
    Return dict mapping style -> sorted list of .txt paths under raw_dir/style.
    """
    return {style: index_txt_files(raw_dir / style) for style in styles}
