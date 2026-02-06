"""
tagging_rendering.py

Render detected issues (Steps 1–3) inline into HTR transcripts for human review.

Goals
-----
Tags produced by this module are:

- Human-readable: reviewers can scan and search easily.
- Machine-parseable: deterministic, consistent syntax.
- Unambiguous: every tag includes its step + code (e.g. S1G, S2X).
- Nested-safe: tags are properly opened/closed and rendering is deterministic.
- Reviewer-friendly: optional short descriptions can be included in the opening tag.

Tag format
----------
By default, issues are rendered as balanced bracket tags:

    [S1G]...[/S1G]
    [S2D]Ø[/S2D]        # deletions (zero-length spans) render as Ø
    [S2I]...[/S2I]      # insertions may be encoded as zero-length spans too

Optionally, opening tags can include a description:

    [S1G|Non-Latin glyph]...[/S1G]

Assumptions about issue objects
-------------------------------
Each issue dict must contain:
- "tag": str  (e.g. "S1G", "S2X", "S3Q")
- "start": int  (0-based char offset in the HTR text)
- "end": int    (0-based char offset in the HTR text; end is exclusive)

Optional fields:
- "desc": str  (free text to show in opening tag; if absent, a schema lookup may be used)
- "id": str/int (deterministic issue id; not required for rendering)

This module does not modify issue logs; it only renders tags into text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils.file_io import read_json, read_text, safe_write_text
from utils.config import SCHEMAS_DIR

# Use the same null char across the project (Step 2 deletions)
NULL_CHAR = "Ø"


# ---------------------------------------------------------------------
# Tag schema loading (optional)
# ---------------------------------------------------------------------

def load_tag_schema(schema_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Load tag_schema.json, returning a mapping like:
        {"S1": {"G": "Non-Latin ...", ...}, "S2": {...}, "S3": {...}}

    If schema file is missing, returns {}.
    """
    schema_path = schema_path or (SCHEMAS_DIR / "tag_schema.json")
    if not schema_path.exists():
        return {}
    try:
        return read_json(schema_path)
    except Exception:
        return {}


def describe_tag(tag: str, schema: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Map a full tag like 'S1G' to a description using the schema.
    Returns None if no description found.
    """
    if not tag or len(tag) < 3 or not tag.startswith("S"):
        return None

    step = tag[:2]  # 'S1', 'S2', 'S3'
    code = tag[2:]  # 'G', 'X', 'DC', ...
    return schema.get(step, {}).get(code)


# ---------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------

def _validate_issue(issue: Dict) -> None:
    if "tag" not in issue:
        raise ValueError("Issue missing required key: 'tag'")
    if "start" not in issue or "end" not in issue:
        raise ValueError("Issue missing required keys: 'start'/'end'")
    if not isinstance(issue["start"], int) or not isinstance(issue["end"], int):
        raise ValueError("Issue 'start'/'end' must be integers")
    if issue["start"] < 0 or issue["end"] < 0:
        raise ValueError("Issue 'start'/'end' must be >= 0")
    if issue["end"] < issue["start"]:
        raise ValueError("Issue 'end' must be >= 'start'")


def _opening_token(tag: str, desc: Optional[str] = None) -> str:
    """
    Build opening token. If desc present, use: [TAG|desc]
    """
    if desc:
        # Avoid newlines to keep tags single-line friendly
        desc = " ".join(str(desc).splitlines()).strip()
        return f"[{tag}|{desc}]"
    return f"[{tag}]"


def _closing_token(tag: str) -> str:
    return f"[/{tag}]"


def render_tags_for_text(
    text: str,
    issues: List[Dict],
    tag_schema: Optional[Dict[str, Dict[str, str]]] = None,
    include_descriptions: bool = True,
) -> str:
    """
    Render issues into a provided text string.

    Notes on correctness / nested-safety
    -----------------------------------
    - Insertions are applied from the bottom up (descending offsets) so earlier
      insertions do not shift later offsets.
    - For span issues (start < end), we insert an opening token at start and a
      closing token at end.
    - For zero-length issues (start == end), we insert:
          [TAG]Ø[/TAG]
      at that position.

    If spans overlap, tags may become nested. Rendering remains well-formed because
    we always insert close tokens at the end index and open tokens at the start
    index, and we apply all insertions from right to left.
    """

    tag_schema = tag_schema or {}
    inserts: List[Tuple[int, str]] = []

    for issue in issues:
        _validate_issue(issue)

        tag = str(issue["tag"]).strip()
        start = issue["start"]
        end = issue["end"]

        desc = None
        if include_descriptions:
            desc = issue.get("desc") or describe_tag(tag, tag_schema)

        open_tok = _opening_token(tag, desc)
        close_tok = _closing_token(tag)

        if start == end:
            inserts.append((start, f"{open_tok}{NULL_CHAR}{close_tok}"))
        else:
            inserts.append((start, open_tok))
            inserts.append((end, close_tok))

    # Sort by position descending; at same position, insert longer tokens first
    # to keep deterministic output.
    inserts.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

    tagged = text
    for pos, token in inserts:
        if pos > len(tagged):
            # Defensive: skip out-of-range issues (shouldn't happen if offsets are correct)
            continue
        tagged = tagged[:pos] + token + tagged[pos:]

    return tagged


def render_tags_for_document(
    htr_path: Path,
    issues: List[Dict],
    tag_schema: Optional[Dict[str, Dict[str, str]]] = None,
    include_descriptions: bool = True,
) -> str:
    """
    Load HTR text from htr_path and render tags inline.
    """
    text = read_text(htr_path)
    return render_tags_for_text(
        text=text,
        issues=issues,
        tag_schema=tag_schema,
        include_descriptions=include_descriptions,
    )


# ---------------------------------------------------------------------
# Batch rendering
# ---------------------------------------------------------------------

def render_tagged_corpus(
    pairs: Iterable[Dict],
    logs_dir: Path,
    output_root: Path,
    issues_filename: str = "issues.json",
    include_descriptions: bool = True,
    schema_path: Optional[Path] = None,
) -> None:
    """
    Render tagged HTR files for a set of paired documents.

    Expected logs layout:
        logs/<style>/<doc_id>/issues.json

    Output layout:
        output_root/<style>/<doc_id>.txt

    Parameters
    ----------
    pairs:
        Iterable of pair dicts. Each pair dict must contain:
            - "style"
            - "id"
            - "htr_path"
    logs_dir:
        Path to logs root (e.g. LOGS_DIR).
    output_root:
        Folder to write tagged HTR copies into (e.g. DATA_DIR / "tagged").
    issues_filename:
        Usually "issues.json".
    include_descriptions:
        If True, uses schema descriptions in opening tags: [S1G|desc]
    schema_path:
        Optional override for tag_schema.json
    """
    tag_schema = load_tag_schema(schema_path)

    output_root.mkdir(parents=True, exist_ok=True)

    from utils.file_io import load_json_if_exists

    for pair in pairs:
        style = pair["style"]
        doc_id = pair["id"]
        htr_path = Path(pair["htr_path"])

        issues_path = logs_dir / style / doc_id / issues_filename
        issues = load_json_if_exists(issues_path, [])

        if not issues:
            continue

        tagged_text = render_tags_for_document(
            htr_path=htr_path,
            issues=issues,
            tag_schema=tag_schema,
            include_descriptions=include_descriptions,
        )

        out_dir = output_root / style
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{doc_id}.txt"
        safe_write_text(tagged_text, out_path)