"""
issue_ids.py

Assign deterministic, stable IDs to detected issues AFTER Steps 1–3.

This is a post-pass over existing logs.

Reads:
    logs/<style>/<doc_id>/issues.json

Writes:
    logs/<style>/<doc_id>/issues_with_ids.json

Design principles:

- Original issues.json is NEVER modified.
- IDs depend ONLY on document-local content:
    doc_id, step, tag, start, end, line, gt, htr
- Canonical sorting ensures repeatability.
- Duplicate identical issues are disambiguated by occurrence index.
- Adding new documents never changes IDs of existing ones.

This module performs no detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import hashlib

from utils.config import LOGS_DIR
from utils.file_io import load_json_if_exists, safe_write_json


S1_TAGS = {"L", "T", "W", "SP", "C", "P", "G", "M", "MP"}
S2_TAGS = {"X", "I", "D"}


def infer_step(tag: str) -> str:
    if tag in S1_TAGS:
        return "S1"
    if tag in S2_TAGS:
        return "S2"
    return "S3"


def _fingerprint(doc_id: str, step: str, issue: Dict) -> str:
    """
    Build a stable fingerprint string for an issue.
    """
    parts = [
        doc_id,
        step,
        str(issue.get("tag", "")),
        str(issue.get("start", "")),
        str(issue.get("end", "")),
        str(issue.get("line", "")),
        str(issue.get("gt", "")),
        str(issue.get("htr", "")),
    ]
    return "||".join(parts)


def assign_issue_ids_for_doc(doc_id: str, issues: List[Dict]) -> List[Dict]:
    """
    Return a new list of issues where each issue has:
        - issue_id
        - step

    Original issue dicts are not modified.
    """

    # Canonical ordering so duplicates are handled deterministically
    def sort_key(i: Dict):
        return (
            i.get("start", -1),
            i.get("end", -1),
            i.get("tag", ""),
            i.get("line", -1),
            i.get("gt", ""),
            i.get("htr", ""),
        )

    sorted_issues = sorted(issues, key=sort_key)

    seen = {}
    out = []

    for issue in sorted_issues:
        tag = issue.get("tag", "")
        step = infer_step(tag)

        fp = _fingerprint(doc_id, step, issue)
        seen[fp] = seen.get(fp, 0) + 1
        occurrence = seen[fp]

        digest = hashlib.sha1(f"{fp}||{occurrence}".encode("utf-8")).hexdigest()[:10]
        issue_id = f"{doc_id}-{step}-{tag}-{digest}"

        new_issue = dict(issue)
        new_issue["step"] = step
        new_issue["issue_id"] = issue_id

        out.append(new_issue)

    return out


def assign_issue_ids_all_logs():
    """
    Walk logs/<style>/<doc_id>/issues.json and write issues_with_ids.json
    alongside each file.
    """

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir():
            continue

        for doc_dir in style_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            issues_path = doc_dir / "issues.json"

            if not issues_path.exists():
                continue

            issues = load_json_if_exists(issues_path, [])
            if not issues:
                continue

            with_ids = assign_issue_ids_for_doc(doc_id, issues)
            safe_write_json(with_ids, doc_dir / "issues_with_ids.json")