"""
issue_ids.py

Assign deterministic, stable IDs to detected issues AFTER tagging Steps 1-3.

This is a post-pass over existing logs.

Reads:
    logs/<style>/<doc_id>/issues.json

Writes:
    logs/<style>/<doc_id>/issues_with_ids.json

Works as follows:

- Original issues.json is NEVER modified.
- IDs depend ONLY on document-local content:
    doc_id, step, tag, start, end, line, gt, htr
- Ensures consistent, repeatable ordering.
- Each occurrence is given a unique index so that repeats of the same issue can be tracked independently.
- Adding new documents never changes IDs of existing ones.

"""
import hashlib
import json
import os
from collections import defaultdict
from typing import Dict, List

# ---------------------------------------------------------------------
# Prefix-based stage inference
# ---------------------------------------------------------------------

def infer_step(tag: str) -> str:
    """
    Infer pipeline step from the tag prefix.
    """
    if not tag:
        return "UNKNOWN"

    if tag.startswith("S1"):
        return "S1"
    if tag.startswith("S2"):
        return "S2"
    if tag.startswith("S3"):
        return "S3"

    return "UNKNOWN"


# ---------------------------------------------------------------------
# Fingerprint builder
# ---------------------------------------------------------------------

def build_fingerprint(issue: Dict, doc_id: str) -> str:
    """
    Create deterministic fingerprint for an issue.
    """
    tag = issue.get("tag", "")
    step = infer_step(tag)

    parts = [
        doc_id,
        step,
        tag,
        str(issue.get("_abs_start")),
        str(issue.get("_abs_end")),
        str(issue.get("line")),
        issue.get("gt_text") or "",
        issue.get("htr_text") or "",
    ]

    return "||".join(parts)


# ---------------------------------------------------------------------
# Single document ID assignment 
# ---------------------------------------------------------------------

def assign_issue_ids_for_doc(doc_id: str, issues: List[Dict]) -> List[Dict]:
    """
    Assign deterministic issue_ids to a list of issues for a document.
    Handles duplicate identical fingerprints by adding occurrence count.
    """
    seen = defaultdict(int)
    updated = []

    for issue in issues:
        fp = build_fingerprint(issue, doc_id)
        seen[fp] += 1
        occurrence = seen[fp]

        digest = hashlib.sha1(
            f"{fp}||{occurrence}".encode("utf-8")
        ).hexdigest()[:10]

        new_issue = issue.copy()
        new_issue["issue_id"] = digest
        updated.append(new_issue)

    return updated


# ---------------------------------------------------------------------
# Batch assignment across all logs
# ---------------------------------------------------------------------

def assign_issue_ids_all_logs(logs_root: str = "logs"):
    """
    Walk through logs/<style>/<doc>/issues.json
    Generate issues_with_ids.json per document.
    """
    if not os.path.isdir(logs_root):
        print(f"[issue_ids] Logs directory not found: {logs_root}")
        return

    for style in os.listdir(logs_root):
        style_path = os.path.join(logs_root, style)

        if not os.path.isdir(style_path):
            continue

        for doc in os.listdir(style_path):
            doc_path = os.path.join(style_path, doc)

            if not os.path.isdir(doc_path):
                continue

            issues_path = os.path.join(doc_path, "issues.json")
            if not os.path.exists(issues_path):
                continue

            with open(issues_path, "r", encoding = "utf-8") as f:
                issues = json.load(f)

            updated = assign_issue_ids_for_doc(doc, issues)

            output_path = os.path.join(doc_path, "issues_with_ids.json")
            with open(output_path, "w", encoding = "utf-8") as f:
                json.dump(updated, f, indent=2, ensure_ascii = False)

            print(f"[issue_ids] Updated: {output_path}")