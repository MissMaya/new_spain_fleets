"""
logging.py

Helpers for recording transcription issues.

Two representations for each HTR file in the train set:

1. A machine-readable JSON log
2. A human-readable TXT log

The helpers do the following:
- Create per-document log files grouped by calligraphy style
- Append new issues
- Prevent duplicate entries
- Maintain both JSON and TXT logs

"""

from pathlib import Path
from typing import Dict, Any

from utils.file_io import load_json_if_exists, safe_write_json


# ----------------------------------------------------------------------
# Duplicate detection
# ----------------------------------------------------------------------

def is_duplicate(existing_entries, new_entry):
    """
    Check whether an identical issue has already been logged.

    Two issues are considered duplicates if they share:
    - the same tag
    - the same absolute span (_abs_start, _abs_end)
    """
    for e in existing_entries:
        if (
            e.get("tag") == new_entry.get("tag")
            and e.get("_abs_start") == new_entry.get("_abs_start")
            and e.get("_abs_end") == new_entry.get("_abs_end")
        ):
            return True
    return False


# ----------------------------------------------------------------------
# Human-readable formatting
# ----------------------------------------------------------------------

def format_issue_for_text(issue: Dict[str, Any]) -> str:
    """
    Convert an issue dict into a readable multi-field one-line text representation.

    This record exists to help humans. The JSON log is the canonical form.
    """
    tag = issue.get("tag", "")
    desc = issue.get("description", "")
    line = issue.get("line", "")
    cs = issue.get("char_start", "")
    ce = issue.get("char_end", "")

    review_status = issue.get("review", {}).get("status", "unknown")

    htr = issue.get("htr_text", "")
    gt = issue.get("gt_text", "")

    parts = [
        f"[{tag}]",
        f"line {line}:{cs}-{ce}",
        f"({review_status})",
    ]

    if desc:
        parts.append(desc)

    text = " ".join(parts)

    if htr:
        text += f"\n  HTR: {htr}"
    if gt:
        text += f"\n  GT : {gt}"

    return text + "\n\n"


# ----------------------------------------------------------------------
# Main logger
# ----------------------------------------------------------------------

def log_issue(
    logs_dir: Path,
    calligraphy_type: str,
    document_id: str,
    issue: Dict[str, Any],
    ):

    """
    Append a detected issue to the per-document JSON and TXT logs.

    Parameters
    ----------
    logs_dir : Path
        Root logs directory.

    calligraphy_type : str
        Style/category of the document (e.g. encadenada).

    document_id : str
        Base document identifier.

    issue : dict
        Issue payload (the schema).
    """

    doc_log_dir = logs_dir / calligraphy_type / document_id
    doc_log_dir.mkdir(parents = True, exist_ok = True)

    json_path = doc_log_dir / f"{document_id}_issues.json"
    txt_path = doc_log_dir / f"{document_id}_issues.txt"

    existing = load_json_if_exists(json_path, default=[])

    if not is_duplicate(existing, issue):

        # add issue
        existing.append(issue)

        # sort in order of line number and character positions 
        existing.sort(
            key = lambda x: (
                x.get("line", 0),
                x.get("_abs_start", 0),
                x.get("char_start", 0),
            )
        )

        # write JSON
        safe_write_json(existing, json_path)

        # append TXT log
        with open(txt_path, "a", encoding = "utf-8") as f:
            f.write(format_issue_for_text(issue))
