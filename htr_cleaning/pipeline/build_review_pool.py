"""
build_review_pool.py

Build a single review pool from all issues_with_ids.json files.
Produce this single pool as a master csv file of issues.
"""

from pathlib import Path
import json
import pandas as pd
from typing import List, Dict

from utils.config import LOGS_DIR


REVIEW_DIR = LOGS_DIR / "review"
REVIEW_DIR.mkdir(parents = True, exist_ok = True)


def _extract_step(tag: str) -> str:
    """Extract step identifier from tag (S1X → S1, etc.)."""
    if tag.startswith("S1"):
        return "S1"
    if tag.startswith("S2"):
        return "S2"
    if tag.startswith("S3"):
        return "S3"
    return "UNKNOWN"


def _load_issues(path: Path) -> List[Dict]:
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)


def build_review_pool():
    rows = []

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir():
            continue
        if style_dir.name in ("posthoc", "meta", "review"):
            continue

        style = style_dir.name

        for doc_dir in style_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            issues_path = doc_dir / "issues_with_ids.json"

            if not issues_path.exists():
                continue

            issues = _load_issues(issues_path)

            for issue in issues:
                issue_id = issue.get("issue_id")
                if not issue_id:
                    raise ValueError(f"Missing issue_id in {doc_id}")

                tag = issue.get("tag", "")
                step = _extract_step(tag)

                rows.append({
                    "issue_id": issue_id,
                    "calligraphy_type": style,
                    "doc_id": doc_id,
                    "step": step,
                    "tag": tag,
                    "description": issue.get("description"),

                    "line": issue.get("line"),
                    "char_start": issue.get("char_start"),
                    "char_end": issue.get("char_end"),
                    "_abs_start": issue.get("_abs_start"),
                    "_abs_end": issue.get("_abs_end"),

                    "htr_text": issue.get("htr_text", ""),
                    "gt_text": issue.get("gt_text", ""),

                    "word_gt": issue.get("word_gt"),
                    "word_htr": issue.get("word_htr"),

                    "has_overlap_s1": bool(issue.get("overlaps_step1", [])),
                    "has_overlap_s2": bool(issue.get("overlaps_step2", [])),

                    # Review fields
                    "assigned_reviewer": "",
                    "review_status": "unassigned",
                    "decision": "",
                    "correction": "",
                    "reviewer": "",
                    "notes": "",
                })

    df = pd.DataFrame(rows)

    output_path = REVIEW_DIR / "review_pool.csv"
    df.to_csv(output_path, index = False)

    print(f"\nReview pool written to: {output_path}")
    print(f"Total issues in pool: {len(df)}")


if __name__ == "__main__":
    build_review_pool()