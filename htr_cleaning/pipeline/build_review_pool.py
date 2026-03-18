"""
build_review_pool.py

Builds the review pool used for manual validation.

Major improvements:
- Reads each issues.json file exactly once (fast)
- Normalises column names across steps
- Ensures compatibility with allocate_reviews.py
- Handles missing fields gracefully
- Avoids unnecessary DataFrame operations until the end
"""

from pathlib import Path
import json
import pandas as pd

from utils.config import LOGS_DIR
from utils.file_io import protect_for_excel


REVIEW_DIR = LOGS_DIR / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Required columns expected by allocate_reviews.py
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "issue_id",
    "calligraphy_type",
    "doc_id",
    "step",
    "tag",
    "description",
    "line",
    "char_start",
    "char_end",
    "htr_text",
    "gt_text",
    "word_gt",
    "word_htr",
]


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def _normalise_issue(issue, style, doc_id):
    """
    Convert issue dict to the canonical review schema.

    Handles differences between Step1 / Step2 / Step3 logs.
    """

    start = issue.get("char_start", issue.get("_abs_start", 0))
    end = issue.get("char_end", issue.get("_abs_end", 0))

    return {
        "issue_id": issue.get("issue_id"),
        "calligraphy_type": style,
        "doc_id": doc_id,
        "step": issue["tag"][:2],   # S1 / S2 / S3
        "tag": issue["tag"],
        "description": issue.get("description"),
        "line": issue.get("line"),
        "char_start": start,
        "char_end": end,
        "htr_text": issue.get("htr_text"),
        "gt_text": issue.get("gt_text"),
        "word_gt": issue.get("word_gt"),
        "word_htr": issue.get("word_htr"),
    }


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

def build_review_pool():

    rows = []

    # -----------------------------------------------------------------
    # Scan logs once (fast)
    # -----------------------------------------------------------------

    for style_dir in LOGS_DIR.iterdir():

        if not style_dir.is_dir():
            continue

        style = style_dir.name

        # skip meta folders
        if style in {"meta", "review", "posthoc"}:
            continue

        for doc_dir in style_dir.iterdir():

            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name

            issues_path = doc_dir / f"{doc_id}_issues.json"

            if not issues_path.exists():
                continue

            with open(issues_path, "r", encoding="utf-8") as f:
                issues = json.load(f)

            for issue in issues:

                rows.append(
                    _normalise_issue(issue, style, doc_id)
                )

    if not rows:
        raise ValueError("No issues found when building review pool.")

    # -----------------------------------------------------------------
    # Convert to dataframe once
    # -----------------------------------------------------------------

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Ensure required columns exist
    # -----------------------------------------------------------------

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[REQUIRED_COLUMNS]

    # -----------------------------------------------------------------
    # Protect values for Excel
    # -----------------------------------------------------------------

    df = protect_for_excel(df)

    # -----------------------------------------------------------------
    # Write master file
    # -----------------------------------------------------------------

    review_pool_path = REVIEW_DIR / "review_pool.csv"
    review_master_path = REVIEW_DIR / "review_master.csv"

    df.to_csv(review_pool_path, index=False, encoding="utf-8-sig")
    df.to_csv(review_master_path, index=False, encoding="utf-8-sig")

    print(f"Review pool written: {review_pool_path}")
    print(f"Review master written: {review_master_path}")
    print(f"Total issues: {len(df)}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_review_pool()