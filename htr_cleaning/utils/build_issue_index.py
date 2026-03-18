"""
build_issue_index.py

Script to create a corpus-wide index of detected issues across all documents.

This function scans every document log produced by Steps 1-3 of the
pipeline and aggregates issue counts into a single index file.

The index provides a fast overview of the distribution of detected
issues across the training corpus. It allows downstream components
(e.g. review sampling and diagnostics) to analyse issue frequency,
document-level error density, and step-level tag distribution without
having to re-read individual log files.

Inputs
------
Reads per-document issue logs:

    logs/<style>/<doc_id>/<doc_id>_issues.json

For the aggregation to work, each issue record should contain:
    - tag
    - _abs_start
    - _abs_end
    - line
    - char_start
    - char_end

Outputs
-------
Writes a corpus-level issue index:

    logs/meta/issue_index.json

The index contains a list of entries summarising each issue with key fields,
including:

    - document_id
    - style
    - step
    - tag
    - line
    - char_start
    - char_end

Pipeline position
-----------------
This function should be run after:

    run_step1()
    run_step2()
    run_step3()

and before:

    run_posthoc_analysis()
    assign_issue_ids_all_logs()
    build_review_pool()
"""

from pathlib import Path
from collections import defaultdict
from utils.file_io import load_json_if_exists, safe_write_json
from utils.config import LOGS_DIR


def build_issue_index():

    index_rows = []

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir():
            continue

        style = style_dir.name

        for doc_dir in style_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name

            issues_path = doc_dir / f"{doc_id}_issues.json"

            issues = load_json_if_exists(issues_path, [])

            if not issues:
                continue

            counts = defaultdict(int)

            for issue in issues:
                tag = issue.get("tag", "")

                if tag.startswith("S1"):
                    counts["S1"] += 1
                elif tag.startswith("S2"):
                    counts["S2"] += 1
                elif tag.startswith("S3"):
                    counts["S3"] += 1

            total = counts["S1"] + counts["S2"] + counts["S3"]

            index_rows.append({
                "document_id": doc_id,
                "style": style,
                "total_issues": total,
                "S1": counts["S1"],
                "S2": counts["S2"],
                "S3": counts["S3"],
            })

    # Sort by total issues descending
    index_rows.sort(key = lambda x: x["total_issues"], reverse = True)

    meta_dir = LOGS_DIR / "meta"
    meta_dir.mkdir(parents = True, exist_ok = True)

    out_path = meta_dir / "issues_index.json"

    safe_write_json(index_rows, out_path)

    print(f"Issue index written: {out_path}")