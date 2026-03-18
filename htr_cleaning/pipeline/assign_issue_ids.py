"""
assign_issue_ids.py

Attach unique, deterministic IDs to all issues after Steps 1-3.

Reads:
    logs/<style>/<doc_id>/<doc_id>_issues.json

Writes:
    logs/<style>/<doc_id>/issues_with_ids.json

Must be run AFTER:
    run_step1() 
    run_step2() 
    run_step3()
"""

from pathlib import Path

from utils.config import LOGS_DIR
from utils.issue_ids import generate_issue_id
from utils.file_io import issues_json_path, safe_write_json, load_json_if_exists


def assign_issue_ids_all_logs():
    total_count = 0

    for style_dir in LOGS_DIR.iterdir():

        if not style_dir.is_dir():
            continue

        if style_dir.name in ("posthoc", "meta", "review"):
            continue

        for doc_dir in style_dir.iterdir():

            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            issues_path = issues_json_path(doc_dir, doc_id)

            if not issues_path.exists():
                continue

            issues = load_json_if_exists(issues_path, [])

            updated = []

            for issue in issues:

                tag = issue.get("tag")
                start = issue.get("_abs_start")
                end = issue.get("_abs_end")

                if tag is None or start is None or end is None:
                    updated.append(issue)
                    continue

                issue_id = generate_issue_id(doc_id, tag, start, end)

                issue["issue_id"] = issue_id

                updated.append(issue)
                total_count += 1

            output_path = doc_dir / "issues_with_ids.json"

            safe_write_json(updated, output_path)

    print(f"\nAssigned IDs to {total_count} issues.")
    print("issues_with_ids.json files written.")


if __name__ == "__main__":
    assign_issue_ids_all_logs()