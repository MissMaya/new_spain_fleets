"""
assign_issue_ids.py

Script attaches a unique ID to all issues once steps 1-3 have been run.

Reads:
    logs/<style>/<doc_id>/issues.json

Writes:
    logs/<style>/<doc_id>/issues_with_ids.json

Must be run AFTER:
    run_step1()
    run_step2()
    run_step3()
    run_posthoc_analysis()
"""

from pathlib import Path
import json

from utils.config import LOGS_DIR
from utils.issue_ids import generate_issue_id


def assign_issue_ids_all_logs():
    total_count = 0

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
            issues_path = doc_dir / "issues.json"

            if not issues_path.exists():
                continue

            with open(issues_path, "r", encoding="utf-8") as f:
                issues = json.load(f)

            updated = []

            for issue in issues:
                tag = issue.get("tag")
                start = issue.get("_abs_start")
                end = issue.get("_abs_end")

                if tag is None or start is None or end is None:
                    continue

                issue_id = generate_issue_id(doc_id, tag, start, end)
                issue["issue_id"] = issue_id

                updated.append(issue)
                total_count += 1

            output_path = doc_dir / "issues_with_ids.json"

            with open(output_path, "w", encoding = "utf-8") as f:
                json.dump(updated, f, ensure_ascii = False, indent = 2)

    print(f"\nAssigned IDs to {total_count} issues.")
    print("issues_with_ids.json files written.")


if __name__ == "__main__":
    assign_issue_ids_all_logs()