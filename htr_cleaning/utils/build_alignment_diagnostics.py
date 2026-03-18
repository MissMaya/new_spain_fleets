"""
build_alignment_diagnostics

Generate document-level diagnostics for Step 2 alignment behaviour.

This function analyses the distribution of Step 2 alignment tags
(S2X, S2I, S2D) across all documents in the corpus and identifies
cases where alignment may have drifted or failed.

We measure the relative frequency of:

    - S2X : substitutions
    - S2I : insertions (extra HTR text)
    - S2D : deletions (missing HTR text)

Documents with unusually high insertion or deletion ratios are flagged
as potential alignment failures.

Inputs
------
Reads per-document issue logs:

    logs/<style>/<doc_id>/<doc_id>_issues.json

Only Step 2 tags are analysed.

Outputs
-------
Writes a corpus-level diagnostics file:

    logs/meta/alignment_diagnostics.json

Each entry contains:

    - document_id
    - style
    - total_s2
    - S2X
    - S2I
    - S2D
    - insert_ratio
    - delete_ratio
    - replace_ratio
    - flag

Where `flag` is:

    OK
        Alignment behaviour appears normal.

    ALIGNMENT_DRIFT
        High insertion or deletion ratios suggest alignment may have
        drifted or failed.

Pipeline position
-----------------
This function should be run after:

    run_step2()
    run_step3()
    build_issue_index()

and before:

    assign_issue_ids_all_logs()

"""

from pathlib import Path
from collections import defaultdict

from utils.config import LOGS_DIR
from utils.file_io import load_json_if_exists, safe_write_json


def build_alignment_diagnostics():

    rows = []

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

            s2_counts = defaultdict(int)

            for issue in issues:
                tag = issue.get("tag")

                if tag == "S2X":
                    s2_counts["X"] += 1
                elif tag == "S2I":
                    s2_counts["I"] += 1
                elif tag == "S2D":
                    s2_counts["D"] += 1

            total = sum(s2_counts.values())

            if total == 0:
                continue

            insert_ratio = s2_counts["I"] / total
            delete_ratio = s2_counts["D"] / total
            replace_ratio = s2_counts["X"] / total

            flag = "OK"

            if insert_ratio > 0.4 or delete_ratio > 0.4:
                flag = "ALIGNMENT_DRIFT"

            rows.append({
                "document_id": doc_id,
                "style": style,
                "total_s2": total,
                "S2X": s2_counts["X"],
                "S2I": s2_counts["I"],
                "S2D": s2_counts["D"],
                "insert_ratio": round(insert_ratio, 3),
                "delete_ratio": round(delete_ratio, 3),
                "replace_ratio": round(replace_ratio, 3),
                "flag": flag,
            })

    rows.sort(key = lambda x: x["total_s2"], reverse = True)

    meta_dir = LOGS_DIR / "meta"
    meta_dir.mkdir(parents = True, exist_ok = True)

    safe_write_json(rows, meta_dir / "alignment_diagnostics.json")

    print("Alignment diagnostics written.")