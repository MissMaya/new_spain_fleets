"""
run_render_tags.py

Script for rendering Step 1-3 tags into copies of TEST HTR transcripts
for human review.

This stage assumes Steps 1-3 have already completed.

The script performs two actions:

1) Assigns deterministic IDs to all detected issues (post-pass),
   writing issues_with_ids.json alongside issues.json.

2) Renders tagged copies of TEST HTR files only, writing to:

       data/tagged/<style>/<doc_id>.txt

Raw HTR files and training data are never modified.

Typical usage:

    python pipeline/run_render_tags.py

or via:

    python run_pipeline.py
"""

from utils.config import LOGS_DIR, DATA_DIR
from utils.file_io import load_json_if_exists
from utils.issue_ids import assign_issue_ids_all_logs
from utils.tagging_rendering import render_all_tagged_transcripts


def run_render_tags():
    print("Assigning deterministic issue IDs (post-pass)...")
    assign_issue_ids_all_logs()

    print("Rendering tagged TEST transcripts...")

    meta_dir = LOGS_DIR / "meta"
    test_pairs = load_json_if_exists(meta_dir / "test_pairs.json", [])

    if not test_pairs:
        raise RuntimeError("No test_pairs.json found. Run run_split and Steps 1-3 first.")

    tagged_root = DATA_DIR / "tagged"

    # Explicitly pass TEST pairs into tagging function
    render_all_tagged_transcripts(
        train_pairs = test_pairs,
        logs_dir = LOGS_DIR,
        output_root = tagged_root,
    )

    print("Tag rendering complete (TEST set only).")


if __name__ == "__main__":
    run_render_tags()
