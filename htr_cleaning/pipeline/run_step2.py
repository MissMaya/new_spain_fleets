"""
run_step2.py

Pipeline stage for Step 2: character-level GT-HTR alignment.

This stage:

- Loads training HTR-GT pairs from logs/meta/train_pairs.json
- Loads Step 1 spans (produced by run_step1)
- Performs full-text alignment using utils.alignment
- Logs Step 2 issues (X / I / D)
- Computes confusion matrices by calligraphy style
- Writes S1-S2 overlap metadata to logs/posthoc/s1_s2_overlap.json
- Produces CSV + PNG confusion matrices for review by humans in the loop

Typical usage:

    python pipeline/run_step2.py

or via:

    python run_pipeline.py
"""

from pathlib import Path

from utils.config import LOGS_DIR
from utils.file_io import load_json_if_exists
from utils.processing import process_step2_issues
from utils.visualise import write_confusion_matrices


def run_step2():
    print("Starting Step 2: character-level alignment")

    meta_dir = LOGS_DIR / "meta"
    train_pairs_path = meta_dir / "train_pairs.json"

    train_pairs = load_json_if_exists(train_pairs_path, [])

    if not train_pairs:
        raise RuntimeError(
            "No training pairs found. Please run run_split.py and run_step1.py first."
        )

    # ------------------------------------------------------------------
    # Load Step 1 spans from logs (per-document)
    # ------------------------------------------------------------------

    # Step 1 spans are stored in per-document logs.
    # process_step1_issues already returned spans during Step 1 but at this
    # stage we reload them from disk for reproducibility.

    step1_spans_by_file = {}

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
            step1_spans = [i for i in issues if i.get("tag") in ["L", "T", "W", "SP", "C", "P", "G", "M", "MP"]]

            if step1_spans:
                step1_spans_by_file[doc_id] = step1_spans

    # ------------------------------------------------------------------
    # Run Step 2 processing
    # ------------------------------------------------------------------

    confusion_by_style, overlap_metadata = process_step2_issues(
        train_pairs = train_pairs,
        step1_spans_by_file = step1_spans_by_file,
        logs_dir = LOGS_DIR,
    )

    # ------------------------------------------------------------------
    # Visualisation outputs
    # ------------------------------------------------------------------

    summaries_dir = LOGS_DIR / "step_summaries"
    write_confusion_matrices(confusion_by_style, summaries_dir)

    print("Step 2 complete.")


if __name__ == "__main__":
    run_step2()
