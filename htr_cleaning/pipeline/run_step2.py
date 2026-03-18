"""
run_step2.py

Pipeline stage for Step 2: Runs character-level HTR-GT alignment.

This stage:

- Loads training HTR-GT pairs
- Reloads Step 1 spans from logs (for reproducibility)
- Performs full-text alignment
- Logs Step 2 issues (S2X => substitution / S2I => insertion / S2D => deletion)
- Computes confusion matrices by calligraphy style
- Writes S1<->S2 overlap metadata
- Produces CSV + PNG confusion matrices

Typical usage:

    python pipeline/run_step2.py

or via:

    python run_pipeline.py
"""

from utils.config import LOGS_DIR, SCHEMAS_DIR
from utils.file_io import load_json_if_exists, read_json
from utils.processing import process_step2_issues
from utils.visualise import write_confusion_matrices
from utils.file_io import issues_json_path


def _load_step1_spans(tag_schema):
    """
    Reload Step 1 spans from issues.json using tag_schema as the source of truth.

    Returns:
        dict: doc_id -> list of {tag,start,end}
    """

    # Build S1 tag set dynamically from schema
    s1_tags = {f"S1{t}" for t in tag_schema["S1"].keys()}

    spans_by_file = {}

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir():
            continue

        for doc_dir in style_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            issues_path = issues_json_path(doc_dir, doc_id)

            if not issues_path.exists():
                continue

            issues = load_json_if_exists(issues_path, [])

            spans = [
                {
                    "tag": i["tag"],
                    "start": i["_abs_start"],
                    "end": i["_abs_end"],
                }
                for i in issues
                if i.get("tag") in s1_tags and "_abs_start" in i
            ]

            if spans:
                spans_by_file[doc_id] = spans

    return spans_by_file


def run_step2():
    print("Starting Step 2: character-level alignment")

    meta_dir = LOGS_DIR / "meta"
    train_pairs = load_json_if_exists(meta_dir / "train_pairs.json", [])

    if not train_pairs:
        raise RuntimeError(
            "No training pairs found. Please run run_split.py and run_step1.py first."
        )

    # ------------------------------------------------------------------
    # Load schema
    # ------------------------------------------------------------------

    tag_schema = read_json(SCHEMAS_DIR / "tag_schema.json")

    # ------------------------------------------------------------------
    # Reload Step 1 spans
    # ------------------------------------------------------------------

    step1_spans_by_file = _load_step1_spans(tag_schema)

    # ------------------------------------------------------------------
    # Run Step 2 processing
    # ------------------------------------------------------------------

    confusion_by_style, overlap_metadata, step2_spans_by_file = process_step2_issues(
        train_pairs = train_pairs,
        step1_spans_by_file = step1_spans_by_file,
        tag_schema = tag_schema,
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