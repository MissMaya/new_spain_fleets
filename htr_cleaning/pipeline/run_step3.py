"""
run_step3.py

Pipeline stage for Step 3: Runs linguistic / paleographic heuristics.

This stage:

- Loads training HTR_GT pairs
- Reloads Step 1 and Step 2 spans from logs (for reproducibility)
- Applies Step 3 heuristic regex rules
- Logs Step 3 issues
- Computes S1<->S3 and S2<->S3 overlap metadata
- Produces per-style Step 3 summaries

Typical usage:

    python pipeline/run_step3.py

or via:

    python run_pipeline.py
"""

from utils.config import LOGS_DIR, SCHEMAS_DIR
from utils.file_io import load_json_if_exists, read_json, issues_json_path
from utils.processing import process_step3_issues
from utils.visualise import generate_all_outputs


def _load_step1_and_step2_spans(step1_tags, step2_tags):
    """
    Load Step 1 and Step 2 spans from logs in a single pass.

    Returns:
        step1_spans_by_file, step2_spans_by_file
    """

    step1_spans_by_file = {}
    step2_spans_by_file = {}

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir():
            continue
        if style_dir.name in {"meta", "posthoc", "review"}:
            continue

        for doc_dir in style_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            issues_path = issues_json_path(doc_dir, doc_id)

            if not issues_path.exists():
                continue

            issues = load_json_if_exists(issues_path, [])

            step1_spans = []
            step2_spans = []

            for i in issues:
                tag = i.get("tag")
                start = i.get("_abs_start")
                end = i.get("_abs_end")

                if tag is None or start is None or end is None:
                    continue

                span = {
                    "tag": tag,
                    "start": start,
                    "end": end,
                }

                if tag in step1_tags:
                    step1_spans.append(span)
                elif tag in step2_tags:
                    step2_spans.append(span)

            if step1_spans:
                step1_spans_by_file[doc_id] = step1_spans

            if step2_spans:
                step2_spans_by_file[doc_id] = step2_spans

    return step1_spans_by_file, step2_spans_by_file


def run_step3():
    print("Starting Step 3: heuristic linguistic tagging")

    meta_dir = LOGS_DIR / "meta"
    train_pairs = load_json_if_exists(meta_dir / "train_pairs.json", [])

    if not train_pairs:
        raise RuntimeError(
            "No training pairs found. Please run run_split.py and tagging for Steps 1 and 2 first."
        )

    # ------------------------------------------------------------------
    # Load schema
    # ------------------------------------------------------------------

    tag_schema = read_json(SCHEMAS_DIR / "tag_schema.json")

    # Build tag sets
    step1_tags = {f"S1{t}" for t in tag_schema["S1"].keys()}
    step2_tags = {f"S2{t}" for t in tag_schema["S2"].keys()}

    # ------------------------------------------------------------------
    # Reload Step 1 + Step 2 spans in one pass
    # ------------------------------------------------------------------

    step1_spans_by_file, step2_spans_by_file = _load_step1_and_step2_spans(
        step1_tags,
        step2_tags,
    )

    # ------------------------------------------------------------------
    # Run Step 3 processing
    # ------------------------------------------------------------------

    error_counts_by_style = process_step3_issues(
        train_pairs=train_pairs,
        step1_spans_by_file=step1_spans_by_file,
        step2_spans_by_file=step2_spans_by_file,
        tag_schema=tag_schema,
        logs_dir=LOGS_DIR,
    )

    # ------------------------------------------------------------------
    # Visual summaries
    # ------------------------------------------------------------------

    summaries_dir = LOGS_DIR / "step_summaries"
    generate_all_outputs(error_counts_by_style, "step3", summaries_dir)

    print("Step 3 complete.")


if __name__ == "__main__":
    run_step3()