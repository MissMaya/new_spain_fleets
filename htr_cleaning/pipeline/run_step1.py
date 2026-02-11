"""
run_step1.py

Pipeline Stage 1: Runs basic preprocessing and normalisation of HTR transcriptions.
"""

from utils.config import LOGS_DIR, SCHEMAS_DIR
from utils.file_io import read_json
from utils.tag_rules import all_step1_tags
from utils.processing import process_step1_issues
from utils.visualise import generate_all_outputs


def run_step1():
    print("Starting Step 1: preprocessing + initial tagging")

    meta_dir = LOGS_DIR / "meta"

    train_pairs_path = meta_dir / "train_pairs.json"
    tag_schema_path = SCHEMAS_DIR / "tag_schema.json"

    train_pairs = read_json(train_pairs_path)
    tag_schema = read_json(tag_schema_path)

    if not train_pairs:
        raise RuntimeError("No training pairs found. Please run run_split.py first.")

    # ------------------------------------------------------------------
    # Validate Step 1 tags against schema
    # ------------------------------------------------------------------

    schema_tags = set(tag_schema["S1"].keys())
    rule_tags = set(all_step1_tags.keys())

    if schema_tags != rule_tags:
        raise RuntimeError(
            f"Step 1 tag mismatch:\n"
            f"Schema: {sorted(schema_tags)}\n"
            f"Rules : {sorted(rule_tags)}"
        )

    # Styles derived from training data only
    calligraphy_types = sorted({p["style"] for p in train_pairs})

    # ------------------------------------------------------------------
    # Run Step 1 processing
    # ------------------------------------------------------------------

    error_counts_by_style, _ = process_step1_issues(
        train_pairs = train_pairs,
        step1_tags = all_step1_tags,
        tag_schema = tag_schema,
        calligraphy_types = calligraphy_types,
        logs_dir = LOGS_DIR,
    )

    # ------------------------------------------------------------------
    # Visual summaries
    # ------------------------------------------------------------------

    generate_all_outputs(
        error_counts_by_style = error_counts_by_style,
        step_name = "step1",
        output_dir = LOGS_DIR / "step_summaries",
    )

    print("Step 1 complete.")


if __name__ == "__main__":
    run_step1()