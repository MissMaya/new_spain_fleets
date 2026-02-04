"""
run_step1.py

Pipeline Stage 1: Basic preprocessing and normalisation of HTR transcriptions.

This stage applies rule-based tagging for common transcription issues such as:
- leading/trailing whitespace
- internal spacing anomalies
- malformed or invisible Unicode characters
- punctuation irregularities

It operates only on the training split produced by run_split.py.

Inputs (logs/meta/):

- train_pairs.json
- paired_data.json
- schemas_and_manifests/tag_schema.json

Outputs (logs/):

- Per-document JSON logs of detected issues (grouped by calligraphy style)
- Step-level summary statistics
- Tables and plots summarising error distributions

Typical usage:

    python pipeline/run_step1.py

or as part of the full pipeline:

    python run_pipeline.py

This module performs orchestration only.
All reusable logic lives in utils/.
"""

from pathlib import Path

from utils.config import LOGS_DIR, SCHEMAS_DIR
from utils.file_io import read_json
from utils.tag_rules import all_step1_tags
from utils.processing import process_step1_issues
from utils.visualise import generate_all_outputs


def run_step1():
    print("Starting Step 1: preprocessing + initial tagging")

    meta_dir = LOGS_DIR / "meta"

    train_pairs_path = meta_dir / "train_pairs.json"
    paired_data_path = meta_dir / "paired_data.json"
    tag_schema_path = SCHEMAS_DIR / "tag_schema.json"

    train_pairs = read_json(train_pairs_path)
    paired_data = read_json(paired_data_path)
    tag_schema = read_json(tag_schema_path)

    # Extract calligraphy styles from paired data
    calligraphy_types = sorted({p["style"] for p in paired_data})

    # Core Step 1 processing
    error_counts_by_style = process_step1_issues(
        train_pairs = train_pairs,
        step1_tags = all_step1_tags,
        tag_schema = tag_schema,
        calligraphy_types = calligraphy_types,
        logs_dir = LOGS_DIR,
    )

    # Generate tables and plots summarising results
    generate_all_outputs(
        error_counts_by_style = error_counts_by_style,
        step_name = "step1",
        output_dir = LOGS_DIR / "step_summaries",
    )

    print("Step 1 complete.")


if __name__ == "__main__":
    run_step1()
