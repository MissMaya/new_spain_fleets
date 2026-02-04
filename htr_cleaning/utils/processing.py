"""
processing.py

Core processing functions for the HTR cleaning pipeline.

This module applies tagging rules to HTR transcriptions and records detected
issues with precise character spans and line numbers.

Responsibilities:
- Iterate through training HTR files
- Apply regex-based tagging rules
- Compute global character offsets and line numbers
- Log each detected issue
- Aggregate per-style error counts

Pipeline stages (run_step1.py, run_step2.py, run_step3.py) orchestrate execution.
This module contains reusable processing logic only.
"""

from pathlib import Path
from typing import Dict, List

from utils.logging import log_issue


def _compute_line_offsets(text: str):
    """
    Precompute the starting character offset of each line.

    Returns a list where index i contains the global offset of line i (0-based).
    """
    lines = text.splitlines(keepends = True)
    offsets = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)
    return offsets


def _find_line_number(offsets, char_index):
    """
    Given line start offsets and a character index, return 1-based line number.
    """
    line = 1
    for i, start in enumerate(offsets):
        if start > char_index:
            break
        line = i + 1
    return line


def process_step1_issues(
    train_pairs: List[Dict],
    step1_tags: Dict[str, object],
    tag_schema: Dict,
    calligraphy_types: List[str],
    logs_dir: Path,
):
    """
    Apply Step 1 tagging rules to training HTR files and log detected issues.

    Each detected issue records:
    - tag code
    - human-readable description
    - line number (1-based)
    - global character start offset
    - global character end offset

    Parameters
    ----------
    train_pairs : list of dict
        Training subset produced by run_split.py.

    step1_tags : dict
        Mapping tag_code -> compiled regex.

    tag_schema : dict
        Loaded from schemas_and_manifests/tag_schema.json.

    calligraphy_types : list[str]
        Calligraphy styles present in dataset.

    logs_dir : Path
        Root logs directory.

    Returns
    -------
    error_counts_by_style : dict
        Nested dict: style -> tag -> count.
    """

    error_counts_by_style = {
        style: {tag: 0 for tag in step1_tags.keys()}
        for style in calligraphy_types
    }

    for pair in train_pairs:
        style = pair["style"]
        htr_path = Path(pair["htr_path"])
        document_id = pair["id"]

        try:
            text = htr_path.read_text(encoding = "utf-8")
        except Exception as e:
            print(f"Failed to read {htr_path}: {e}")
            continue

        line_offsets = _compute_line_offsets(text)

        for tag, regex in step1_tags.items():
            for match in regex.finditer(text):
                start = match.start()
                end = match.end()
                line_num = _find_line_number(line_offsets, start)

                issue = {
                    "tag": tag,
                    "description": tag_schema["S1"].get(tag, ""),
                    "line": line_num,
                    "start": start,
                    "end": end,
                    "message": tag_schema["S1"].get(tag, ""),
                }

                log_issue(
                    logs_dir = logs_dir,
                    calligraphy_type = style,
                    document_id = document_id,
                    issue = issue,
                )

                error_counts_by_style[style][tag] += 1

    return error_counts_by_style
