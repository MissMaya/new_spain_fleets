"""
processing.py

Core processing functions for the HTR cleaning pipeline.

This module orchestrates Step 1 and Step 2 tagging using reusable utilities:

- Step 1: rule-based preprocessing tags (regex)
- Step 2: character-level alignment tags (difflib)

Responsibilities:
- Iterate through training HTR–GT pairs
- Apply Step 1 regex rules
- Apply Step 2 alignment
- Compute global character spans and line numbers
- Log detected issues
- Aggregate per-style summaries
- Produce Step1–Step2 overlap metadata
- Accumulate confusion-matrix counts (with Ø for insert/delete)

Pipeline stages (run_step1.py, run_step2.py) call into this module.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from utils.logging import log_issue
from utils.alignment import align_and_tag, NULL_CHAR
from utils.file_io import safe_write_json, read_text


# ----------------------------------------------------------------------
# Shared helpers (also used by Step 1)
# ----------------------------------------------------------------------

def _compute_line_offsets(text: str):
    lines = text.splitlines(keepends=True)
    offsets = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)
    return offsets


def _find_line_number(offsets, char_index):
    line = 1
    for i, start in enumerate(offsets):
        if start > char_index:
            break
        line = i + 1
    return line


# ----------------------------------------------------------------------
# STEP 1
# ----------------------------------------------------------------------

def process_step1_issues(
    train_pairs: List[Dict],
    step1_tags: Dict[str, object],
    tag_schema: Dict,
    calligraphy_types: List[str],
    logs_dir: Path,
):
    """
    Apply Step 1 tagging rules to training HTR files and log detected issues.

    Returns:
        error_counts_by_style
        step1_spans_by_file: dict[file_id] -> list of spans
    """

    error_counts_by_style = {
        style: {tag: 0 for tag in step1_tags.keys()}
        for style in calligraphy_types
    }

    step1_spans_by_file = defaultdict(list)

    for pair in train_pairs:
        style = pair["style"]
        htr_path = Path(pair["htr_path"])
        document_id = pair["id"]

        try:
            text = htr_path.read_text(encoding="utf-8")
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
                }

                log_issue(
                    logs_dir=logs_dir,
                    calligraphy_type=style,
                    document_id=document_id,
                    issue=issue,
                )

                error_counts_by_style[style][tag] += 1
                step1_spans_by_file[document_id].append(issue)

    return error_counts_by_style, step1_spans_by_file


# ----------------------------------------------------------------------
# STEP 2
# ----------------------------------------------------------------------

def process_step2_issues(
    train_pairs: List[Dict],
    step1_spans_by_file: Dict[str, List[Dict]],
    logs_dir: Path,
):
    """
    Perform full-text GT↔HTR alignment and log Step 2 issues.

    Produces:
    - Per-document Step 2 logs
    - Confusion-matrix counts by style (including Ø)
    - Step1–Step2 overlap metadata

    Returns
    -------
    confusion_by_style : dict[style][gt_char][htr_char] -> count
    overlap_metadata : dict suitable for writing to s1_s2_overlap.json
    """

    # confusion_by_style[style][gt_char][htr_char] += 1
    confusion_by_style = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # overlap counters
    overlap_totals = defaultdict(int)
    overlap_by_style = defaultdict(lambda: defaultdict(int))

    for pair in train_pairs:
        style = pair["style"]
        doc_id = pair["id"]

        gt_text = read_text(pair["gt_path"])
        htr_text = read_text(pair["htr_path"])

        step1_spans = step1_spans_by_file.get(doc_id, [])

        issues = align_and_tag(gt_text, htr_text, step1_spans)

        for issue in issues:
            log_issue(
                logs_dir=logs_dir,
                calligraphy_type=style,
                document_id=doc_id,
                issue=issue,
            )

            gt_seg = issue["gt"] if issue["gt"] else NULL_CHAR
            htr_seg = issue["htr"] if issue["htr"] else NULL_CHAR

            # For spans, accumulate per-character to build confusion matrix
            if len(gt_seg) == 1 and len(htr_seg) == 1:
                confusion_by_style[style][gt_seg][htr_seg] += 1
            else:
                # For longer spans, count character-wise where possible
                max_len = max(len(gt_seg), len(htr_seg))
                for i in range(max_len):
                    g = gt_seg[i] if i < len(gt_seg) else NULL_CHAR
                    h = htr_seg[i] if i < len(htr_seg) else NULL_CHAR
                    confusion_by_style[style][g][h] += 1

            if issue["overlaps_step1"]:
                overlap_totals["total"] += 1
                overlap_by_style[style]["total"] += 1
                for t in issue["overlaps_step1"]:
                    overlap_by_style[style][t] += 1

    overlap_metadata = {
        "overall": dict(overlap_totals),
        "by_style": {k: dict(v) for k, v in overlap_by_style.items()},
    }

    posthoc_dir = logs_dir / "posthoc"
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    safe_write_json(overlap_metadata, posthoc_dir / "s1_s2_overlap.json")

    return confusion_by_style, overlap_metadata
