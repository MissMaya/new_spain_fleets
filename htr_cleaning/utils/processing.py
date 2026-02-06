"""
processing.py

Core processing functions for the HTR cleaning pipeline.

Supports:

- Step 1: regex-based surface anomaly detection
- Step 2: full-text GT↔HTR alignment
- Step 3: linguistic / paleographic heuristics

Responsibilities:
- Apply tagging rules
- Compute spans and line numbers
- Log detected issues
- Track overlaps across steps
- Produce posthoc metadata

Pipeline stages (run_step1.py, run_step2.py, run_step3.py) orchestrate execution.
"""

from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from utils.logging import log_issue
from utils.alignment import align_and_tag, NULL_CHAR, spans_overlap
from utils.file_io import safe_write_json, read_text
from utils.tag_rules import all_step3_tags


# ----------------------------------------------------------------------
# Shared helpers
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
    train_pairs,
    step1_tags,
    tag_schema,
    calligraphy_types,
    logs_dir,
):

    from collections import defaultdict
    from pathlib import Path
    from utils.file_io import safe_write_json, read_text

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def compute_line_offsets(text: str):
        offsets = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                offsets.append(i + 1)
        return offsets

    def offset_to_line_col(offsets, idx):
        line = 0
        for i, start in enumerate(offsets):
            if start > idx:
                break
            line = i
        col = idx - offsets[line]
        return line + 1, col + 1

    # ------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------

    error_counts_by_style = {
        style: defaultdict(int) for style in calligraphy_types
    }

    step1_spans_by_file = defaultdict(list)

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------

    for pair in train_pairs:
        style = pair["style"]
        doc_id = pair["id"]
        htr_path = Path(pair["htr_path"])

        text = read_text(htr_path)
        line_offsets = compute_line_offsets(text)

        issues = []

        # step1_tags is: { "G": regex, ... }
        for code, regex in step1_tags.items():
            for match in regex.finditer(text):

                start = match.start()
                end = match.end()

                tag = f"S1{code}"

                line, char_start = offset_to_line_col(line_offsets, start)
                _, char_end = offset_to_line_col(line_offsets, max(end - 1, start))

                snippet = text[start:end]

                issue = {
                    "tag": tag,
                    "line": line,
                    "char_start": char_start,
                    "char_end": char_end,
                    "htr_text": snippet,
                    "gt_text": None,
                    "_abs_start": start,
                    "_abs_end": end,
                }

                issues.append(issue)

                step1_spans_by_file[doc_id].append(
                    {
                        "tag": tag,
                        "start": start,
                        "end": end,
                    }
                )

                error_counts_by_style[style][tag] += 1

        # --------------------------------------------------------
        # Write per-document issues
        # --------------------------------------------------------

        doc_dir = logs_dir / style / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        clean = [
            {k: v for k, v in issue.items() if not k.startswith("_")}
            for issue in issues
        ]

        safe_write_json(clean, doc_dir / "issues.json")

    return error_counts_by_style, step1_spans_by_file

# ----------------------------------------------------------------------
# STEP 2 (unchanged)
# ----------------------------------------------------------------------

def process_step2_issues(
    train_pairs: List[Dict],
    step1_spans_by_file: Dict[str, List[Dict]],
    tag_schema: Dict,
    logs_dir: Path,
):

    confusion_by_style = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    overlap_totals = defaultdict(int)
    overlap_by_style = defaultdict(lambda: defaultdict(int))

    step2_spans_by_file = defaultdict(list)

    for pair in train_pairs:
        style = pair["style"]
        doc_id = pair["id"]

        gt_text = read_text(pair["gt_path"])
        htr_text = read_text(pair["htr_path"])

        step1_spans = step1_spans_by_file.get(doc_id, [])

        issues = align_and_tag(gt_text, htr_text, step1_spans)

        for issue in issues:

            raw_tag = issue["tag"]     
            full_tag = f"S2{raw_tag}"
            issue["tag"] = full_tag
            issue["description"] = tag_schema["S2"][raw_tag]

            log_issue(
                logs_dir=logs_dir,
                calligraphy_type=style,
                document_id=doc_id,
                issue=issue,
            )

            step2_spans_by_file[doc_id].append(issue)

            gt_seg = issue["gt"] if issue["gt"] else NULL_CHAR
            htr_seg = issue["htr"] if issue["htr"] else NULL_CHAR

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

    return confusion_by_style, overlap_metadata, step2_spans_by_file


# ----------------------------------------------------------------------
# STEP 3 (unchanged)
# ----------------------------------------------------------------------

def process_step3_issues(
    train_pairs: List[Dict],
    step1_spans_by_file: Dict[str, List[Dict]],
    step2_spans_by_file: Dict[str, List[Dict]],
    tag_schema: Dict,
    logs_dir: Path,
):

    error_counts_by_style = defaultdict(lambda: defaultdict(int))
    s1_s3_overlap = defaultdict(lambda: defaultdict(int))
    s2_s3_overlap = defaultdict(lambda: defaultdict(int))

    # Helper for line/column (1-based)
    def offset_to_line_col(offsets, idx):
        line = 0
        for i, start in enumerate(offsets):
            if start > idx:
                break
            line = i
        col = idx - offsets[line]
        return line + 1, col + 1

    for pair in train_pairs:
        style = pair["style"]
        doc_id = pair["id"]
        htr_text = read_text(pair["htr_path"])

        step1_spans = step1_spans_by_file.get(doc_id, [])
        step2_spans = step2_spans_by_file.get(doc_id, [])

        line_offsets = _compute_line_offsets(htr_text)

        for raw_tag, regex in all_step3_tags.items():
            for match in regex.finditer(htr_text):

                full_tag = f"S3{raw_tag}"
                description = tag_schema["S3"][raw_tag]

                start = match.start()
                end = match.end()

                line, char_start = offset_to_line_col(line_offsets, start)
                _, char_end = offset_to_line_col(line_offsets, max(end - 1, start))

                snippet = htr_text[start:end]

                # Overlaps with Step 1
                overlapping_s1 = [
                    s1["tag"]
                    for s1 in step1_spans
                    if spans_overlap(start, end, s1["start"], s1["end"])
                ]

                # Overlaps with Step 2
                overlapping_s2 = [
                    s2["tag"]
                    for s2 in step2_spans
                    if spans_overlap(
                        start,
                        end if end > start else start + 1,
                        s2["start"],
                        s2["end"],
                    )
                ]

                issue = {
                    "tag": full_tag,
                    "description": description,
                    "line": line,
                    "char_start": char_start,
                    "char_end": char_end,
                    "htr_text": snippet,
                    "gt_text": None,
                    "overlaps_step1": sorted(set(overlapping_s1)),
                    "overlaps_step2": sorted(set(overlapping_s2)),
                    "_abs_start": start,
                    "_abs_end": end,
                }

                log_issue(
                    logs_dir=logs_dir,
                    calligraphy_type=style,
                    document_id=doc_id,
                    issue=issue,
                )

                error_counts_by_style[style][full_tag] += 1

                for t in overlapping_s1:
                    s1_s3_overlap[style][t] += 1

                for t in overlapping_s2:
                    s2_s3_overlap[style][t] += 1

    posthoc_dir = logs_dir / "posthoc"
    posthoc_dir.mkdir(parents=True, exist_ok=True)

    safe_write_json(
        {k: dict(v) for k, v in s1_s3_overlap.items()},
        posthoc_dir / "s1_s3_overlap.json",
    )

    safe_write_json(
        {k: dict(v) for k, v in s2_s3_overlap.items()},
        posthoc_dir / "s2_s3_overlap.json",
    )

    return error_counts_by_style