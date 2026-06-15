"""
processing.py

Processing functions for the HTR cleaning pipeline.

Supports:

- Step 1: regex-based basic anomaly detection
- Step 2: full-text GT<->HTR alignment
- Step 3: linguistic / paleographic heuristics

Used to:
- Apply tagging rules
- Compute spans and line numbers
- Log detected issues
- Track overlaps across steps
- Produce posthoc metadata

Pipeline stages (run_step1.py, run_step2.py, run_step3.py) orchestrate execution.
"""

from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from utils.logging import log_issue
from utils.alignment import align_and_tag, NULL_CHAR, spans_overlap
from utils.file_io import safe_write_json, read_text, issues_json_path, load_json_if_exists
from utils.tag_rules import all_step3_tags
from utils.normalisation import normalise_pair


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def _compute_line_offsets(text: str) -> List[int]:
    """
    Return a list of absolute character offsets for the start of each line.
    """
    lines = text.splitlines(keepends = True)
    offsets: List[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)
    if not offsets:
        offsets = [0]
    return offsets


def _find_line_number(offsets: List[int], char_index: int) -> int:
    """
    Convert absolute char index to 1-based line number.
    """
    line = 1
    for i, start in enumerate(offsets):
        if start > char_index:
            break
        line = i + 1
    return line


def _offset_to_line_col(offsets: List[int], idx: int) -> tuple[int, int]:
    """
    Convert absolute char index to (line_number, col_number), both 1-based.
    """
    line_idx = 0
    for i, start in enumerate(offsets):
        if start > idx:
            break
        line_idx = i
    col = idx - offsets[line_idx]
    return line_idx + 1, col + 1


def _line_lengths_from_offsets(text: str, offsets: List[int]) -> List[int]:
    """
    Return visible line lengths corresponding to _compute_line_offsets().

    Newline characters are excluded so relative line-position describes
    position within the transcribed line rather than within the linebreak.
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return [0]
    return [len(line.rstrip("\r\n")) for line in lines]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """
    Return numerator / denominator, or None when denominator is not positive.
    """
    if denominator <= 0:
        return None
    return numerator / denominator


def _position_bin(relative_position: float | None, bins: int = 10) -> str | None:
    """
    Convert a relative position in [0, 1] to a labelled position bin.

    Examples for bins=10:
        0.00 -> "0-10%"
        0.35 -> "30-40%"
        1.00 -> "90-100%"
    """
    if relative_position is None:
        return None

    rel = max(0.0, min(1.0, float(relative_position)))
    idx = min(int(rel * bins), bins - 1)
    lo = int(idx * (100 / bins))
    hi = int((idx + 1) * (100 / bins))
    return f"{lo}-{hi}%"


def _add_position_metadata(
    issue: Dict[str, Any],
    text: str,
    line_offsets: List[int],
    line_lengths: List[int],
    *,
    position_basis: str,
    bins: int = 10,
) -> Dict[str, Any]:
    """
    Add relative line/document position metadata to an issue.

    The current pipeline records Step 1, Step 2 and Step 3 issue positions
    against the HTR text. For Step 2 deletions, the position should therefore
    be interpreted as the HTR/alignment anchor where missing GT material is
    detected rather than as the exact GT-side character coordinate.

    Added fields:
        - line_length
        - document_length
        - relative_line_position
        - relative_document_position
        - line_position_bin
        - document_position_bin
        - position_basis
    """
    document_length = len(text)

    abs_start = int(issue.get("_abs_start", 0) or 0)
    abs_end = int(issue.get("_abs_end", abs_start) or abs_start)

    # Use the midpoint of the span where possible. For zero-length spans
    # such as insertions/deletions, use the anchor position.
    if abs_end > abs_start:
        abs_mid = abs_start + ((abs_end - abs_start) / 2.0)
    else:
        abs_mid = float(abs_start)

    line = int(issue.get("line", 1) or 1)
    line_idx = max(0, min(line - 1, len(line_offsets) - 1))
    line_length = line_lengths[line_idx] if line_idx < len(line_lengths) else 0

    char_start = int(issue.get("char_start", 1) or 1)
    char_end = int(issue.get("char_end", char_start) or char_start)

    if char_end > char_start:
        char_mid = char_start + ((char_end - char_start) / 2.0)
    else:
        char_mid = float(char_start)

    # Convert 1-based character coordinate to a centre-ish relative coordinate.
    relative_line_position = _safe_ratio(max(char_mid - 0.5, 0.0), line_length)
    relative_document_position = _safe_ratio(max(abs_mid, 0.0), document_length)

    if relative_line_position is not None:
        relative_line_position = max(0.0, min(1.0, relative_line_position))
    if relative_document_position is not None:
        relative_document_position = max(0.0, min(1.0, relative_document_position))

    issue["line_length"] = line_length
    issue["document_length"] = document_length
    issue["relative_line_position"] = relative_line_position
    issue["relative_document_position"] = relative_document_position
    issue["line_position_bin"] = _position_bin(relative_line_position, bins=bins)
    issue["document_position_bin"] = _position_bin(relative_document_position, bins=bins)
    issue["position_basis"] = position_basis

    return issue


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
    """
    Step 1: apply rule-based regex tags to HTR text.
    """

    error_counts_by_style = {style: defaultdict(int) for style in calligraphy_types}
    step1_spans_by_file = defaultdict(list)

    for pair in train_pairs:
        style = pair["style"]
        doc_id = pair["id"]
        htr_path = Path(pair["htr_path"])

        text = read_text(htr_path)
        line_offsets = _compute_line_offsets(text)
        line_lengths = _line_lengths_from_offsets(text, line_offsets)

        issues = []

        for code, regex in step1_tags.items():
            for match in regex.finditer(text):

                start = match.start()
                end = match.end()

                tag = f"S1{code}"

                line, char_start = _offset_to_line_col(line_offsets, start)
                _, char_end = _offset_to_line_col(line_offsets, max(end - 1, start))

                snippet = text[start:end]

                issue = {
                    "tag": tag,
                    "description": tag_schema["S1"][code],
                    "line": line,
                    "char_start": char_start,
                    "char_end": char_end,
                    "htr_text": snippet,
                    "gt_text": None,
                    "overlaps_step1": [],
                    "overlaps_step2": [],
                    "review": {"status": "unreviewed"},
                    "_abs_start": start,
                    "_abs_end": end,
                }

                _add_position_metadata(
                    issue,
                    text,
                    line_offsets,
                    line_lengths,
                    position_basis="htr",
                )

                issues.append(issue)

                step1_spans_by_file[doc_id].append(
                    {"tag": tag, "start": start, "end": end}
                )

                error_counts_by_style[style][tag] += 1

        doc_dir = logs_dir / style / doc_id
        doc_dir.mkdir(parents = True, exist_ok = True)
        safe_write_json(issues, issues_json_path(doc_dir, doc_id))

    return error_counts_by_style, step1_spans_by_file


# ----------------------------------------------------------------------
# STEP 2
# ----------------------------------------------------------------------

def process_step2_issues(
    train_pairs: List[Dict],
    step1_spans_by_file: Dict[str, List[Dict]],
    tag_schema: Dict,
    logs_dir: Path,
):
    """
    Step 2: align GT<->HTR and tag with Step 2 tags.
    """

    confusion_by_style = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    overlap_totals = defaultdict(int)
    overlap_by_style = defaultdict(lambda: defaultdict(int))

    step2_spans_by_file = defaultdict(list)

    for pair in train_pairs:

        style = pair["style"]
        doc_id = pair["id"]

        gt_text_full = read_text(pair["gt_path"])
        htr_text_full = read_text(pair["htr_path"])

        step1_spans = step1_spans_by_file.get(doc_id, [])

        issues = align_and_tag(gt_text_full, htr_text_full, step1_spans)

        line_offsets = _compute_line_offsets(htr_text_full)
        line_lengths = _line_lengths_from_offsets(htr_text_full, line_offsets)

        doc_issues = []

        for issue in issues:

            gt_seg = issue.get("gt") or ""
            htr_seg = issue.get("htr") or ""

            raw_tag = issue["tag"]
            full_tag = f"S2{raw_tag}"

            issue["tag"] = full_tag
            issue["description"] = tag_schema["S2"][raw_tag]

            start = issue.pop("start")
            end = issue.pop("end")

            issue["_abs_start"] = start
            issue["_abs_end"] = end

            line = _find_line_number(line_offsets, start)
            issue["line"] = line

            issue["char_start"] = start - line_offsets[line - 1] + 1
            issue["char_end"] = max(issue["char_start"], end - line_offsets[line - 1])

            issue["htr_text"] = issue.pop("htr")
            issue["gt_text"] = issue.pop("gt")

            issue.setdefault("overlaps_step1", [])
            issue["overlaps_step2"] = []

            issue["review"] = {"status": "unreviewed"}

            position_basis = "htr_alignment_anchor"
            if full_tag == "S2X":
                position_basis = "htr_substitution_span"
            elif full_tag == "S2I":
                position_basis = "htr_insertion_span"
            elif full_tag == "S2D":
                position_basis = "htr_deletion_anchor"

            _add_position_metadata(
                issue,
                htr_text_full,
                line_offsets,
                line_lengths,
                position_basis=position_basis,
            )

            doc_issues.append(issue)

            step2_spans_by_file[doc_id].append(
                {"tag": full_tag, "start": start, "end": end}
            )

            gt_seg = issue["gt_text"] if issue["gt_text"] else NULL_CHAR
            htr_seg = issue["htr_text"] if issue["htr_text"] else NULL_CHAR

            max_len = max(len(gt_seg), len(htr_seg))

            for i in range(max_len):

                g_raw = gt_seg[i] if i < len(gt_seg) else NULL_CHAR
                h_raw = htr_seg[i] if i < len(htr_seg) else NULL_CHAR

                g, h = normalise_pair(g_raw, h_raw, lowercase = False)

                confusion_by_style[style][g][h] += 1

            if issue["overlaps_step1"]:
                overlap_totals["total"] += 1
                overlap_by_style[style]["total"] += 1
                for t in issue["overlaps_step1"]:
                    overlap_by_style[style][t] += 1

        doc_dir = logs_dir / style / doc_id
        doc_dir.mkdir(parents = True, exist_ok = True)

        issues_path = issues_json_path(doc_dir, doc_id)

        # Load existing Step-1 issues
        existing_issues = load_json_if_exists(issues_path, [])

        # Append Step-2 issues
        existing_issues.extend(doc_issues)

        # Write combined issues
        safe_write_json(existing_issues, issues_path)
        
    overlap_metadata = {
        "overall": dict(overlap_totals),
        "by_style": {k: dict(v) for k, v in overlap_by_style.items()},
    }

    posthoc_dir = logs_dir / "posthoc"
    posthoc_dir.mkdir(parents = True, exist_ok = True)

    safe_write_json(overlap_metadata, posthoc_dir / "s1_s2_overlap.json")

    return confusion_by_style, overlap_metadata, step2_spans_by_file


# ----------------------------------------------------------------------
# STEP 3
# ----------------------------------------------------------------------

def process_step3_issues(
    train_pairs: List[Dict],
    step1_spans_by_file: Dict[str, List[Dict]],
    step2_spans_by_file: Dict[str, List[Dict]],
    tag_schema: Dict,
    logs_dir: Path,
):
    """
    Step 3: apply linguistic/paleographic rules to HTR and compute overlaps with Steps 1/2.

    Returns:
      - error_counts_by_style
    """

    error_counts_by_style = defaultdict(lambda: defaultdict(int))
    s1_s3_overlap = defaultdict(lambda: defaultdict(int))
    s2_s3_overlap = defaultdict(lambda: defaultdict(int))

    for pair in train_pairs:
        style = pair["style"]
        doc_id = pair["id"]
        htr_text = read_text(pair["htr_path"])

        step1_spans = step1_spans_by_file.get(doc_id, [])
        step2_spans = step2_spans_by_file.get(doc_id, [])

        line_offsets = _compute_line_offsets(htr_text)
        line_lengths = _line_lengths_from_offsets(htr_text, line_offsets)

        doc_issues = []

        for raw_tag, regex in all_step3_tags.items():
            for match in regex.finditer(htr_text):
                full_tag = f"S3{raw_tag}"
                description = tag_schema["S3"][raw_tag]

                start = match.start()
                end = match.end()

                line, char_start = _offset_to_line_col(line_offsets, start)
                _, char_end = _offset_to_line_col(line_offsets, max(end - 1, start))

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
                    "review": {"status": "unreviewed"},
                    "_abs_start": start,
                    "_abs_end": end,
                }

                _add_position_metadata(
                    issue,
                    htr_text,
                    line_offsets,
                    line_lengths,
                    position_basis="htr",
                )

                doc_issues.append(issue)

                error_counts_by_style[style][full_tag] += 1

                for t in overlapping_s1:
                    s1_s3_overlap[style][t] += 1

                for t in overlapping_s2:
                    s2_s3_overlap[style][t] += 1

        # --------------------------------------------------------------
        # Write once per document
        # --------------------------------------------------------------

        doc_dir = logs_dir / style / doc_id
        doc_dir.mkdir(parents = True, exist_ok = True)

        # Step 1 + Step 2 issues should already be in the file; append Step 3 issues
        existing_issues = load_json_if_exists(issues_json_path(doc_dir, doc_id), [])
        combined = existing_issues + doc_issues

        safe_write_json(combined, issues_json_path(doc_dir, doc_id))

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