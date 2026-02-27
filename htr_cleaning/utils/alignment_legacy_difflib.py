"""
alignment.py (legacy)

Utilities to achieve character-level alignment for Step 2 of the HTR cleaning pipeline.

Aligns GT and HTR documents using difflib and tags:

- X : substitution (GT → HTR)
- I : insertion (Ø → HTR)
- D : deletion (GT → Ø)

Key properties:

- Alignment is performed on full documents (not line-by-line) to allow for
  line breaks to differ between GT and HTR. I've seen differing line breaks when 
  comparing multiple GT and HTR files so I know that this will be a common feauture 
  in this dataset.
- All issues are reported as character spans.
- Deletions are anchored at an insertion point.
- Line numbers are reconstructed after alignment.
- Step 1 / Step 2 coupling is computed via span overlap.

Issues are returned as Python dictionaries suitable for JSON logging.
"""

import difflib
from typing import List, Dict, Tuple


NULL_CHAR = "Ø"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def compute_line_offsets(text: str) -> List[int]:
    """
    Compute global character offsets for the start of each line.

    Returns a list where index i contains the 0-based offset of line i.
    """
    lines = text.splitlines(keepends = True)
    offsets = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)
    return offsets


def find_line_number(offsets: List[int], char_index: int) -> int:
    """
    Given line start offsets and a character index, return 1-based line number.
    """
    line = 1
    for i, start in enumerate(offsets):
        if start > char_index:
            break
        line = i + 1
    return line


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """
    Return True if two half-open spans [a_start, a_end) and [b_start, b_end) overlap.
    """
    return max(a_start, b_start) < min(a_end, b_end)


# ----------------------------------------------------------------------
# Core alignment
# ----------------------------------------------------------------------

def align_and_tag(
    gt_text: str,
    htr_text: str,
    step1_spans: List[Dict],
):
    """
    Align GT and HTR full texts and produce Step 2 span-based issues.

    Parameters
    ----------
    gt_text : str
        Ground truth document (full text).

    htr_text : str
        HTR document (full text).

    step1_spans : list of dict Step 1 issues for this file. Each dict must contain:
            - start (global HTR offset)
            - end (global HTR offset)
            - tag

    Returns
    -------
    issues : list of dict

        Each issue has:

        - tag : "X" (substitution), "I" (insertion), or "D" (deletion)
        - start : global HTR start offset
        - end : global HTR end offset (may be the same as start in the case of deletions)
        - gt : GT substring (or "")
        - htr : HTR substring (or "")
        - line : 1-based line number (in HTR)
        - overlaps_step1 : list of Step 1 tags whose spans overlap this issue
    """

    matcher = difflib.SequenceMatcher(a = gt_text, b = htr_text)
    opcodes = matcher.get_opcodes()

    htr_line_offsets = compute_line_offsets(htr_text)

    issues = []

    for tag, i1, i2, j1, j2 in opcodes:

        if tag == "equal":
            continue

        # GT and HTR substrings
        gt_seg = gt_text[i1:i2]
        htr_seg = htr_text[j1:j2]

        if tag == "replace":
            issue_tag = "X"
            start = j1
            end = j2

        elif tag == "insert":
            issue_tag = "I"
            start = j1
            end = j2

        elif tag == "delete":
            issue_tag = "D"
            # Deletions have no HTR span so we just anchor at insertion point
            start = j1
            end = j1

        else:
            continue

        line_num = find_line_number(htr_line_offsets, start)

        # --------------------------------------------------------------
        # Step 1 overlap detection (span-based)
        # --------------------------------------------------------------

        overlapping_step1 = []

        for s1 in step1_spans:
            s1_start = s1["start"]
            s1_end = s1["end"]

            if spans_overlap(start, end if end > start else start + 1, s1_start, s1_end):
                overlapping_step1.append(s1["tag"])

        issue = {
            "tag": issue_tag,
            "start": start,
            "end": end,
            "gt": gt_seg if issue_tag != "I" else "",
            "htr": htr_seg if issue_tag != "D" else "",
            "line": line_num,
            "overlaps_step1": sorted(set(overlapping_step1)),
        }

        issues.append(issue)

    return issues
