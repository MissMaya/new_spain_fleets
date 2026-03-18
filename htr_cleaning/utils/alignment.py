"""
alignment.py

Contains utilities to achieve hierarchical alignment for step 2 of the pipeline. 

Previously, we used full-document, character-level alignment
(with difflib) but this gave meaningless results. The new method of aligning uses
tokenisation to align at word-level before using character-level alignment on mismatched
words only.

Step 2A — Word-Level Alignment
    - Tokenise GT and HTR into token sequences,
      preserving absolute character spans.
    - Align token sequences.
    - Detect word-level operations: equal / replace / insert / delete.

Step 2B — Character-Level Alignment (Within Word Substitutions)
    - For each word-level replacement, perform character-level
      alignment. Crucially, this operation now happens 
      inside the mismatched word pair only.
    - Tag issues as follows:
        - X : substitution (GT → HTR)
        - I : insertion (Ø → HTR)
        - D : deletion (GT → Ø)
"""

from typing import List, Dict, Tuple, Optional

from .alignment_word_level import tokenise_with_spans, align_word_sequences
from .alignment_char_level import align_chars, char_ops_to_span_issues


NULL_CHAR = "Ø"


# ----------------------------------------------------------------------
# Helper functions
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


def _find_deletion_anchor(
    word_ops: List[Dict],
    idx: int,
    last_htr_end: Optional[int],
    htr_text_len: int) -> int:
    """
    For deletions, work out an HTR anchor position. Preference:
      1) next available HTR token start
      2) last seen HTR token end
      3) 0
    """
    # Look ahead for next op with HTR token
    for j in range(idx + 1, len(word_ops)):
        htr_tok = word_ops[j].get("htr")
        if htr_tok is not None:
            return int(htr_tok["start"])

    if last_htr_end is not None:
        return int(last_htr_end)

    return 0


def _compute_step1_overlaps(
    start: int,
    end: int,
    step1_spans: List[Dict]
) -> List[str]:
    """
    Compute Step 1 overlap tags for a Step 2 issue span.
    For spans of length 0 (deletions), treat as [start, start+1).
    """
    overlaps = []
    a_end = end if end > start else start + 1

    for s1 in step1_spans:
        s1_start = s1["start"]
        s1_end = s1["end"]
        if spans_overlap(start, a_end, s1_start, s1_end):
            overlaps.append(s1["tag"])

    return sorted(set(overlaps))


# ----------------------------------------------------------------------
# Hierarchical alignment
# ----------------------------------------------------------------------

def align_and_tag_hierarchical(
    gt_text: str,
    htr_text: str,
    step1_spans: List[Dict],
    similarity_threshold: float = 0.5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Hierarchical Step 2 alignment.

    Returns:
      issues: list of span issues (using Step 2 tags X/I/D), anchored in HTR absolute offsets
      word_ops: the Stage 2A word-level operation stream (for stats/debug)
    """
    gt_tokens = tokenise_with_spans(gt_text)
    htr_tokens = tokenise_with_spans(htr_text)

    word_ops = align_word_sequences(gt_tokens, htr_tokens)

    htr_line_offsets = compute_line_offsets(htr_text)
    issues: List[Dict] = []

    last_htr_end: Optional[int] = None
    htr_len = len(htr_text)

    for idx, op in enumerate(word_ops):
        op_type = op["op"]
        gt_tok = op.get("gt")
        htr_tok = op.get("htr")

        if htr_tok is not None:
            last_htr_end = int(htr_tok["end"])

        if op_type == "equal":
            continue

        # --------------------------
        # Word insertion (Ø → HTR)
        # --------------------------
        if op_type == "insert" and htr_tok is not None:
            start = int(htr_tok["start"])
            end = int(htr_tok["end"])
            line_num = find_line_number(htr_line_offsets, start)
            overlaps_step1 = _compute_step1_overlaps(start, end, step1_spans)

            issues.append({
                "tag": "I",
                "start": start,
                "end": end,
                "gt": "",
                "htr": htr_text[start:end],
                "line": line_num,
                "overlaps_step1": overlaps_step1,

                # metadata
                "word_op": "insert",
                "word_gt": None,
                "word_htr": htr_tok["text"],
                "word_htr_span": [start, end],
                "word_gt_span": None,
            })
            continue

        # --------------------------
        # Word deletion (GT → Ø)
        # --------------------------
        if op_type == "delete" and gt_tok is not None:
            anchor = _find_deletion_anchor(word_ops, idx, last_htr_end, htr_len)
            start = int(anchor)
            end = int(anchor)  # point anchor
            line_num = find_line_number(htr_line_offsets, start)
            overlaps_step1 = _compute_step1_overlaps(start, end, step1_spans)

            issues.append({
                "tag": "D",
                "start": start,
                "end": end,
                "gt": gt_tok["text"],
                "htr": "",
                "line": line_num,
                "overlaps_step1": overlaps_step1,

                # metadata
                "word_op": "delete",
                "word_gt": gt_tok["text"],
                "word_htr": None,
                "word_gt_span": [int(gt_tok["start"]), int(gt_tok["end"])],
                "word_htr_span": None,
            })
            continue

        # --------------------------
        # Word substitution: run char-level alignment inside word pair only
        # --------------------------
        if op_type == "replace" and gt_tok is not None and htr_tok is not None:
            gt_word = gt_tok["text"]
            htr_word = htr_tok["text"]

            htr_word_abs_start = int(htr_tok["start"])
            htr_word_abs_end = int(htr_tok["end"])

            char_ops = align_chars(gt_word, htr_word)
            span_issues = char_ops_to_span_issues(
                char_ops = char_ops,
                htr_abs_start = htr_word_abs_start,
                gt_word = gt_word,
                htr_word = htr_word,
            )

            # Add additional details to span issues
            for si in span_issues:
                start = int(si["start"])
                end = int(si["end"])
                line_num = find_line_number(htr_line_offsets, start)
                overlaps_step1 = _compute_step1_overlaps(start, end, step1_spans)

                issues.append({
                    "tag": si["tag"],  # X/I/D
                    "start": start,
                    "end": end,
                    "gt": si.get("gt", ""),
                    "htr": si.get("htr", ""),
                    "line": line_num,
                    "overlaps_step1": overlaps_step1,

                    # metadata
                    "word_op": "replace",
                    "word_gt": gt_word,
                    "word_htr": htr_word,
                    "word_gt_span": [int(gt_tok["start"]), int(gt_tok["end"])],
                    "word_htr_span": [htr_word_abs_start, htr_word_abs_end],
                })

            continue

        continue

    return issues, word_ops


# ----------------------------------------------------------------------
# Make backwards-compatible
# ----------------------------------------------------------------------

def align_and_tag(
    gt_text: str,
    htr_text: str,
    step1_spans: List[Dict],
):
    """
    Returns only the Step 2 span issues list (tag X/I/D) as expected
    by the initial processing code.
    """
    issues, _word_ops = align_and_tag_hierarchical(gt_text, htr_text, step1_spans)
    return issues