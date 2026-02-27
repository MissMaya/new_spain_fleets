"""
alignment_char_level.py

Character-level alignment utilities for Step 2B of the hierarchical Step 2 refactor.

This module performs deterministic dynamic-programming alignment between two strings
(GT word vs HTR word) and emits character-level edit operations:

- equal   (GT char == HTR char)
- replace (GT char -> HTR char)
- insert  (Ø -> HTR char)
- delete  (GT char -> Ø)

It also converts the op stream into span-based issues anchored in the HTR document
(using an absolute HTR offset for the start of the word span).

Outputs produced here are "raw Step 2 issues" (tag X/I/D, start/end, gt/htr segment),
which are later wrapped by processing.py into the canonical issue schema.
"""

from typing import List, Dict, Tuple, Optional


NULL_CHAR = "Ø"


# ----------------------------------------------------------------------
# DP alignment at character-level
# ----------------------------------------------------------------------

def align_chars(gt: str, htr: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Deterministic character-level alignment using edit-distance DP.

    Returns a list of operations of form:
        (op, gt_char_or_None, htr_char_or_None)

    op is one of: "equal", "replace", "insert", "delete"

    Deterministic tie-break (when costs are equal):
      diagonal (equal/replace) > delete > insert

    Costs:
      equal:   0
      replace: 1
      insert:  1
      delete:  1
    """
    n = len(gt)
    m = len(htr)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # base cases
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # fill
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = 0 if gt[i - 1] == htr[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,           # delete
                dp[i][j - 1] + 1,           # insert
                dp[i - 1][j - 1] + cost_sub # sub/equal
            )

    # traceback (deterministic: diag > delete > insert)
    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m

    while i > 0 or j > 0:
        # diagonal preferred
        if i > 0 and j > 0:
            cost_sub = 0 if gt[i - 1] == htr[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost_sub:
                if cost_sub == 0:
                    ops.append(("equal", gt[i - 1], htr[j - 1]))
                else:
                    ops.append(("replace", gt[i - 1], htr[j - 1]))
                i -= 1
                j -= 1
                continue

        # delete next
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", gt[i - 1], None))
            i -= 1
            continue

        # insert last
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("insert", None, htr[j - 1]))
            j -= 1
            continue

        # Should never happen, but just in case:
        raise RuntimeError("DP traceback failed (no valid predecessor found).")

    ops.reverse()
    return ops


# ----------------------------------------------------------------------
# Convert ops -> span issues (anchored in HTR absolute offsets)
# ----------------------------------------------------------------------

def char_ops_to_span_issues(
    char_ops: List[Tuple[str, Optional[str], Optional[str]]],
    htr_abs_start: int,
    gt_word: str,
    htr_word: str,
) -> List[Dict]:
    """
    Convert a character op stream into span-based issues anchored in HTR absolute offsets.

    Returns list of dicts each with:
      - tag: "X" | "I" | "D"
      - start: abs HTR start offset
      - end: abs HTR end offset (end may == start for deletions)
      - gt: substring involved (or "")
      - htr: substring involved (or "")

    Grouping strategy:
      - Consecutive ops of the same issue type are merged into a single span issue.
      - Spans are computed in HTR space:
          insert/replace consume HTR characters -> span expands
          delete consumes no HTR char -> anchored at current HTR cursor (point)
    """
    issues: List[Dict] = []

    # cursors within the word strings
    gt_i = 0
    htr_i = 0

    # active aggregation
    active_tag: Optional[str] = None
    active_start: Optional[int] = None
    active_end: Optional[int] = None
    active_gt_parts: List[str] = []
    active_htr_parts: List[str] = []

    def flush():
        nonlocal active_tag, active_start, active_end, active_gt_parts, active_htr_parts
        if active_tag is None:
            return
        issues.append({
            "tag": active_tag,
            "start": int(active_start) if active_start is not None else int(htr_abs_start + htr_i),
            "end": int(active_end) if active_end is not None else int(htr_abs_start + htr_i),
            "gt": "".join(active_gt_parts),
            "htr": "".join(active_htr_parts),
        })
        active_tag = None
        active_start = None
        active_end = None
        active_gt_parts = []
        active_htr_parts = []

    def begin(tag: str, start: int, end: int):
        nonlocal active_tag, active_start, active_end
        active_tag = tag
        active_start = start
        active_end = end

    for op, gch, hch in char_ops:
        if op == "equal":
            flush()
            gt_i += 1 if gch is not None else 0
            htr_i += 1 if hch is not None else 0
            continue

        if op == "replace":
            # substitution -> X
            abs_pos = htr_abs_start + htr_i
            tag = "X"
            if active_tag != tag:
                flush()
                begin(tag, abs_pos, abs_pos + 1)
            else:
                # extend by 1 htr char
                active_end = (active_end or abs_pos) + 1

            active_gt_parts.append(gch or "")
            active_htr_parts.append(hch or "")

            gt_i += 1
            htr_i += 1
            continue

        if op == "insert":
            # insertion -> I (consumes HTR)
            abs_pos = htr_abs_start + htr_i
            tag = "I"
            if active_tag != tag:
                flush()
                begin(tag, abs_pos, abs_pos + 1)
            else:
                active_end = (active_end or abs_pos) + 1

            active_gt_parts.append("")  # Ø
            active_htr_parts.append(hch or "")

            htr_i += 1
            continue

        if op == "delete":
            # deletion -> D (consumes GT, no HTR advance)
            abs_pos = htr_abs_start + htr_i
            tag = "D"
            if active_tag != tag:
                flush()
                begin(tag, abs_pos, abs_pos)  # point span
            # do not extend end for deletions (still point-anchored)

            active_gt_parts.append(gch or "")
            active_htr_parts.append("")  # Ø

            gt_i += 1
            continue

        raise ValueError(f"Unknown op: {op}")

    flush()
    return issues