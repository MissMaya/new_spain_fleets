"""
alignment_word_level.py

Word-level alignment utilities for implementing step 2A of a hierarchical alignment process.

Steps:
    1. Tokenise GT and HTR into word (and optional punctuation) tokens
       while preserving absolute character spans.
    2. Perform deterministic dynamic-programming alignment over token sequences.
    3. Emit word-level alignment operations:
         - equal
         - replace
         - insert  (Ø → HTR word)
         - delete  (GT word → Ø)

Key properties:
- Tokenisation preserves absolute character offsets in the original text.
- Whitespace is ignored during alignment but spans remain anchored to the
  original document.
- Unicode and diacritics are preserved; no lowercasing or normalisation
  is performed unless explicitly configured.
- Alignment is deterministic:
      - Fixed DP scoring.
      - Deterministic tie-breaking (diagonal > delete > insert).
- Replacement cost includes a similarity heuristic:
      - Similar words incur low substitution cost.
      - Dissimilar words incur higher cost.
  This reduces alignment drift when structural blocks differ between GT and HTR.
- Designed to handle historical Spanish orthography without assumptions
  about modern spelling.

This module does NOT emit final Step 2 issues directly.
It produces word-level alignment operations that are later refined
by character-level alignment inside mismatched word pairs (Stage 2B).

Returns plain Python dictionaries independent of logging and overlap logic.
"""

import re
from typing import List, Dict


# ------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------

WORD_REGEX = re.compile(
    r"[^\W_]+(?:[’'\-][^\W_]+)*",  # used to recognise word boundaries in tokenisation
    re.UNICODE
)


def tokenise_with_spans(text: str) -> List[Dict]:
    """
    Tokenise text into tokens with absolute spans.

    - Words are matched with WORD_REGEX.
    - Whitespace is ignored.
    - Non-word characters are emitted as single "punct" tokens (future-proofing).
      (In your current corpus HTR/GT are expected to have minimal punctuation.)
    """
    tokens: List[Dict] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char.isspace():
            index += 1
            continue

        match = WORD_REGEX.match(text, index)

        if match:
            start = match.start()
            end = match.end()
            tokens.append({
                "text": text[start:end],
                "start": start,
                "end": end,
                "kind": "word"
            })
            index = end
        else:
            # single punctuation token
            tokens.append({
                "text": char,
                "start": index,
                "end": index + 1,
                "kind": "punct"
            })
            index += 1

    return tokens


# ------------------------------------------------------------
# Similarity heuristic
# ------------------------------------------------------------

def normalised_levenshtein_similarity(a: str, b: str) -> float:
    """
    Basic deterministic, normalised Levenshtein similarity.

    Returns a float in [0,1].
    Uses plain DP edit distance normalised by max length.
    """
    if a == b:
        return 1.0

    len_a = len(a)
    len_b = len(b)

    if len_a == 0 or len_b == 0:
        return 0.0

    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # delete
                dp[i][j - 1] + 1,        # insert
                dp[i - 1][j - 1] + cost  # substitute
            )

    distance = dp[len_a][len_b]
    max_len = max(len_a, len_b)

    return 1.0 - (distance / max_len)


# ------------------------------------------------------------
# Word-level alignment (DP)
# ------------------------------------------------------------

def align_word_sequences(
    gt_tokens: List[Dict],
    htr_tokens: List[Dict],
    similarity_threshold: float = 0.5
) -> List[Dict]:
    """
    Deterministic word-level DP alignment.

    Returns list of word operations:
        {
            "op": "equal" | "replace" | "insert" | "delete",
            "gt": token_or_None,
            "htr": token_or_None,

            # NEW (for downstream logging/diagnostics):
            "word_gt": str|None,
            "word_htr": str|None
        }

    NOTE:
    - word_gt/word_htr are lightweight string fields used by review export and
      diagnostics (e.g. top deletions/insertions).
    - gt/htr still contain the full token dicts with spans for precise mapping.
    """

    n = len(gt_tokens)
    m = len(htr_tokens)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialise base costs
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):

            gt_word = gt_tokens[i - 1]["text"]
            htr_word = htr_tokens[j - 1]["text"]

            if gt_word == htr_word:
                replace_cost = 0
            else:
                sim = normalised_levenshtein_similarity(gt_word, htr_word)
                replace_cost = 1 if sim >= similarity_threshold else 2

            dp[i][j] = min(
                dp[i - 1][j] + 1,                 # delete
                dp[i][j - 1] + 1,                 # insert
                dp[i - 1][j - 1] + replace_cost   # replace/match
            )

    # Traceback (deterministic: diag > delete > insert)
    ops: List[Dict] = []
    i = n
    j = m

    while i > 0 or j > 0:

        if i > 0 and j > 0:
            gt_word = gt_tokens[i - 1]["text"]
            htr_word = htr_tokens[j - 1]["text"]

            if gt_word == htr_word:
                replace_cost = 0
            else:
                sim = normalised_levenshtein_similarity(gt_word, htr_word)
                replace_cost = 1 if sim >= similarity_threshold else 2

            if dp[i][j] == dp[i - 1][j - 1] + replace_cost:
                if replace_cost == 0:
                    ops.append({
                        "op": "equal",
                        "gt": gt_tokens[i - 1],
                        "htr": htr_tokens[j - 1],
                        "word_gt": gt_word,
                        "word_htr": htr_word,
                    })
                else:
                    ops.append({
                        "op": "replace",
                        "gt": gt_tokens[i - 1],
                        "htr": htr_tokens[j - 1],
                        "word_gt": gt_word,
                        "word_htr": htr_word,
                    })
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            gt_word = gt_tokens[i - 1]["text"]
            ops.append({
                "op": "delete",
                "gt": gt_tokens[i - 1],
                "htr": None,
                "word_gt": gt_word,
                "word_htr": None,
            })
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            htr_word = htr_tokens[j - 1]["text"]
            ops.append({
                "op": "insert",
                "gt": None,
                "htr": htr_tokens[j - 1],
                "word_gt": None,
                "word_htr": htr_word,
            })
            j -= 1
            continue

    ops.reverse()
    return ops