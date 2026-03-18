"""
alignment_word_level.py

Robust GT-anchored word alignment for Step 2A (fast + stable).

Core idea (GT is the source of truth)
-------------------------------------
We do NOT try to globally align two arbitrary documents.

Instead we:

1) Tokenise GT and HTR into word tokens with absolute spans.
2) Walk GT left-to-right (GT is authoritative).
3) For each GT token, search a *nearby* window of HTR tokens (starting from the
   current HTR cursor) and pick the best fuzzy match.
4) Allow the best match to be:
      - 1 HTR token  (normal case)
      - an HTR n-gram (1..MAX_NGRAM tokens) to handle cases like:
            GT: "Rey"   -> HTR: "Remio Y Pronuno"  (or vice versa)
5) Emit a word-op stream:
      - equal   (exact same original token)
      - replace (matched but not identical)
      - delete  (no acceptable match found for GT token)
      - insert  (HTR tokens skipped/leftover)

Important: Normalisation is used ONLY for matching
-------------------------------------------------
We match using normalised tokens (lowercase, strip diacritics) to keep alignment stable.
But we emit ops with ORIGINAL token strings so Step 2B can still log paleographic
confusions like "ç -> c" inside a word.

Performance
-----------
This is designed to be fast on ~1900 docs:

- Greedy left-to-right (no O(n*m) full DP)
- Window-limited matching
- RapidFuzz scoring
- Optional stopwords + numeric handling

Downstream expectations
-----------------------
- tokenise_with_spans(text) -> List[Dict] with keys: text,start,end,kind
- align_word_sequences(gt_tokens, htr_tokens, similarity_threshold=...) -> List[Dict]
  Each op dict contains:
      {
        "op": "equal"|"replace"|"insert"|"delete",
        "gt": token_or_None,
        "htr": token_or_None,          # for multi-token matches, first token is here
        "word_gt": str|None,
        "word_htr": str|None,          # may be multi-token string for multi matches
        "htr_span_tokens": List[token] # empty for delete, 1+ for replace/equal
        "match_score": int|None
      }

Multi-token matches
-------------------
If a GT token best-matches an HTR span of k>1 tokens:
- We emit one "replace" (or "equal") op where "htr" is the FIRST token,
  and "word_htr" is the JOINED span.
- Then we emit (k-1) "insert" ops for the remaining tokens in the span,
  marked with metadata: {"part_of_multiword_match": True, ...}

This keeps compatibility with existing downstream code that expects
single-token htr per op, while still preserving the multi-token mapping.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz


# ---------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------

WORD_REGEX = re.compile(
    r"[^\W_]+(?:[’'\-][^\W_]+)*",
    re.UNICODE
)


def tokenise_with_spans(text: str) -> List[Dict]:
    """
    Tokenise into word tokens (and minimal punct tokens) with absolute spans.

    Returns list of:
        {"text": str, "start": int, "end": int, "kind": "word"|"punct"}
    """
    tokens: List[Dict] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        m = WORD_REGEX.match(text, i)
        if m:
            start, end = m.start(), m.end()
            tokens.append({"text": text[start:end], "start": start, "end": end, "kind": "word"})
            i = end
        else:
            # punctuation / stray char
            tokens.append({"text": ch, "start": i, "end": i + 1, "kind": "punct"})
            i += 1

    return tokens


# ---------------------------------------------------------------------
# Normalisation (for matching only)
# ---------------------------------------------------------------------

def _strip_diacritics(s: str) -> str:
    # NFD splits accent marks into combining chars; we drop them.
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def normalise_for_match(s: str) -> str:
    """
    Normalise token for matching:
    - lowercase
    - strip diacritics (á->a, ç->c, ñ->n)
    - keep digits/letters
    - remove stray punctuation inside token (very conservative)
    """
    s = s.lower()
    s = _strip_diacritics(s)
    # remove characters that are not letters/digits (keep simple)
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s


def _is_probably_numeric(tok: str) -> bool:
    # treat "1621" or "580" as numeric; also allow roman-like? keep simple for now
    return tok.isdigit()


# ---------------------------------------------------------------------
# Matching configuration (defaults chosen for robustness)
# ---------------------------------------------------------------------

DEFAULT_MIN_LEN = 4
DEFAULT_SIM_THRESHOLD = 90  # RapidFuzz QRatio scale 0..100
DEFAULT_WINDOW = 80         # how far ahead we search in HTR tokens from cursor
DEFAULT_MAX_NGRAM = 4       # allow matching GT token against up to 4 HTR tokens
DEFAULT_ALLOW_NUMERIC = True
DEFAULT_INCLUDE_PUNCT = False  # your corpus: almost no punct; this reduces noise


# Optional stopwords: helps avoid anchoring on ultra-common short function words.
# (You can extend this; safe defaults are conservative.)
STOPWORDS = {
    "de", "la", "el", "del", "y", "en", "a", "por", "que", "los", "las", "un", "una",
}


def _eligible_for_match(word: str, min_len: int, allow_numeric: bool) -> bool:
    if not word:
        return False
    if allow_numeric and _is_probably_numeric(word):
        return True
    if len(word) < min_len:
        return False
    if word.lower() in STOPWORDS:
        return False
    return True


# ---------------------------------------------------------------------
# Core: find best match for a single GT token within a window
# ---------------------------------------------------------------------

def _best_htr_span_for_gt(
    gt_word_orig: str,
    htr_tokens: List[Dict],
    htr_cursor: int,
    window: int,
    max_ngram: int,
    sim_threshold: int,
    min_len: int,
    allow_numeric: bool,
) -> Tuple[Optional[int], Optional[int], int, str]:
    """
    Returns (best_start_idx, best_len, best_score, best_htr_joined_orig)

    best_len is number of tokens in the matched HTR span (1..max_ngram).
    best_htr_joined_orig is the original-space joined string for that span.
    """
    gt_norm = normalise_for_match(gt_word_orig)

    # If GT token not eligible, we still *try* to match 1:1 cheaply to keep flow,
    # but we don't do expensive multi-ngram scans.
    eligible = _eligible_for_match(gt_word_orig, min_len=min_len, allow_numeric=allow_numeric)

    search_end = min(len(htr_tokens), htr_cursor + window)

    best_score = -1
    best_j: Optional[int] = None
    best_k: Optional[int] = None
    best_joined = ""

    # Fast path: if not eligible, only consider single-token exact-ish matches
    # to avoid nonsense.
    k_values = range(1, max_ngram + 1) if eligible else range(1, 2)

    # Evaluate candidates
    for j in range(htr_cursor, search_end):
        # try span lengths 1..max_ngram
        for k in k_values:
            if j + k > search_end:
                break

            span_tokens = htr_tokens[j : j + k]
            joined_orig = " ".join(t["text"] for t in span_tokens)
            joined_norm = normalise_for_match(joined_orig)

            if not joined_norm:
                continue

            # scoring:
            # - QRatio is a good general-purpose choice
            # - for numeric GT, prefer exact numeric matches strongly
            if allow_numeric and _is_probably_numeric(gt_word_orig):
                if joined_norm == gt_norm:
                    score = 100
                else:
                    # numeric mismatch should not "kind of match"
                    score = fuzz.QRatio(gt_norm, joined_norm) - 10
            else:
                score = fuzz.QRatio(gt_norm, joined_norm)

            # tie-breakers:
            # 1) higher score
            # 2) shorter span (prefer 1 token over multi-token)
            # 3) closer to cursor (prefer smaller j)
            if score > best_score:
                best_score = score
                best_j = j
                best_k = k
                best_joined = joined_orig
            elif score == best_score and best_j is not None and best_k is not None:
                if k < best_k:
                    best_j = j
                    best_k = k
                    best_joined = joined_orig
                elif k == best_k and j < best_j:
                    best_j = j
                    best_k = k
                    best_joined = joined_orig

            # Early exit: perfect match
            if best_score >= 100:
                break
        if best_score >= 100:
            break

    # accept / reject
    if best_score >= sim_threshold and best_j is not None and best_k is not None:
        return best_j, best_k, int(best_score), best_joined

    return None, None, int(best_score), ""


# ---------------------------------------------------------------------
# Public API: align word sequences
# ---------------------------------------------------------------------

def align_word_sequences(
    gt_tokens: List[Dict],
    htr_tokens: List[Dict],
    similarity_threshold: float = 0.5,  # kept for backwards-compat calls; not used directly
    *,
    sim_threshold: int = DEFAULT_SIM_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    max_ngram: int = DEFAULT_MAX_NGRAM,
    min_len: int = DEFAULT_MIN_LEN,
    allow_numeric: bool = DEFAULT_ALLOW_NUMERIC,
    include_punct: bool = DEFAULT_INCLUDE_PUNCT,
) -> List[Dict]:
    """
    GT-anchored greedy alignment.

    Notes
    -----
    - similarity_threshold param is kept so existing callers that pass it
      don't crash. The effective threshold used is sim_threshold (0..100).
    - include_punct=False is recommended for your data (reduces noise).
    """

    # Filter tokens used for matching
    if not include_punct:
        gt_seq = [t for t in gt_tokens if t.get("kind") == "word"]
        htr_seq = [t for t in htr_tokens if t.get("kind") == "word"]
    else:
        gt_seq = gt_tokens[:]
        htr_seq = htr_tokens[:]

    ops: List[Dict] = []
    htr_cursor = 0

    # Walk GT in order
    for gt_tok in gt_seq:
        gt_word = gt_tok["text"]

        # Find best HTR match near the cursor
        best_j, best_k, best_score, best_joined = _best_htr_span_for_gt(
            gt_word_orig=gt_word,
            htr_tokens=htr_seq,
            htr_cursor=htr_cursor,
            window=window,
            max_ngram=max_ngram,
            sim_threshold=sim_threshold if sim_threshold is not None else sim_threshold,  # noop; clarity
            min_len=min_len,
            allow_numeric=allow_numeric,
        )

        # No acceptable match -> delete GT token
        if best_j is None or best_k is None:
            ops.append({
                "op": "delete",
                "gt": gt_tok,
                "htr": None,
                "word_gt": gt_word,
                "word_htr": None,
                "htr_span_tokens": [],
                "match_score": None,
            })
            continue

        # Any HTR tokens skipped before the match are insertions
        if best_j > htr_cursor:
            for skipped in htr_seq[htr_cursor:best_j]:
                ops.append({
                    "op": "insert",
                    "gt": None,
                    "htr": skipped,
                    "word_gt": None,
                    "word_htr": skipped["text"],
                    "htr_span_tokens": [skipped],
                    "match_score": None,
                })

        span_tokens = htr_seq[best_j:best_j + best_k]
        first_htr = span_tokens[0]

        # equal vs replace:
        # - "equal" only if original token text matches exactly AND k==1
        # - otherwise "replace"
        if best_k == 1 and gt_word == first_htr["text"]:
            op_type = "equal"
        else:
            op_type = "replace"

        ops.append({
            "op": op_type,
            "gt": gt_tok,
            "htr": first_htr,
            "word_gt": gt_word,
            "word_htr": best_joined,  # joined original span
            "htr_span_tokens": span_tokens,
            "match_score": best_score,
        })

        # If match span is multi-token, emit inserts for the remaining tokens
        if best_k > 1:
            for extra in span_tokens[1:]:
                ops.append({
                    "op": "insert",
                    "gt": None,
                    "htr": extra,
                    "word_gt": None,
                    "word_htr": extra["text"],
                    "htr_span_tokens": [extra],
                    "match_score": None,
                    "part_of_multiword_match": True,
                    "multiword_gt": gt_word,
                    "multiword_htr": best_joined,
                })

        # Advance cursor past matched span
        htr_cursor = best_j + best_k

    # Any remaining HTR tokens are insertions
    for remaining in htr_seq[htr_cursor:]:
        ops.append({
            "op": "insert",
            "gt": None,
            "htr": remaining,
            "word_gt": None,
            "word_htr": remaining["text"],
            "htr_span_tokens": [remaining],
            "match_score": None,
        })

    return ops