"""
report_metrics.py

Script contains helpers to generate the corpus report
Called by build_corpus_report.py

Script carries out thr following:

- loading and aggregating logged issues
- describing the training corpus by style
- computing true CER from GT/HTR text
- computing style-level comparison metrics
- estimating boundary-error behaviour
- estimating alignment drift from Step 2 issue ratios
- extracting style-specific character / bigram / word confusion signals
- preparing document-level diagnostic tables

"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import math
import statistics
import unicodedata
from typing import Any

import regex as re
from rapidfuzz.distance import Levenshtein

from utils.config import LOGS_DIR, SCHEMAS_DIR
from utils.file_io import load_json_if_exists, read_json, read_text


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DRIFT_THRESHOLD = 0.40
MAX_BOUNDARY_NGRAM = 4
MIN_WORD_CONFUSION_LEN = 3


# ---------------------------------------------------------------------
# Formatting-neutral helpers
# ---------------------------------------------------------------------

def percentile(values: list[float | int], p: float) -> float:
    """
    Return a simple percentile from a sorted list using the same deterministic
    index method used elsewhere in the codebase

    Parameters
    ----------
    values:
        Sequence of numeric values
    p:
        Percentile in the interval [0, 1]

    Returns
    -------
    float
    """
    if not values:
        return 0.0

    ordered = sorted(values)
    idx = int(len(ordered) * p)
    idx = min(idx, len(ordered) - 1)
    return float(ordered[idx])


def gini(values: list[int | float]) -> float:
    """
    Compute the Gini coefficient for a non-negative sequence.

    Interpretation
    --------------
    0.0
        Perfectly uniform burden across documents.
    1.0
        All burden concentrated in one document.
    """
    ordered = sorted(v for v in values if v >= 0)
    if not ordered:
        return 0.0

    total = sum(ordered)
    if total == 0:
        return 0.0

    weighted = 0.0
    for idx, value in enumerate(ordered, start = 1):
        weighted += idx * value

    n = len(ordered)
    return (2 * weighted) / (n * total) - (n + 1) / n


def lorenz_points(values: list[int | float]) -> list[tuple[float, float]]:
    """
    Build Lorenz-curve coordinates for a sequence of document burdens.
    """
    ordered = sorted(v for v in values if v >= 0)
    if not ordered:
        return [(0.0, 0.0), (1.0, 1.0)]

    total = sum(ordered)
    if total == 0:
        return [(0.0, 0.0), (1.0, 1.0)]

    points = [(0.0, 0.0)]
    running = 0.0

    for idx, value in enumerate(ordered, start = 1):
        running += value
        points.append((idx / len(ordered), running / total))

    return points


def weight_label(diff_pp: float, threshold_pp: float = 1.0) -> str:
    """
    Labels whether a style is over- or under-represented in issues relative to
    its share of the training corpus
    """
    if diff_pp > threshold_pp:
        return "Overweight"
    if diff_pp < -threshold_pp:
        return "Underweight"
    return "Balanced"


# ---------------------------------------------------------------------
# Text normalisation for CER
# ---------------------------------------------------------------------

def normalise_text_for_cer(text: str) -> str:
    """
    Normalise text for model-oriented CER

    Steps
    -----
    - casefold
    - normalise Unicode to NFC
    - retain Latin letters only
    - remove whitespace, punctuation, digits, and other symbols

    Reasining
    ---------
    The first-stage cleaning goal is not punctuation expansion or whitespace
    normalisation. A normalised CER therefore gives a cleaner measure of
    orthographic/transcription burden for downstream ML.

    Notes
    -----
    This intentionally keeps historical Latin graphemes such as accented vowels,
    ñ, ü, and ç, while excluding digits and punctuation.
    """
    text = unicodedata.normalize("NFC", text.casefold())
    return "".join(ch for ch in text if re.match(r"\p{Latin}", ch))


# ---------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------

def load_stopwords() -> set[str]:
    """
    Load stopwords from the first known schema location that exists
    """
    candidates = [
        SCHEMAS_DIR / "stopwords.json",
        SCHEMAS_DIR / "spanish_stopwords.json",
        Path("schemas_and_manifests/stopwords.json"),
        Path("schemas_and_manifests/spanish_stopwords.json"),
    ]

    for path in candidates:
        if path.exists():
            return {str(x).strip().lower() for x in read_json(path)}

    return set()


def load_all_issues() -> list[dict[str, Any]]:
    """
    Load all per-document issues and inject style/doc_id from directory structure

    Only training documents have issue logs in the current pipeline, so this is
    effectively a training-corpus view of logged issues
    """
    issues: list[dict[str, Any]] = []

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir():
            continue
        if style_dir.name in {"meta", "posthoc", "review", "step_summaries"}:
            continue

        style = style_dir.name

        for doc_dir in style_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            issues_path = doc_dir / f"{doc_id}_issues.json"
            if not issues_path.exists():
                continue

            doc_issues = load_json_if_exists(issues_path, [])

            for issue in doc_issues:
                item = dict(issue)
                item["style"] = style
                item["doc_id"] = doc_id
                issues.append(item)

    return issues


# ---------------------------------------------------------------------
# Corpus description
# ---------------------------------------------------------------------

def _splitlines_nonempty(text: str) -> list[str]:
    """
    Return non-empty lines for descriptive corpus geometry
    """
    return [line for line in text.splitlines() if line.strip()]


def train_distribution_by_style(train_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Count training documents by style and compute share of the training corpus
    """
    counts = Counter(p["style"] for p in train_pairs)
    total = sum(counts.values())

    rows = []
    for style in sorted(counts):
        n_docs = counts[style]
        pct_train = (n_docs / total) if total else 0.0

        rows.append({
            "style": style,
            "train_docs": n_docs,
            "pct_train": pct_train,
        })

    return rows


def geometry_by_style(train_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Describe the training corpus by style to include:

    - document count
    - total lines / tokens / chars
    - per-document line-count percentiles
    - per-document token-count percentiles
    - per-line character-length percentiles
    """
    line_lengths_by_style: dict[str, list[int]] = defaultdict(list)
    doc_lines_by_style: dict[str, list[int]] = defaultdict(list)
    doc_tokens_by_style: dict[str, list[int]] = defaultdict(list)
    total_lines_by_style: Counter = Counter()
    total_tokens_by_style: Counter = Counter()
    total_chars_by_style: Counter = Counter()
    doc_counts: Counter = Counter()

    for pair in train_pairs:
        style = pair["style"]
        text = read_text(Path(pair["htr_path"]))
        lines = _splitlines_nonempty(text)

        doc_counts[style] += 1
        doc_lines_by_style[style].append(len(lines))
        total_lines_by_style[style] += len(lines)

        token_count = 0
        char_count = 0

        for line in lines:
            tokens = line.split()
            token_count += len(tokens)
            char_count += len(line)
            line_lengths_by_style[style].append(len(line))

        doc_tokens_by_style[style].append(token_count)
        total_tokens_by_style[style] += token_count
        total_chars_by_style[style] += char_count

    rows = []
    for style in sorted(doc_counts):
        line_lengths = sorted(line_lengths_by_style[style])
        doc_line_counts = sorted(doc_lines_by_style[style])
        doc_token_counts = sorted(doc_tokens_by_style[style])

        rows.append({
            "style": style,
            "docs": doc_counts[style],
            "total_lines": total_lines_by_style[style],
            "total_tokens": total_tokens_by_style[style],
            "total_chars": total_chars_by_style[style],
            "doc_lines_p25": int(percentile(doc_line_counts, 0.25)),
            "doc_lines_median": int(percentile(doc_line_counts, 0.50)),
            "doc_lines_p75": int(percentile(doc_line_counts, 0.75)),
            "doc_tokens_p25": int(percentile(doc_token_counts, 0.25)),
            "doc_tokens_median": int(percentile(doc_token_counts, 0.50)),
            "doc_tokens_p75": int(percentile(doc_token_counts, 0.75)),
            "line_len_p25": int(percentile(line_lengths, 0.25)),
            "line_len_median": int(percentile(line_lengths, 0.50)),
            "line_len_p75": int(percentile(line_lengths, 0.75)),
        })

    return rows


# ---------------------------------------------------------------------
# Logged issue overview
# ---------------------------------------------------------------------

def issue_stage_overview(issues: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarise total issues and split out by pipeline stage
    """
    counts = Counter(i["tag"][:2] for i in issues)
    total = sum(counts.values())

    return {
        "total_issues": total,
        "S1": counts.get("S1", 0),
        "S2": counts.get("S2", 0),
        "S3": counts.get("S3", 0),
        "S1_pct": counts.get("S1", 0) / total if total else 0.0,
        "S2_pct": counts.get("S2", 0) / total if total else 0.0,
        "S3_pct": counts.get("S3", 0) / total if total else 0.0,
    }


def issue_distribution_by_style(
    issues: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compare each style's share of logged issues against its share of the
    training corpus
    """
    issue_counts = Counter(i["style"] for i in issues)
    train_counts = Counter(p["style"] for p in train_pairs)
    stage_counts = defaultdict(Counter)

    total_issues = sum(issue_counts.values())
    total_train = sum(train_counts.values())

    for issue in issues:
        stage_counts[issue["style"]][issue["tag"][:2]] += 1

    rows = []
    for style in sorted(set(issue_counts) | set(train_counts)):
        issue_n = issue_counts.get(style, 0)
        train_n = train_counts.get(style, 0)

        corpus_pct = train_n / total_train if total_train else 0.0
        issue_pct = issue_n / total_issues if total_issues else 0.0
        diff_pp = (issue_pct - corpus_pct) * 100

        rows.append({
            "style": style,
            "train_docs": train_n,
            "pct_train": corpus_pct,
            "issues": issue_n,
            "pct_issues": issue_pct,
            "diff_pp": diff_pp,
            "weight": weight_label(diff_pp),
            "S1": stage_counts[style].get("S1", 0),
            "S2": stage_counts[style].get("S2", 0),
            "S3": stage_counts[style].get("S3", 0),
        })

    return rows


# ---------------------------------------------------------------------
# True edit burden from raw text
# ---------------------------------------------------------------------

def compute_edit_counts(gt_text: str, htr_text: str) -> dict[str, int]:
    """
    Compute substitutions, deletions, insertions, and total edits from raw text

    This uses RapidFuzz edit operations on the normalised text used for the
    report's main CER comparisons.
    """
    gt_norm = normalise_text_for_cer(gt_text)
    htr_norm = normalise_text_for_cer(htr_text)

    s = d = i = 0
    for op in Levenshtein.editops(gt_norm, htr_norm):
        tag = getattr(op, "tag", op[0])
        if tag == "replace":
            s += 1
        elif tag == "delete":
            d += 1
        elif tag == "insert":
            i += 1

    return {
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "edits": s + d + i,
        "norm_len": len(gt_norm),
    }


# ---------------------------------------------------------------------
# Boundary-event diagnostics
# ---------------------------------------------------------------------

def tokenise_for_boundary(text: str) -> list[str]:
    """
    Tokenise text for boundary-event analysis using simple whitespace tokenisation
    """
    return text.split()


def boundary_norm(token: str) -> str:
    """
    Remove whitespace and lowercase for token-boundary comparisons
    """
    return "".join(token.split()).casefold()


def detect_boundary_events_for_doc(
    gt_text: str,
    htr_text: str,
    max_span: int = MAX_BOUNDARY_NGRAM,
) -> dict[str, Any]:
    """
    Detect boundary-only candidate events in a conservative local scan

    Event types
    -----------
    split
        One GT token corresponds to multiple HTR tokens.
    merge
        Multiple GT tokens correspond to one HTR token.
    complex
        Multi-token-to-multi-token boundary-only remapping.

    Matching rule
    -------------
    Concatenated token spans must match exactly after whitespace removal and
    casefolding. This keeps the boundary signal conservative and avoids
    conflating boundary shifts with broader character errors.

    Notes
    ---------
    This is intentionally conservative and approximate. It is most trustworthy
    when alignment drift is limited. For high-drift documents, boundary metrics
    should be interpreted with caution.
    """
    gt_tokens = tokenise_for_boundary(gt_text)
    htr_tokens = tokenise_for_boundary(htr_text)

    i = 0
    j = 0
    split_count = 0
    merge_count = 0
    complex_count = 0
    examples = Counter()

    while i < len(gt_tokens) and j < len(htr_tokens):
        if boundary_norm(gt_tokens[i]) == boundary_norm(htr_tokens[j]):
            i += 1
            j += 1
            continue

        matched = False

        for a in range(1, max_span + 1):
            if i + a > len(gt_tokens):
                break

            gt_span = gt_tokens[i:i + a]
            gt_join = "".join(boundary_norm(tok) for tok in gt_span)

            for b in range(1, max_span + 1):
                if j + b > len(htr_tokens):
                    break
                if a == 1 and b == 1:
                    continue

                htr_span = htr_tokens[j:j + b]
                htr_join = "".join(boundary_norm(tok) for tok in htr_span)

                if gt_join and gt_join == htr_join:
                    gt_label = " ".join(gt_span)
                    htr_label = " ".join(htr_span)

                    if a == 1 and b > 1:
                        split_count += 1
                        examples[("split", gt_label, htr_label)] += 1
                    elif a > 1 and b == 1:
                        merge_count += 1
                        examples[("merge", gt_label, htr_label)] += 1
                    else:
                        complex_count += 1
                        examples[("complex", gt_label, htr_label)] += 1

                    i += a
                    j += b
                    matched = True
                    break

            if matched:
                break

        if not matched:
            i += 1
            j += 1

    return {
        "gt_tokens": len(gt_tokens),
        "htr_tokens": len(htr_tokens),
        "splits": split_count,
        "merges": merge_count,
        "complex_boundary": complex_count,
        "boundary_events": split_count + merge_count + complex_count,
        "examples": examples,
    }


# ---------------------------------------------------------------------
# Alignment drift from logged Step 2 issues
# ---------------------------------------------------------------------

def compute_drift_from_doc_issues(doc_issues: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Recompute alignment-drift diagnostics from the per-document Step 2 issue mix

    A document is flagged as drift if insertion or deletion ratio exceeds the
    configured threshold

    In addition to the binary drift flag, the returned `drift_skew` helps
    distinguish insertion-heavy vs deletion-heavy drift behaviour
    """
    counts = Counter(i["tag"] for i in doc_issues if i.get("tag", "").startswith("S2"))

    s2x = counts.get("S2X", 0)
    s2i = counts.get("S2I", 0)
    s2d = counts.get("S2D", 0)
    total = s2x + s2i + s2d

    insert_ratio = s2i / total if total else 0.0
    delete_ratio = s2d / total if total else 0.0
    replace_ratio = s2x / total if total else 0.0

    return {
        "s2_total": total,
        "s2x": s2x,
        "s2i": s2i,
        "s2d": s2d,
        "insert_ratio": insert_ratio,
        "delete_ratio": delete_ratio,
        "replace_ratio": replace_ratio,
        "drift_skew": abs(insert_ratio - delete_ratio),
        "drift_flag": (insert_ratio > DRIFT_THRESHOLD) or (delete_ratio > DRIFT_THRESHOLD),
    }


# ---------------------------------------------------------------------
# Per-document master metrics
# ---------------------------------------------------------------------

def build_doc_issue_lookup(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Group issues by document id.
    """
    lookup = defaultdict(list)
    for issue in issues:
        lookup[issue["doc_id"]].append(issue)
    return lookup


def compute_doc_metrics(
    train_pairs: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compute the full document-level metric set used by the report

    Notes
    -----
    - true edits are based on normalised GT/HTR text
    - issue counts are loaded from logs and kept separate
    - boundary events are computed from raw token sequences
    - edit_density is included so long documents do not dominate absolute
      burden views by length alone
    """
    doc_issue_lookup = build_doc_issue_lookup(issues)
    rows: list[dict[str, Any]] = []

    for pair in train_pairs:
        doc_id = pair["id"]
        style = pair["style"]

        gt_text = read_text(Path(pair["gt_path"]))
        htr_text = read_text(Path(pair["htr_path"]))

        cer_raw = Levenshtein.distance(gt_text, htr_text) / max(len(gt_text), 1)
        edits = compute_edit_counts(gt_text, htr_text)
        boundary = detect_boundary_events_for_doc(gt_text, htr_text)
        doc_issues = doc_issue_lookup.get(doc_id, [])
        drift = compute_drift_from_doc_issues(doc_issues)

        stage_counts = Counter(i["tag"][:2] for i in doc_issues)

        rows.append({
            "doc_id": doc_id,
            "style": style,
            "cer_raw": cer_raw,
            "cer_norm": edits["edits"] / max(edits["norm_len"], 1),
            "ref_chars_norm": edits["norm_len"],
            "edit_density": edits["edits"] / max(edits["norm_len"], 1),
            "edits_per_1k_chars": (edits["edits"] / max(edits["norm_len"], 1)) * 1000,
            "substitutions": edits["substitutions"],
            "deletions": edits["deletions"],
            "insertions": edits["insertions"],
            "edits": edits["edits"],
            "split_count": boundary["splits"],
            "merge_count": boundary["merges"],
            "complex_boundary": boundary["complex_boundary"],
            "boundary_events": boundary["boundary_events"],
            "gt_tokens": boundary["gt_tokens"],
            "boundary_pct_of_edits": (
                boundary["boundary_events"] / edits["edits"] if edits["edits"] else 0.0
            ),
            "logged_issues": len(doc_issues),
            "S1_issues": stage_counts.get("S1", 0),
            "S2_issues": stage_counts.get("S2", 0),
            "S3_issues": stage_counts.get("S3", 0),
            "drift_flag": drift["drift_flag"],
            "drift_insert_ratio": drift["insert_ratio"],
            "drift_delete_ratio": drift["delete_ratio"],
            "drift_replace_ratio": drift["replace_ratio"],
            "drift_skew": drift["drift_skew"],
            "s2_total": drift["s2_total"],
        })

    return rows


# ---------------------------------------------------------------------
# Style-first comparison layer
# ---------------------------------------------------------------------

def aggregate_style_metrics(doc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate document metrics into the primary style-comparison layer.

    Returns
    -------
    dict with:
        - corpus_summary
        - style_rows
        - concentration_rows
        - lorenz_data
    """
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_rows:
        by_style[row["style"]].append(row)

    corpus_docs = len(doc_rows)
    corpus_edits = sum(r["edits"] for r in doc_rows)
    corpus_chars = sum(r["ref_chars_norm"] for r in doc_rows)
    corpus_boundary_events = sum(r["boundary_events"] for r in doc_rows)
    corpus_splits = sum(r["split_count"] for r in doc_rows)
    corpus_merges = sum(r["merge_count"] for r in doc_rows)
    corpus_gt_tokens = sum(r["gt_tokens"] for r in doc_rows)

    corpus_mean_cer = statistics.mean(r["cer_norm"] for r in doc_rows) if doc_rows else 0.0
    corpus_boundary_pct = corpus_boundary_events / corpus_edits if corpus_edits else 0.0
    corpus_split_rate = (corpus_splits / corpus_gt_tokens) * 1000 if corpus_gt_tokens else 0.0
    corpus_merge_rate = (corpus_merges / corpus_gt_tokens) * 1000 if corpus_gt_tokens else 0.0
    corpus_drift_rate = sum(1 for r in doc_rows if r["drift_flag"]) / corpus_docs if corpus_docs else 0.0
    corpus_edit_density = corpus_edits / corpus_chars if corpus_chars else 0.0

    corpus_summary = {
        "docs": corpus_docs,
        "ref_chars_norm": corpus_chars,
        "edits": corpus_edits,
        "mean_cer_norm": corpus_mean_cer,
        "boundary_pct": corpus_boundary_pct,
        "split_rate": corpus_split_rate,
        "merge_rate": corpus_merge_rate,
        "drift_rate": corpus_drift_rate,
        "edit_density": corpus_edit_density,
    }

    style_rows = []
    concentration_rows = []
    lorenz_data = {}

    for style in sorted(by_style):
        docs = by_style[style]
        cer_vals = [r["cer_norm"] for r in docs]
        edit_vals = [r["edits"] for r in docs]

        total_gt_tokens = sum(r["gt_tokens"] for r in docs)
        total_ref_chars = sum(r["ref_chars_norm"] for r in docs)
        total_edits = sum(r["edits"] for r in docs)
        total_s = sum(r["substitutions"] for r in docs)
        total_d = sum(r["deletions"] for r in docs)
        total_i = sum(r["insertions"] for r in docs)
        total_splits = sum(r["split_count"] for r in docs)
        total_merges = sum(r["merge_count"] for r in docs)
        total_boundary = sum(r["boundary_events"] for r in docs)

        drift_docs = sum(1 for r in docs if r["drift_flag"])
        clean_1 = sum(1 for r in docs if r["cer_norm"] < 0.01)
        clean_2 = sum(1 for r in docs if r["cer_norm"] < 0.02)
        clean_5 = sum(1 for r in docs if r["cer_norm"] < 0.05)

        mean_cer = statistics.mean(cer_vals) if cer_vals else 0.0
        median_cer = statistics.median(cer_vals) if cer_vals else 0.0
        p90_cer = percentile(cer_vals, 0.90) if cer_vals else 0.0

        split_rate = (total_splits / total_gt_tokens) * 1000 if total_gt_tokens else 0.0
        merge_rate = (total_merges / total_gt_tokens) * 1000 if total_gt_tokens else 0.0
        boundary_pct = total_boundary / total_edits if total_edits else 0.0
        drift_rate = drift_docs / len(docs) if docs else 0.0
        edit_density = total_edits / total_ref_chars if total_ref_chars else 0.0

        style_rows.append({
            "style": style,
            "docs": len(docs),
            "mean_cer": mean_cer,
            "median_cer": median_cer,
            "p90_cer": p90_cer,
            "gini": gini(edit_vals),
            "edit_density": edit_density,
            "split_rate": split_rate,
            "merge_rate": merge_rate,
            "boundary_pct": boundary_pct,
            "drift_rate": drift_rate,
            "clean_lt_1_pct": clean_1 / len(docs) if docs else 0.0,
            "clean_lt_2_pct": clean_2 / len(docs) if docs else 0.0,
            "clean_lt_5_pct": clean_5 / len(docs) if docs else 0.0,
            "sub_pct": total_s / total_edits if total_edits else 0.0,
            "del_pct": total_d / total_edits if total_edits else 0.0,
            "ins_pct": total_i / total_edits if total_edits else 0.0,
            "delta_cer": mean_cer - corpus_mean_cer,
            "delta_boundary_pct": boundary_pct - corpus_boundary_pct,
            "delta_split_rate": split_rate - corpus_split_rate,
            "delta_merge_rate": merge_rate - corpus_merge_rate,
            "delta_drift_rate": drift_rate - corpus_drift_rate,
            "delta_edit_density": edit_density - corpus_edit_density,
            "total_edits": total_edits,
        })

        ranked_edits = sorted(edit_vals, reverse = True)
        top_decile_n = max(1, math.ceil(len(ranked_edits) * 0.10))
        top_decile_share = sum(ranked_edits[:top_decile_n]) / total_edits if total_edits else 0.0
        top_5_share = sum(ranked_edits[:5]) / total_edits if total_edits else 0.0

        concentration_rows.append({
            "style": style,
            "docs": len(docs),
            "total_edits": total_edits,
            "gini": gini(edit_vals),
            "top_10pct_docs_share": top_decile_share,
            "top_5_docs_share": top_5_share,
        })

        lorenz_data[style] = lorenz_points(edit_vals)

    return {
        "corpus_summary": corpus_summary,
        "style_rows": style_rows,
        "concentration_rows": concentration_rows,
        "lorenz_data": lorenz_data,
    }


# ---------------------------------------------------------------------
# Confusion drivers by style
# ---------------------------------------------------------------------

def char_confusions_by_style(
    issues: list[dict[str, Any]],
    top_n: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """
    Top character confusions by style from Step 2 substitution spans.
    """
    table = defaultdict(Counter)
    totals = defaultdict(int)

    for issue in issues:
        if issue.get("tag") != "S2X":
            continue

        style = issue["style"]
        gt = issue.get("gt_text", "") or ""
        htr = issue.get("htr_text", "") or ""

        for g, h in zip(gt, htr):
            if g != h:
                table[style][(g, h)] += 1
                totals[style] += 1

    out: dict[str, list[dict[str, Any]]] = {}
    for style in sorted(table):
        rows = []
        for (g, h), count in table[style].most_common(top_n):
            rows.append({
                "style": style,
                "gt": g,
                "htr": h,
                "count": count,
                "pct_style_char_confusions": count / totals[style] if totals[style] else 0.0,
            })
        out[style] = rows

    return out


def bigram_confusions_by_style(
    issues: list[dict[str, Any]],
    top_n: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """
    Top bigram confusions by style.

    Conservative definition
    -----------------------
    Use Step 2 substitution spans where the GT span normalises to exactly two
    characters after whitespace removal. The HTR output can be length 1+.

    Limitation
    ----------
    This captures only a conservative subset of transition-level errors. It does
    not recover every possible bigram/ligature failure from the full alignment path.
    """
    table = defaultdict(Counter)
    totals = defaultdict(int)

    for issue in issues:
        if issue.get("tag") != "S2X":
            continue

        style = issue["style"]
        gt = re.sub(r"\s+", "", (issue.get("gt_text") or ""))
        htr = re.sub(r"\s+", "", (issue.get("htr_text") or ""))

        if len(gt) != 2:
            continue
        if not htr:
            continue

        table[style][(gt, htr)] += 1
        totals[style] += 1

    out: dict[str, list[dict[str, Any]]] = {}
    for style in sorted(table):
        rows = []
        for (gt_bg, htr_out), count in table[style].most_common(top_n):
            rows.append({
                "style": style,
                "gt_bigram": gt_bg,
                "htr_out": htr_out,
                "count": count,
                "pct_style_bigram_confusions": count / totals[style] if totals[style] else 0.0,
            })
        out[style] = rows

    return out


def word_confusions_by_style(
    issues: list[dict[str, Any]],
    stopwords: set[str],
    top_n: int = 20,
    min_len: int = MIN_WORD_CONFUSION_LEN,
) -> dict[str, list[dict[str, Any]]]:
    """
    Top word-level substitution confusions by style.

    Short function-word noise is reduced by:
    - stopword filtering
    - minimum token-length filtering
    """
    table = defaultdict(Counter)
    totals = defaultdict(int)

    for issue in issues:
        if issue.get("tag") != "S2X":
            continue

        style = issue["style"]
        gt = (issue.get("word_gt") or "").strip().lower()
        htr = (issue.get("word_htr") or "").strip().lower()

        if not gt or not htr:
            continue
        if gt in stopwords:
            continue
        if len(gt) < min_len:
            continue
        if gt == htr:
            continue

        table[style][(gt, htr)] += 1
        totals[style] += 1

    out: dict[str, list[dict[str, Any]]] = {}
    for style in sorted(table):
        rows = []
        for (gt_word, htr_word), count in table[style].most_common(top_n):
            rows.append({
                "style": style,
                "gt_word": gt_word,
                "htr_word": htr_word,
                "count": count,
                "pct_style_word_confusions": count / totals[style] if totals[style] else 0.0,
            })
        out[style] = rows

    return out


# ---------------------------------------------------------------------
# Problematic-document views
# ---------------------------------------------------------------------

def top_documents_overall(
    doc_rows: list[dict[str, Any]],
    top_n: int = 25,
) -> list[dict[str, Any]]:
    """
    Top documents overall by true edit burden
    """
    ranked = sorted(
        doc_rows,
        key = lambda r: (-r["edits"], -r["cer_norm"], -r["logged_issues"], r["doc_id"]),
    )
    return ranked[:top_n]


def per_style_document_blocks(
    doc_rows: list[dict[str, Any]],
    top_n: int = 10,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Prepare per-style document diagnostic blocks

    Each style gets:
    - top by edit burden
    - top by logged issue count
    - top by boundary burden
    - all drift-flagged docs
    """
    by_style = defaultdict(list)
    for row in doc_rows:
        by_style[row["style"]].append(row)

    out = {}

    for style in sorted(by_style):
        docs = by_style[style]

        out[style] = {
            "top_edit_burden": sorted(
                docs,
                key = lambda r: (-r["edits"], -r["cer_norm"], r["doc_id"]),
            )[:top_n],
            "top_issue_count": sorted(
                docs,
                key = lambda r: (-r["logged_issues"], -r["S2_issues"], -r["edits"], r["doc_id"]),
            )[:top_n],
            "top_boundary_burden": sorted(
                docs,
                key = lambda r: (-r["boundary_pct_of_edits"], -r["boundary_events"], -r["edits"], r["doc_id"]),
            )[:top_n],
            "drift_docs": sorted(
                [r for r in docs if r["drift_flag"]],
                key = lambda r: (-r["s2_total"], -r["drift_skew"], -r["edits"], r["doc_id"]),
            ),
        }

    return out