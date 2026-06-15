"""
report_metrics.py

Canonical analytics layer for the simplified HTR corpus report.

This refactored version focuses on the operational document metrics needed to:
- describe the corpus clearly
- support risk-weighted, stratified human review sampling
- inform downstream ML transcript-cleaning work

Design principles
-----------------
- S1/S3 log counts are used only for surface/anomaly densities.
- S2 log counts are not used for CER/WER/SDR/IDR; those are computed directly
  from GT/HTR alignment.
- The main output is a compact document-level operational metrics table with
  robust style-relative risk scores and risk bands.
- Debugging-oriented legacy diagnostics are deliberately minimised.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import statistics
import unicodedata
from typing import Any

import regex as re
from rapidfuzz.distance import Levenshtein

from utils.config import LOGS_DIR, SCHEMAS_DIR
from utils.file_io import load_json_if_exists, read_json, read_text


# ---------------------------------------------------------------------
# Report metric configuration
# ---------------------------------------------------------------------

MAX_BOUNDARY_NGRAM = 4
MIN_WORD_CONFUSION_LEN = 3
MIN_COHERENT_SPAN_CHARS = 50

WINDOW_MODE = "relative"
RELATIVE_WINDOW_COUNT = 10
FIXED_WINDOW_SIZE_CHARS = 500
MIN_WINDOW_REF_CHARS = 100

LINE_COUNT_DIAGNOSTICS_PATH = LOGS_DIR / "line_counts" / "line_count_diagnostics.jsonl"

RISK_WEIGHTS = {
    "basic_anomaly_density": 0.5,
    "orthographic_violation_density": 1.0,
    "cer_norm": 2.0,
    "wer_norm": 1.0,
    "structural_drift_ratio": 1.5,
    "indel_disruption_rate": 2.0,
    "coverage_risk": 2.5,
    "continuity_risk": 2.5,
    "lasr_risk": 2.0,
    "fragmentation": 1.5,
    "window_max_cer": 1.5,
    "window_sd_cer": 1.0,
    "window_max_sdr": 1.5,
    "boundary_burden_proportion": 1.0,
    "adjusted_line_delta": 0.75,
    "unmatched_htr_blank_density": 0.75,
}


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def percentile(values: list[float | int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(len(ordered) * p)
    idx = min(idx, len(ordered) - 1)
    return float(ordered[idx])


def mean_or_zero(values: list[float | int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def median_or_zero(values: list[float | int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def stdev_or_zero(values: list[float | int]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def median_absolute_deviation(values: list[float | int]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    return float(statistics.median([abs(float(x) - med) for x in values]))


def robust_z(value: float, median: float, mad: float) -> float:
    return (float(value) - median) / mad if mad else 0.0


# ---------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------

def normalise_text_for_cer(text: str) -> str:
    text = unicodedata.normalize("NFC", text.casefold())
    return "".join(ch for ch in text if re.match(r"\p{Latin}", ch))


def normalise_tokens_for_wer(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.casefold())
    return re.findall(r"\p{Latin}+", text)


def _splitlines_nonempty(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------

def load_stopwords() -> set[str]:
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


def build_doc_issue_lookup(issues: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for issue in issues:
        lookup[issue["doc_id"]].append(issue)
    return lookup


# ---------------------------------------------------------------------
# Corpus description
# ---------------------------------------------------------------------

def train_distribution_by_style(train_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(p["style"] for p in train_pairs)
    total = sum(counts.values())
    return [
        {
            "style": style,
            "train_docs": counts[style],
            "pct_train": safe_div(counts[style], total),
        }
        for style in sorted(counts)
    ]


def geometry_by_style(train_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    line_lengths_by_style: dict[str, list[int]] = defaultdict(list)
    doc_lines_by_style: dict[str, list[int]] = defaultdict(list)
    doc_tokens_by_style: dict[str, list[int]] = defaultdict(list)
    total_lines_by_style: Counter = Counter()
    total_tokens_by_style: Counter = Counter()
    total_chars_by_style: Counter = Counter()
    doc_counts: Counter = Counter()

    for pair in train_pairs:
        style = pair["style"]
        text = read_text(Path(pair.get("gt_path") or pair["htr_path"]))
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
# Issue-log metrics
# ---------------------------------------------------------------------

def issue_stage_overview(issues: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(i["tag"][:2] for i in issues)
    total = sum(counts.values())
    return {
        "total_issues": total,
        "S1": counts.get("S1", 0),
        "S2": counts.get("S2", 0),
        "S3": counts.get("S3", 0),
        "S1_pct": safe_div(counts.get("S1", 0), total),
        "S2_pct": safe_div(counts.get("S2", 0), total),
        "S3_pct": safe_div(counts.get("S3", 0), total),
    }


def issue_distribution_by_style(
    issues: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
        corpus_pct = safe_div(train_n, total_train)
        issue_pct = safe_div(issue_n, total_issues)
        diff_pp = (issue_pct - corpus_pct) * 100
        rows.append({
            "style": style,
            "train_docs": train_n,
            "pct_train": corpus_pct,
            "issues": issue_n,
            "pct_issues": issue_pct,
            "diff_pp": diff_pp,
            "weight": "Overweight" if diff_pp > 1 else "Underweight" if diff_pp < -1 else "Balanced",
            "S1": stage_counts[style].get("S1", 0),
            "S2": stage_counts[style].get("S2", 0),
            "S3": stage_counts[style].get("S3", 0),
        })
    return rows


def issue_anomaly_counts(doc_issues: list[dict[str, Any]], ref_chars_norm: int) -> dict[str, Any]:
    stage_counts = Counter((issue.get("tag") or "")[:2] for issue in doc_issues)
    basic_count = stage_counts.get("S1", 0)
    orth_count = stage_counts.get("S3", 0)
    return {
        "basic_anomaly_count": basic_count,
        "orthographic_violation_count": orth_count,
        "basic_anomaly_density": safe_div(basic_count, ref_chars_norm) * 1000,
        "orthographic_violation_density": safe_div(orth_count, ref_chars_norm) * 1000,
    }


# ---------------------------------------------------------------------
# Line-structure diagnostics
# ---------------------------------------------------------------------

def load_line_count_diagnostics(
    path: Path = LINE_COUNT_DIAGNOSTICS_PATH,
) -> dict[str, dict[str, Any]]:
    """
    Load compact line-count diagnostics keyed by both pair_id and doc_id.

    The diagnostics JSONL is produced before report generation by
    run_line_count_diagnostics.py. If the file is absent, the report still
    builds and line-structure fields default to zero.
    """
    lookup: dict[str, dict[str, Any]] = {}

    if not path.exists():
        return lookup

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("pair_id"):
                lookup[str(row["pair_id"])] = row
            if row.get("doc_id"):
                lookup[str(row["doc_id"])] = row

    return lookup


def line_structure_metrics(
    pair: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Return compact GT/HTR line-structure metrics from diagnostics JSONL.

    These metrics summarise whether the paired GT and HTR files remain
    structurally comparable after blank-line handling.
    """
    row = lookup.get(str(pair.get("id")), {})

    if not row:
        htr_path = Path(pair.get("htr_path", ""))
        row = lookup.get(htr_path.stem, {})

    gt_adjusted = int(row.get("gt_adjusted_line_count", 0) or 0)
    htr_adjusted = int(row.get("htr_adjusted_line_count", 0) or 0)
    unmatched_htr_blank = int(row.get("htr_blank_preserved_unmatched_count", 0) or 0)

    adjusted_delta = abs(gt_adjusted - htr_adjusted)

    return {
        "gt_adjusted_line_count": gt_adjusted,
        "htr_adjusted_line_count": htr_adjusted,
        "adjusted_line_delta": adjusted_delta,
        "line_count_ratio": safe_div(
            min(gt_adjusted, htr_adjusted),
            max(gt_adjusted, htr_adjusted),
        ),
        "unmatched_htr_blank_count": unmatched_htr_blank,
        "unmatched_htr_blank_density": safe_div(unmatched_htr_blank, gt_adjusted),
    }


# ---------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------

def _editop_fields(op: Any) -> tuple[str, int, int]:
    tag = getattr(op, "tag", op[0])
    src_pos = getattr(op, "src_pos", op[1])
    dest_pos = getattr(op, "dest_pos", op[2])
    return tag, int(src_pos), int(dest_pos)


def alignment_operations(gt_norm: str, htr_norm: str) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    gt_idx = 0
    htr_idx = 0

    for editop in Levenshtein.editops(gt_norm, htr_norm):
        tag, src_pos, dest_pos = _editop_fields(editop)

        while gt_idx < src_pos and htr_idx < dest_pos:
            ops.append({"op": "M", "gt_pos": gt_idx, "htr_pos": htr_idx})
            gt_idx += 1
            htr_idx += 1

        if tag == "replace":
            ops.append({"op": "S", "gt_pos": src_pos, "htr_pos": dest_pos})
            gt_idx = src_pos + 1
            htr_idx = dest_pos + 1
        elif tag == "delete":
            ops.append({"op": "D", "gt_pos": src_pos, "htr_pos": dest_pos})
            gt_idx = src_pos + 1
            htr_idx = dest_pos
        elif tag == "insert":
            ops.append({"op": "I", "gt_pos": src_pos, "htr_pos": dest_pos})
            gt_idx = src_pos
            htr_idx = dest_pos + 1

    while gt_idx < len(gt_norm) and htr_idx < len(htr_norm):
        ops.append({"op": "M", "gt_pos": gt_idx, "htr_pos": htr_idx})
        gt_idx += 1
        htr_idx += 1

    while gt_idx < len(gt_norm):
        ops.append({"op": "D", "gt_pos": gt_idx, "htr_pos": htr_idx})
        gt_idx += 1

    while htr_idx < len(htr_norm):
        ops.append({"op": "I", "gt_pos": gt_idx, "htr_pos": htr_idx})
        htr_idx += 1

    return ops


def _span_lengths_from_ops(ops: list[dict[str, Any]]) -> list[int]:
    spans: list[int] = []
    current = 0
    for item in ops:
        if item["op"] in {"M", "S"}:
            current += 1
        elif current:
            spans.append(current)
            current = 0
    if current:
        spans.append(current)
    return spans


def _error_run_lengths_from_ops(ops: list[dict[str, Any]]) -> list[int]:
    runs: list[int] = []
    current = 0
    for item in ops:
        if item["op"] != "M":
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _alignment_metrics_from_ops(
    ops: list[dict[str, Any]],
    ref_len: int,
    span_threshold: int = MIN_COHERENT_SPAN_CHARS,
) -> dict[str, Any]:
    counts = Counter(item["op"] for item in ops)
    m = counts.get("M", 0)
    s = counts.get("S", 0)
    d = counts.get("D", 0)
    i = counts.get("I", 0)
    edits = s + d + i
    aligned_total = m + s + d + i

    spans = _span_lengths_from_ops(ops)
    error_runs = _error_run_lengths_from_ops(ops)
    longest_span = max(spans) if spans else 0

    return {
        "matches": m,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "edits": edits,
        "norm_len": ref_len,
        "aligned_total": aligned_total,
        "cer_norm": safe_div(edits, ref_len),
        "structural_drift_ratio": safe_div(i + d, edits),
        "indel_disruption_rate": safe_div(i + d, aligned_total),
        "coverage": safe_div(m + s, m + s + d),
        "continuity": safe_div(sum(x for x in spans if x >= span_threshold), ref_len),
        "lasr": safe_div(longest_span, ref_len),
        "fragmentation": safe_div(len(error_runs), ref_len) * 1000,
    }


def document_window_size(ref_len: int) -> int:
    if ref_len <= 0:
        return 0
    if WINDOW_MODE == "fixed":
        return max(1, FIXED_WINDOW_SIZE_CHARS)
    if WINDOW_MODE != "relative":
        raise ValueError(f"Unsupported WINDOW_MODE {WINDOW_MODE!r}")
    return max(MIN_WINDOW_REF_CHARS, math.ceil(ref_len / max(1, RELATIVE_WINDOW_COUNT)))


def _windowed_alignment_metrics_from_ops(ops: list[dict[str, Any]], ref_len: int) -> dict[str, Any]:
    if ref_len <= 0:
        return {
            "window_mode": WINDOW_MODE,
            "window_size": 0,
            "window_count": 0,
            "window_max_cer": 0.0,
            "window_sd_cer": 0.0,
            "window_max_sdr": 0.0,
        }

    window_size = document_window_size(ref_len)
    window_count = math.ceil(ref_len / window_size)
    windows: list[list[dict[str, Any]]] = [[] for _ in range(window_count)]

    for item in ops:
        gt_pos = max(0, min(int(item.get("gt_pos", 0)), ref_len - 1))
        idx = min(gt_pos // window_size, window_count - 1)
        windows[idx].append(item)

    metrics = []
    for idx, window_ops in enumerate(windows):
        start = idx * window_size
        end = min(start + window_size, ref_len)
        metrics.append(_alignment_metrics_from_ops(window_ops, end - start))

    cer_vals = [m["cer_norm"] for m in metrics]
    sdr_vals = [m["structural_drift_ratio"] for m in metrics]

    return {
        "window_mode": WINDOW_MODE,
        "window_size": window_size,
        "window_count": window_count,
        "window_max_cer": max(cer_vals) if cer_vals else 0.0,
        "window_sd_cer": stdev_or_zero(cer_vals),
        "window_max_sdr": max(sdr_vals) if sdr_vals else 0.0,
    }


def compute_alignment_quality_metrics(gt_text: str, htr_text: str) -> dict[str, Any]:
    gt_norm = normalise_text_for_cer(gt_text)
    htr_norm = normalise_text_for_cer(htr_text)
    ops = alignment_operations(gt_norm, htr_norm)
    whole = _alignment_metrics_from_ops(ops, len(gt_norm))
    windowed = _windowed_alignment_metrics_from_ops(ops, len(gt_norm))
    return {**whole, **windowed}


def compute_edit_counts(gt_text: str, htr_text: str) -> dict[str, int | float]:
    metrics = compute_alignment_quality_metrics(gt_text, htr_text)
    return {
        "substitutions": metrics["substitutions"],
        "deletions": metrics["deletions"],
        "insertions": metrics["insertions"],
        "edits": metrics["edits"],
        "norm_len": metrics["norm_len"],
        "matches": metrics["matches"],
    }


def compute_wer_counts(gt_text: str, htr_text: str) -> dict[str, Any]:
    gt_tokens = normalise_tokens_for_wer(gt_text)
    htr_tokens = normalise_tokens_for_wer(htr_text)

    s = d = i = 0
    for op in Levenshtein.editops(gt_tokens, htr_tokens):
        tag, _, _ = _editop_fields(op)
        if tag == "replace":
            s += 1
        elif tag == "delete":
            d += 1
        elif tag == "insert":
            i += 1

    edits = s + d + i
    return {
        "wer_norm": safe_div(edits, len(gt_tokens)),
        "wer_edits": edits,
        "ref_tokens_norm": len(gt_tokens),
    }


# ---------------------------------------------------------------------
# Boundary metrics
# ---------------------------------------------------------------------

def tokenise_for_boundary(text: str) -> list[str]:
    return text.split()


def boundary_norm(token: str) -> str:
    return "".join(token.split()).casefold()


def detect_boundary_events_for_doc(
    gt_text: str,
    htr_text: str,
    max_span: int = MAX_BOUNDARY_NGRAM,
) -> dict[str, Any]:
    gt_tokens = tokenise_for_boundary(gt_text)
    htr_tokens = tokenise_for_boundary(htr_text)

    i = j = 0
    split_count = merge_count = complex_count = 0
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
# Document metrics and risk scoring
# ---------------------------------------------------------------------

def compute_doc_metrics(
    train_pairs: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    doc_issue_lookup = build_doc_issue_lookup(issues)
    line_lookup = load_line_count_diagnostics()
    rows: list[dict[str, Any]] = []

    for pair in train_pairs:
        doc_id = pair["id"]
        style = pair["style"]

        gt_text = read_text(Path(pair["gt_path"]))
        htr_text = read_text(Path(pair["htr_path"]))

        line_count = len(_splitlines_nonempty(gt_text))
        token_count = len(normalise_tokens_for_wer(gt_text))

        alignment = compute_alignment_quality_metrics(gt_text, htr_text)
        wer = compute_wer_counts(gt_text, htr_text)
        boundary = detect_boundary_events_for_doc(gt_text, htr_text)

        doc_issues = doc_issue_lookup.get(doc_id, [])
        anomaly = issue_anomaly_counts(doc_issues, alignment["norm_len"])
        line_metrics = line_structure_metrics(pair, line_lookup)
        boundary_burden = safe_div(boundary["boundary_events"], alignment["edits"])

        rows.append({
            "doc_id": doc_id,
            "style": style,
            "ref_chars_norm": alignment["norm_len"],
            "token_count": token_count,
            "line_count": line_count,
            "gt_adjusted_line_count": line_metrics["gt_adjusted_line_count"],
            "htr_adjusted_line_count": line_metrics["htr_adjusted_line_count"],
            "adjusted_line_delta": line_metrics["adjusted_line_delta"],
            "line_count_ratio": line_metrics["line_count_ratio"],
            "unmatched_htr_blank_count": line_metrics["unmatched_htr_blank_count"],
            "unmatched_htr_blank_density": line_metrics["unmatched_htr_blank_density"],

            "basic_anomaly_count": anomaly["basic_anomaly_count"],
            "orthographic_violation_count": anomaly["orthographic_violation_count"],
            "basic_anomaly_density": anomaly["basic_anomaly_density"],
            "orthographic_violation_density": anomaly["orthographic_violation_density"],

            "cer_norm": alignment["cer_norm"],
            "wer_norm": wer["wer_norm"],
            "structural_drift_ratio": alignment["structural_drift_ratio"],

            "indel_disruption_rate": alignment["indel_disruption_rate"],
            "coverage": alignment["coverage"],
            "continuity": alignment["continuity"],
            "lasr": alignment["lasr"],
            "fragmentation": alignment["fragmentation"],

            "window_max_cer": alignment["window_max_cer"],
            "window_sd_cer": alignment["window_sd_cer"],
            "window_max_sdr": alignment["window_max_sdr"],

            "boundary_burden_proportion": boundary_burden,

            "edits": alignment["edits"],
            "matches": alignment["matches"],
            "substitutions": alignment["substitutions"],
            "deletions": alignment["deletions"],
            "insertions": alignment["insertions"],
            "wer_edits": wer["wer_edits"],
            "ref_tokens_norm": wer["ref_tokens_norm"],
            "boundary_events": boundary["boundary_events"],
            "split_count": boundary["splits"],
            "merge_count": boundary["merges"],
            "complex_boundary": boundary["complex_boundary"],
            "logged_issues": len(doc_issues),
            "S1_issues": anomaly["basic_anomaly_count"],
            "S3_issues": anomaly["orthographic_violation_count"],
            "window_mode": alignment["window_mode"],
            "window_size": alignment["window_size"],
            "window_count": alignment["window_count"],
        })

    add_risk_features(rows)
    return rows


def add_risk_features(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    for row in rows:
        row["coverage_risk"] = 1.0 - row["coverage"]
        row["continuity_risk"] = 1.0 - row["continuity"]
        row["lasr_risk"] = 1.0 - row["lasr"]

    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_style[row["style"]].append(row)

    for style_rows in by_style.values():
        for metric in RISK_WEIGHTS:
            values = [float(r.get(metric, 0.0)) for r in style_rows]
            med = median_or_zero(values)
            mad = median_absolute_deviation(values)
            z_col = f"{metric}_z"
            for row in style_rows:
                row[z_col] = robust_z(float(row.get(metric, 0.0)), med, mad)

    for row in rows:
        row["risk_score"] = sum(
            row.get(f"{metric}_z", 0.0) * weight
            for metric, weight in RISK_WEIGHTS.items()
        )

    ranked = sorted(rows, key=lambda r: (-r["risk_score"], r["style"], r["doc_id"]))
    for rank, row in enumerate(ranked, start=1):
        row["risk_rank"] = rank

    for style_rows in by_style.values():
        style_ranked = sorted(style_rows, key=lambda r: (-r["risk_score"], r["doc_id"]))
        n = len(style_ranked)
        for idx, row in enumerate(style_ranked):
            frac = idx / n if n else 0.0
            if frac < 1 / 3:
                row["risk_band"] = "High"
            elif frac < 2 / 3:
                row["risk_band"] = "Medium"
            else:
                row["risk_band"] = "Low"


def operational_metric_columns(include_z: bool = False) -> list[str]:
    base_cols = [
        "doc_id",
        "style",
        "ref_chars_norm",
        "token_count",
        "line_count",
        "gt_adjusted_line_count",
        "htr_adjusted_line_count",
        "adjusted_line_delta",
        "line_count_ratio",
        "unmatched_htr_blank_count",
        "unmatched_htr_blank_density",
        "basic_anomaly_density",
        "orthographic_violation_density",
        "cer_norm",
        "wer_norm",
        "structural_drift_ratio",
        "indel_disruption_rate",
        "coverage",
        "continuity",
        "lasr",
        "fragmentation",
        "window_max_cer",
        "window_sd_cer",
        "window_max_sdr",
        "boundary_burden_proportion",
        "risk_score",
        "risk_rank",
        "risk_band",
    ]
    if not include_z:
        return base_cols
    return base_cols + [f"{metric}_z" for metric in RISK_WEIGHTS]


# ---------------------------------------------------------------------
# Style summaries and optional plot data
# ---------------------------------------------------------------------

def aggregate_style_metrics(doc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_rows:
        by_style[row["style"]].append(row)

    corpus_summary = {
        "docs": len(doc_rows),
        "ref_chars_norm": sum(r["ref_chars_norm"] for r in doc_rows),
        "tokens": sum(r["token_count"] for r in doc_rows),
        "lines": sum(r["line_count"] for r in doc_rows),
        "mean_cer_norm": mean_or_zero([r["cer_norm"] for r in doc_rows]),
        "median_cer_norm": median_or_zero([r["cer_norm"] for r in doc_rows]),
        "mean_wer_norm": mean_or_zero([r["wer_norm"] for r in doc_rows]),
        "mean_risk_score": mean_or_zero([r["risk_score"] for r in doc_rows]),
    }

    style_rows = []
    for style in sorted(by_style):
        docs = by_style[style]
        style_rows.append({
            "style": style,
            "docs": len(docs),
            "ref_chars_norm": sum(r["ref_chars_norm"] for r in docs),
            "tokens": sum(r["token_count"] for r in docs),
            "lines": sum(r["line_count"] for r in docs),
            "median_adjusted_line_delta": median_or_zero([r["adjusted_line_delta"] for r in docs]),
            "p90_adjusted_line_delta": percentile([r["adjusted_line_delta"] for r in docs], 0.90),
            "median_unmatched_htr_blank_density": median_or_zero([r["unmatched_htr_blank_density"] for r in docs]),
            "median_cer": median_or_zero([r["cer_norm"] for r in docs]),
            "p90_cer": percentile([r["cer_norm"] for r in docs], 0.90),
            "median_wer": median_or_zero([r["wer_norm"] for r in docs]),
            "p90_wer": percentile([r["wer_norm"] for r in docs], 0.90),
            "median_sdr": median_or_zero([r["structural_drift_ratio"] for r in docs]),
            "median_idr": median_or_zero([r["indel_disruption_rate"] for r in docs]),
            "median_coverage": median_or_zero([r["coverage"] for r in docs]),
            "median_continuity": median_or_zero([r["continuity"] for r in docs]),
            "median_risk_score": median_or_zero([r["risk_score"] for r in docs]),
            "high_risk_docs": sum(1 for r in docs if r["risk_band"] == "High"),
            "medium_risk_docs": sum(1 for r in docs if r["risk_band"] == "Medium"),
            "low_risk_docs": sum(1 for r in docs if r["risk_band"] == "Low"),
        })

    return {
        "corpus_summary": corpus_summary,
        "style_rows": style_rows,
    }


def style_distribution_plot_data(
    doc_rows: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    if metrics is None:
        metrics = [
            "cer_norm",
            "wer_norm",
            "structural_drift_ratio",
            "indel_disruption_rate",
            "coverage",
            "continuity",
            "lasr",
            "fragmentation",
            "adjusted_line_delta",
            "unmatched_htr_blank_density",
            "risk_score",
        ]

    out: dict[str, dict[str, list[float]]] = {
        metric: defaultdict(list) for metric in metrics
    }

    for row in doc_rows:
        style = row["style"]
        for metric in metrics:
            if metric in row:
                out[metric][style].append(float(row[metric]))

    return {metric: dict(style_map) for metric, style_map in out.items()}


# ---------------------------------------------------------------------
# Optional confusion helpers retained for appendix/internal use
# ---------------------------------------------------------------------

def char_confusions_by_style(
    issues: list[dict[str, Any]],
    top_n: int = 15,
) -> dict[str, list[dict[str, Any]]]:
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
        out[style] = [
            {
                "style": style,
                "gt": g,
                "htr": h,
                "count": count,
                "pct_style_char_confusions": safe_div(count, totals[style]),
            }
            for (g, h), count in table[style].most_common(top_n)
        ]
    return out


def bigram_confusions_by_style(
    issues: list[dict[str, Any]],
    top_n: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    table = defaultdict(Counter)
    totals = defaultdict(int)

    for issue in issues:
        if issue.get("tag") != "S2X":
            continue
        style = issue["style"]
        gt = re.sub(r"\s+", "", (issue.get("gt_text") or ""))
        htr = re.sub(r"\s+", "", (issue.get("htr_text") or ""))
        if len(gt) != 2 or not htr:
            continue
        table[style][(gt, htr)] += 1
        totals[style] += 1

    out: dict[str, list[dict[str, Any]]] = {}
    for style in sorted(table):
        out[style] = [
            {
                "style": style,
                "gt_bigram": gt_bg,
                "htr_out": htr_out,
                "count": count,
                "pct_style_bigram_confusions": safe_div(count, totals[style]),
            }
            for (gt_bg, htr_out), count in table[style].most_common(top_n)
        ]
    return out


def word_confusions_by_style(
    issues: list[dict[str, Any]],
    stopwords: set[str],
    top_n: int = 20,
    min_len: int = MIN_WORD_CONFUSION_LEN,
) -> dict[str, list[dict[str, Any]]]:
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
        out[style] = [
            {
                "style": style,
                "gt_word": gt_word,
                "htr_word": htr_word,
                "count": count,
                "pct_style_word_confusions": safe_div(count, totals[style]),
            }
            for (gt_word, htr_word), count in table[style].most_common(top_n)
        ]
    return out


# ---------------------------------------------------------------------
# Document ranking helpers
# ---------------------------------------------------------------------

def top_documents_overall(
    doc_rows: list[dict[str, Any]],
    top_n: int = 25,
) -> list[dict[str, Any]]:
    return sorted(
        doc_rows,
        key=lambda r: (-r["risk_score"], r["style"], r["doc_id"]),
    )[:top_n]


def per_style_document_blocks(
    doc_rows: list[dict[str, Any]],
    top_n: int = 10,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_style = defaultdict(list)
    for row in doc_rows:
        by_style[row["style"]].append(row)

    out = {}
    for style in sorted(by_style):
        docs = by_style[style]
        out[style] = {
            "top_risk": sorted(docs, key=lambda r: (-r["risk_score"], r["doc_id"]))[:top_n],
            "top_cer": sorted(docs, key=lambda r: (-r["cer_norm"], r["doc_id"]))[:top_n],
            "top_structural_drift": sorted(docs, key=lambda r: (-r["structural_drift_ratio"], r["doc_id"]))[:top_n],
            "lowest_coverage": sorted(docs, key=lambda r: (r["coverage"], r["doc_id"]))[:top_n],
            "lowest_continuity": sorted(docs, key=lambda r: (r["continuity"], r["doc_id"]))[:top_n],
        }

    return out
