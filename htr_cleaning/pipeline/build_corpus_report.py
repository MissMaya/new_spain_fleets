"""
build_corpus_report_v7.py

Generates a revised HTR Corpus Diagnostics Report for the New Spain Fleets corpus.

This version combines the strengths of the refactored report and the legacy report, and consumes the relative-position fields now written by processing.py:

- keeps the newer document-level operational/risk framework;
- pares the risk index back to four defensible operational metrics;
- restores corpus-shape, style-difference, clean-subset, concentration, and confusion analyses;
- adds lightweight visual summaries designed for interpretation rather than dashboard complexity;
- adds recurrent confusion-context inventories for top character and bigram confusions;
- adds structural topology analysis comparing substitution, deletion, and insertion density by logged relative line/document position;
- incorporates GT/HTR adjusted line-count diagnostics as line-structure integrity signals;
- separates general metric definitions from the operational risk-index method by adding Appendix B.

The report is intended to support:

- corpus description;
- style-aware comparison;
- error-ecology analysis;
- style-aware corpus governance;
- downstream Document Status Index construction;
- ML/RAG-facing stability awareness.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from html import escape
from typing import Any, Iterable

from utils.config import LOGS_DIR
from utils.file_io import load_json_if_exists, safe_write_text
from utils.report_metrics import *
from pipeline.report_html import *


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

POSTHOC_DIR = LOGS_DIR / "posthoc"
TABLE_DIR = POSTHOC_DIR / "corpus_report_tables"
LINE_COUNT_DIR = LOGS_DIR / "line_counts"

CORPUS_NAME = "New Spain Fleets"
PIPELINE_VERSION = "v8.0-topology-operation-parser-fix"
REPORT_TITLE = "HTR Corpus Diagnostics Report"

# The operational risk index is deliberately narrower than the full
# operational document metrics table. The wider table remains useful for
# analysis, but the index should be interpretable and defensible.
# Diagnostic signals such as WER, anomaly density, orthographic violation
# density, confusions, topology, and line-structure bands remain in the report,
# but they do not contribute directly to the composite score.
RISK_INDEX_METRICS: dict[str, float] = {
    # Overall character-level transcription disagreement.
    "cer_norm": 0.35,
    # Insertion/deletion-driven alignment disruption.
    "indel_disruption_rate": 0.30,
    # Fragmentation of coherent aligned text, computed as 1 - continuity.
    "continuity_risk": 0.25,
    # GT/HTR line-structure disagreement after blank-line handling.
    "adjusted_line_delta": 0.10,
}

# Visualisation defaults
PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>'

# Configurable report limits. These are deliberately conservative defaults:
# enough to expose recurrent behaviour, not enough to swamp the report.
TOP_N_CONFUSIONS = 10
TOP_N_CONTEXTS = 10
TOPOLOGY_BINS = 10
TOPOLOGY_MIN_EVENTS = 1

# Line-structure bands use adjusted GT/HTR line deltas after blank-line handling.
# They are deliberately simple operational categories for ML/RAG triage.
LINE_STABLE_MAX = 0
LINE_MINOR_MAX = 1
LINE_MODERATE_MAX = 3

# Governance escalation thresholds. These are deliberately simple, document-level
# structural danger signals. They do not replace the style-relative risk band;
# they only escalate the final status when a document is structurally dangerous.
VERY_LOW_CONTINUITY_THRESHOLD = 0.15
HIGH_IDR_THRESHOLD = 0.15

GOVERNANCE_STATUS_ORDER = ["Stable", "Review", "Exclude"]
RISK_BAND_TO_GOVERNANCE_STATUS = {
    "Low": "Stable",
    "Medium": "Review",
    "High": "Exclude",
}



# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def write_csv_table(
    name: str,
    headers: list[str],
    rows: list[list[object]],
) -> None:
    """Write a CSV companion file for any rendered table."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{name}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_ready_rows(rows))


def write_dict_csv(
    name: str,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    """Write unformatted dictionary rows to CSV."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{name}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def f_metric(value: float | int | None, digits: int = 6) -> str:
    return f_float(float(value or 0), digits)


def f_density(value: float | int | None) -> str:
    return f_float(float(value or 0), 3)


def f_score(value: float | int | None) -> str:
    return f_float(float(value or 0), 3)


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


# ---------------------------------------------------------------------
# Line-count / line-structure diagnostics
# ---------------------------------------------------------------------

def load_jsonl_if_exists(path) -> list[dict[str, Any]]:
    """Load a JSONL file if present; return [] if absent or unreadable."""
    try:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return []


def line_structure_band(delta_abs: float | int | None) -> str:
    """Categorise adjusted GT/HTR line-count mismatch for operational use."""
    d = abs(number(delta_abs))
    if d <= LINE_STABLE_MAX:
        return "Stable"
    if d <= LINE_MINOR_MAX:
        return "Minor mismatch"
    if d <= LINE_MODERATE_MAX:
        return "Moderate instability"
    return "Severe instability"


def line_structure_flag(delta_abs: float | int | None, unmatched_htr_blank_count: float | int | None = None) -> str:
    """Return a compact ML/RAG-facing flag from line-structure signals."""
    d = abs(number(delta_abs))
    blanks = number(unmatched_htr_blank_count)
    if d > LINE_MODERATE_MAX:
        return "review_line_structure"
    if d > LINE_MINOR_MAX or blanks > 0:
        return "monitor_line_structure"
    return "line_structure_ok"


def load_line_count_diagnostics() -> list[dict[str, Any]]:
    """Load optional output from run_line_count_diagnostics.py."""
    return load_jsonl_if_exists(LINE_COUNT_DIR / "line_count_diagnostics.jsonl")


def merge_line_count_diagnostics(doc_rows: list[dict[str, Any]], line_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add adjusted line-count diagnostics to document rows.

    The line-count diagnostic script uses HTR filename stem as doc_id. Depending
    on upstream pairing, the report document id may be an HTR stem, a pair id,
    or a compound id. We therefore index line diagnostics by several stable keys
    and fall back gracefully if no match is found.
    """
    index: dict[str, dict[str, Any]] = {}
    for lr in line_rows:
        for key in [
            lr.get("doc_id"),
            lr.get("filename"),
            lr.get("pair_id"),
        ]:
            if key:
                index[str(key)] = lr
        filename = str(lr.get("filename", ""))
        if filename:
            index[filename.rsplit(".", 1)[0]] = lr

    for row in doc_rows:
        candidates = [
            row.get("doc_id"),
            row.get("filename"),
            row.get("pair_id"),
            row.get("id"),
        ]
        lr = None
        for cand in candidates:
            if cand and str(cand) in index:
                lr = index[str(cand)]
                break
        if lr is None:
            # Loose fallback: line diagnostic doc_id often appears inside a
            # longer compound document id in report rows.
            doc_id = str(row.get("doc_id", ""))
            for key, candidate in index.items():
                if key and (key in doc_id or doc_id in key):
                    lr = candidate
                    break

        if lr:
            adjusted_delta = number(lr.get("adjusted_delta_gt_minus_htr"))
            adjusted_abs = abs(adjusted_delta)
            row["gt_original_line_count"] = int(number(lr.get("gt_original_line_count")))
            row["htr_original_line_count"] = int(number(lr.get("htr_original_line_count")))
            row["original_delta_gt_minus_htr"] = int(number(lr.get("original_delta_gt_minus_htr")))
            row["gt_adjusted_line_count"] = int(number(lr.get("gt_adjusted_line_count")))
            row["htr_adjusted_line_count"] = int(number(lr.get("htr_adjusted_line_count")))
            row["adjusted_delta_gt_minus_htr"] = int(adjusted_delta)
            row["adjusted_line_delta_abs"] = int(adjusted_abs)
            row["gt_blank_removed_count"] = int(number(lr.get("gt_blank_removed_count")))
            row["htr_blank_removed_matching_gt_count"] = int(number(lr.get("htr_blank_removed_matching_gt_count")))
            row["htr_blank_preserved_unmatched_count"] = int(number(lr.get("htr_blank_preserved_unmatched_count")))
            # Use the external line diagnostic as the authoritative value for
            # the index-facing line delta when available.
            row["adjusted_line_delta"] = int(adjusted_abs)
        else:
            adjusted_abs = abs(number(row.get("adjusted_line_delta", 0)))
            row["adjusted_line_delta_abs"] = int(adjusted_abs)
            row.setdefault("htr_blank_preserved_unmatched_count", int(number(row.get("unmatched_htr_blank_count", 0))))

        row["line_structure_band"] = line_structure_band(row.get("adjusted_line_delta_abs"))
        row["line_structure_flag"] = line_structure_flag(
            row.get("adjusted_line_delta_abs"),
            row.get("htr_blank_preserved_unmatched_count", row.get("unmatched_htr_blank_count", 0)),
        )
    return doc_rows


def line_structure_style_summary(doc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise adjusted line-count risk by style."""
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_rows:
        by_style[str(row.get("style", "UNKNOWN"))].append(row)

    out: list[dict[str, Any]] = []
    for style, rows in sorted(by_style.items()):
        deltas = sorted(abs(number(r.get("adjusted_line_delta_abs", r.get("adjusted_line_delta", 0)))) for r in rows)
        docs = len(rows)
        def pct_band(name: str) -> float:
            return safe_div(sum(1 for r in rows if r.get("line_structure_band") == name), docs)
        p90 = deltas[min(len(deltas)-1, math.ceil(0.9*len(deltas))-1)] if deltas else 0.0
        out.append({
            "style": style,
            "docs": docs,
            "median_adjusted_line_delta_abs": statistics.median(deltas) if deltas else 0.0,
            "p90_adjusted_line_delta_abs": p90,
            "stable_pct": pct_band("Stable"),
            "minor_pct": pct_band("Minor mismatch"),
            "moderate_pct": pct_band("Moderate instability"),
            "severe_pct": pct_band("Severe instability"),
            "docs_with_unmatched_htr_blanks_pct": safe_div(sum(1 for r in rows if number(r.get("htr_blank_preserved_unmatched_count")) > 0), docs),
        })
    return out


def line_structure_style_rows(rows: list[dict[str, Any]]) -> list[list[object]]:
    out: list[list[object]] = []
    for row in rows:
        out.append([
            row.get("style", ""),
            f_int(row.get("docs", 0)),
            f_metric(row.get("median_adjusted_line_delta_abs", 0), 2),
            f_metric(row.get("p90_adjusted_line_delta_abs", 0), 2),
            f_pct(row.get("stable_pct", 0)),
            f_pct(row.get("minor_pct", 0)),
            f_pct(row.get("moderate_pct", 0)),
            f_pct(row.get("severe_pct", 0)),
            f_pct(row.get("docs_with_unmatched_htr_blanks_pct", 0)),
        ])
    return out


def line_structure_document_rows(doc_rows: list[dict[str, Any]]) -> list[list[object]]:
    ordered = sorted(
        doc_rows,
        key=lambda r: (
            str(r.get("style", "")),
            -number(r.get("adjusted_line_delta_abs", r.get("adjusted_line_delta", 0))),
            number(r.get("risk_rank_within_style", 999999)),
        ),
    )
    out: list[list[object]] = []
    for row in ordered:
        if line_structure_band(row.get("adjusted_line_delta_abs", 0)) == "Stable" and number(row.get("htr_blank_preserved_unmatched_count", 0)) == 0:
            continue
        out.append([
            row.get("doc_id", ""),
            row.get("style", ""),
            f_int(row.get("gt_adjusted_line_count", row.get("line_count", 0))),
            f_int(row.get("htr_adjusted_line_count", 0)),
            f_int(row.get("adjusted_delta_gt_minus_htr", 0)),
            f_int(row.get("adjusted_line_delta_abs", row.get("adjusted_line_delta", 0))),
            row.get("line_structure_band", line_structure_band(row.get("adjusted_line_delta_abs", 0))),
            f_int(row.get("htr_blank_preserved_unmatched_count", 0)),
            row.get("line_structure_flag", ""),
            row.get("risk_band", ""),
            f_score(row.get("risk_score", 0)),
        ])
    return out


def line_structure_bars(style_rows: list[dict[str, Any]]) -> str:
    styles = [r.get("style", "") for r in style_rows]
    traces = [
        {"type": "bar", "name": "Stable", "x": styles, "y": [number(r.get("stable_pct"))*100 for r in style_rows]},
        {"type": "bar", "name": "Minor mismatch", "x": styles, "y": [number(r.get("minor_pct"))*100 for r in style_rows]},
        {"type": "bar", "name": "Moderate instability", "x": styles, "y": [number(r.get("moderate_pct"))*100 for r in style_rows]},
        {"type": "bar", "name": "Severe instability", "x": styles, "y": [number(r.get("severe_pct"))*100 for r in style_rows]},
    ]
    layout = {
        "title": "Line-structure bands by style",
        "barmode": "stack",
        "xaxis": {"title": "Style"},
        "yaxis": {"title": "% of documents", "ticksuffix": "%"},
        "legend": {"orientation": "h"},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 80},
    }
    return plotly_div("line_structure_bands_by_style", traces, layout)


# ---------------------------------------------------------------------
# Robust operational risk index
# ---------------------------------------------------------------------

def median_abs_deviation(values: list[float]) -> float:
    """Return MAD around the median. Falls back to 1 for degenerate groups."""
    clean = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not clean:
        return 1.0
    med = statistics.median(clean)
    deviations = [abs(v - med) for v in clean]
    mad = statistics.median(deviations) if deviations else 0.0
    return mad if mad > 0 else 1.0


def robust_z(value: float, median: float, mad: float) -> float:
    return (value - median) / (mad if mad else 1.0)


def metric_value_for_index(row: dict[str, Any], metric: str) -> float:
    """Return the metric value used by the simplified operational index."""
    if metric == "continuity_risk":
        return 1.0 - number(row.get("continuity"), 0.0)
    if metric == "adjusted_line_delta":
        return abs(number(row.get("adjusted_line_delta_abs", row.get("adjusted_line_delta", 0)), 0.0))
    return number(row.get(metric), 0.0)


def apply_simplified_risk_index(doc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Recompute risk_score, risk_rank, and risk_band using the simplified
    style-relative index.  This deliberately overrides any wider index created
    inside utils.report_metrics.compute_doc_metrics.
    """
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_rows:
        by_style[str(row.get("style", "UNKNOWN"))].append(row)

    for style, rows in by_style.items():
        stats: dict[str, tuple[float, float]] = {}
        for metric in RISK_INDEX_METRICS:
            vals = [metric_value_for_index(r, metric) for r in rows]
            med = statistics.median(vals) if vals else 0.0
            mad = median_abs_deviation(vals)
            stats[metric] = (med, mad)

        for row in rows:
            score = 0.0
            for metric, weight in RISK_INDEX_METRICS.items():
                med, mad = stats[metric]
                z = robust_z(metric_value_for_index(row, metric), med, mad)
                row[f"risk_index_{metric}_z"] = z
                score += z * weight
            row["risk_score"] = score

        ordered = sorted(rows, key=lambda r: number(r.get("risk_score")), reverse=True)
        n = len(ordered)
        for idx, row in enumerate(ordered, start=1):
            row["risk_rank_within_style"] = idx
            if n <= 3:
                band = "High" if idx == 1 else ("Medium" if idx == 2 else "Low")
            elif idx <= math.ceil(n / 3):
                band = "High"
            elif idx <= math.ceil((2 * n) / 3):
                band = "Medium"
            else:
                band = "Low"
            row["risk_band"] = band

    global_ordered = sorted(doc_rows, key=lambda r: number(r.get("risk_score")), reverse=True)
    for idx, row in enumerate(global_ordered, start=1):
        row["risk_rank"] = idx

    return doc_rows

# ---------------------------------------------------------------------
# Unified document-status governance
# ---------------------------------------------------------------------

def document_escalation_flags(row: dict[str, Any]) -> list[str]:
    """
    Return structural escalation flags used to convert index-derived risk bands
    into a final document status.

    The flags are deliberately small and plain: they capture structural danger
    signals likely to affect both downstream supervision and retrieval.
    """
    flags: list[str] = []

    if str(row.get("line_structure_band", "")).lower() == "severe instability":
        flags.append("severe_line_instability")
    elif str(row.get("line_structure_flag", "")).lower() == "review_line_structure":
        flags.append("severe_line_instability")

    if number(row.get("continuity"), default=1.0) <= VERY_LOW_CONTINUITY_THRESHOLD:
        flags.append("very_low_continuity")

    if number(row.get("indel_disruption_rate"), default=0.0) >= HIGH_IDR_THRESHOLD:
        flags.append("high_idr")

    return sorted(set(flags))


def escalate_governance_status(base_status: str, flags: list[str]) -> str:
    """Escalate one level when structural danger flags are present."""
    if not flags:
        return base_status
    try:
        idx = GOVERNANCE_STATUS_ORDER.index(base_status)
    except ValueError:
        idx = 1
    return GOVERNANCE_STATUS_ORDER[min(idx + 1, len(GOVERNANCE_STATUS_ORDER) - 1)]


def apply_document_status_governance(doc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add a final, plain-English document status.

    Workflow:
      1. The risk index assigns a style-relative risk band.
      2. Structural danger flags are checked.
      3. The final document status becomes Stable, Review, or Exclude.
    """
    for row in doc_rows:
        risk_band = str(row.get("risk_band", "Medium") or "Medium")
        base_status = RISK_BAND_TO_GOVERNANCE_STATUS.get(risk_band, "Review")
        flags = document_escalation_flags(row)
        row["index_derived_risk_band"] = risk_band
        row["base_governance_status"] = base_status
        row["escalation_flags"] = ";".join(flags)
        row["governance_status"] = escalate_governance_status(base_status, flags)
    return doc_rows


def governance_status_summary_rows(doc_rows: list[dict[str, Any]]) -> list[list[object]]:
    """Summarise final document status by style."""
    by_style: dict[str, Counter] = defaultdict(Counter)
    for row in doc_rows:
        by_style[str(row.get("style", "UNKNOWN"))][str(row.get("governance_status", "Review"))] += 1

    rows: list[list[object]] = []
    for style in sorted(by_style):
        counts = by_style[style]
        total = sum(counts.values())
        rows.append([
            style,
            f_int(total),
            f_int(counts.get("Stable", 0)),
            f_pct(safe_div(counts.get("Stable", 0), total)),
            f_int(counts.get("Review", 0)),
            f_pct(safe_div(counts.get("Review", 0), total)),
            f_int(counts.get("Exclude", 0)),
            f_pct(safe_div(counts.get("Exclude", 0), total)),
        ])
    return rows


def escalation_summary_rows(doc_rows: list[dict[str, Any]]) -> list[list[object]]:
    """Summarise the structural danger flags responsible for status escalation."""
    counts = Counter()
    for row in doc_rows:
        flags = [f for f in str(row.get("escalation_flags", "")).split(";") if f]
        if not flags:
            counts["no_escalation"] += 1
        for flag in flags:
            counts[flag] += 1

    total = len(doc_rows)
    return [[flag, f_int(count), f_pct(safe_div(count, total))] for flag, count in sorted(counts.items())]


def document_status_index_rows(doc_rows: list[dict[str, Any]]) -> list[list[object]]:
    """Compact display rows for the Document Status Index."""
    ordered = sorted(
        doc_rows,
        key=lambda r: (
            str(r.get("style", "")),
            str(r.get("governance_status", "")),
            number(r.get("risk_rank_within_style", 999999)),
        ),
    )
    return [
        [
            r.get("doc_id", ""),
            r.get("style", ""),
            r.get("index_derived_risk_band", r.get("risk_band", "")),
            r.get("escalation_flags", ""),
            r.get("governance_status", ""),
            f_score(r.get("risk_score", 0)),
            f_metric(r.get("cer_norm", 0)),
            f_metric(r.get("indel_disruption_rate", 0)),
            f_metric(r.get("continuity", 0)),
            f_int(r.get("adjusted_line_delta_abs", r.get("adjusted_line_delta", 0))),
            r.get("line_structure_band", ""),
        ]
        for r in ordered
    ]



# ---------------------------------------------------------------------
# Derived summaries used by the new report
# ---------------------------------------------------------------------

def derive_style_rows_from_doc_rows(doc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact style summary directly from document rows."""
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_rows:
        by_style[str(row.get("style", "UNKNOWN"))].append(row)

    out: list[dict[str, Any]] = []
    for style, rows in sorted(by_style.items()):
        def vals(key: str) -> list[float]:
            return [number(r.get(key)) for r in rows]

        def median(key: str) -> float:
            v = vals(key)
            return statistics.median(v) if v else 0.0

        def p90(key: str) -> float:
            v = sorted(vals(key))
            if not v:
                return 0.0
            idx = min(len(v) - 1, math.ceil(0.9 * len(v)) - 1)
            return v[idx]

        docs = len(rows)
        clean_lt_1 = sum(1 for r in rows if number(r.get("cer_norm")) < 0.01)
        clean_lt_2 = sum(1 for r in rows if number(r.get("cer_norm")) < 0.02)
        clean_lt_5 = sum(1 for r in rows if number(r.get("cer_norm")) < 0.05)

        total_sub = sum(number(r.get("substitutions", r.get("S", 0))) for r in rows)
        total_ins = sum(number(r.get("insertions", r.get("I", 0))) for r in rows)
        total_del = sum(number(r.get("deletions", r.get("D", 0))) for r in rows)
        total_edits = total_sub + total_ins + total_del
        if total_edits == 0:
            # fall back where operation counts are not present
            total_edits = sum(number(r.get("edits", 0)) for r in rows)

        out.append({
            "style": style,
            "docs": docs,
            "ref_chars_norm": sum(number(r.get("ref_chars_norm")) for r in rows),
            "tokens": sum(number(r.get("token_count")) for r in rows),
            "lines": sum(number(r.get("line_count")) for r in rows),
            "median_cer": median("cer_norm"),
            "p90_cer": p90("cer_norm"),
            "median_wer": median("wer_norm"),
            "p90_wer": p90("wer_norm"),
            "median_idr": median("indel_disruption_rate"),
            "p90_idr": p90("indel_disruption_rate"),
            "median_continuity": median("continuity"),
            "p90_continuity_risk": p90("continuity_risk"),
            "median_adjusted_line_delta": median("adjusted_line_delta"),
            "p90_adjusted_line_delta": p90("adjusted_line_delta"),
            "stable_line_structure_pct": safe_div(sum(1 for r in rows if r.get("line_structure_band") == "Stable"), docs),
            "severe_line_structure_pct": safe_div(sum(1 for r in rows if r.get("line_structure_band") == "Severe instability"), docs),
            "median_basic_anomaly_density": median("basic_anomaly_density"),
            "median_orthographic_violation_density": median("orthographic_violation_density"),
            "clean_lt_1_pct": safe_div(clean_lt_1, docs),
            "clean_lt_2_pct": safe_div(clean_lt_2, docs),
            "clean_lt_5_pct": safe_div(clean_lt_5, docs),
            "sub_pct": safe_div(total_sub, total_edits),
            "ins_pct": safe_div(total_ins, total_edits),
            "del_pct": safe_div(total_del, total_edits),
            "median_risk_score": median("risk_score"),
            "high_risk_docs": sum(1 for r in rows if r.get("risk_band") == "High"),
            "medium_risk_docs": sum(1 for r in rows if r.get("risk_band") == "Medium"),
            "low_risk_docs": sum(1 for r in rows if r.get("risk_band") == "Low"),
        })
    return out


def concentration_from_doc_rows(doc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_style: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_rows:
        by_style[str(row.get("style", "UNKNOWN"))].append(row)

    out: list[dict[str, Any]] = []
    for style, rows in sorted(by_style.items()):
        edits = [number(r.get("edits", r.get("total_edits", number(r.get("cer_norm")) * number(r.get("ref_chars_norm"))))) for r in rows]
        total = sum(edits)
        ordered = sorted(edits, reverse=True)
        top_10_n = max(1, math.ceil(len(ordered) * 0.10)) if ordered else 0
        top_5_n = min(5, len(ordered))
        out.append({
            "style": style,
            "docs": len(rows),
            "total_edits": total,
            "gini": gini_coefficient(edits),
            "top_10pct_docs_share": safe_div(sum(ordered[:top_10_n]), total),
            "top_5_docs_share": safe_div(sum(ordered[:top_5_n]), total),
        })
    return out


def gini_coefficient(values: list[float]) -> float:
    clean = sorted(v for v in values if v >= 0)
    n = len(clean)
    if n == 0 or sum(clean) == 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for i, value in enumerate(clean, start=1):
        cumulative += value
        weighted_sum += i * value
    return (2 * weighted_sum) / (n * cumulative) - (n + 1) / n


def safe_style_bundle(doc_rows: list[dict[str, Any]], raw_bundle: dict[str, Any]) -> dict[str, Any]:
    """Use available aggregate output, filling gaps from doc rows when needed."""
    bundle = dict(raw_bundle or {})
    bundle["style_rows"] = derive_style_rows_from_doc_rows(doc_rows)
    if "concentration_rows" not in bundle:
        bundle["concentration_rows"] = concentration_from_doc_rows(doc_rows)
    return bundle


# ---------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------

def metadata_rows(
    generated: str,
    train_pairs: list[dict],
    test_pairs: list[dict],
    corpus_summary: dict[str, Any],
    stage_overview: dict[str, Any],
) -> list[list[object]]:
    return [
        ["Generated", generated],
        ["Corpus", CORPUS_NAME],
        ["Pipeline version", PIPELINE_VERSION],
        ["Training documents", f_int(len(train_pairs))],
        ["Test documents", f_int(len(test_pairs))],
        ["Reference characters, normalised", f_int(corpus_summary.get("ref_chars_norm", 0))],
        ["Reference tokens, normalised", f_int(corpus_summary.get("tokens", 0))],
        ["Reference lines", f_int(corpus_summary.get("lines", 0))],
        ["Mean normalised CER", f_float(number(corpus_summary.get("mean_cer_norm")), 6)],
        ["Median normalised CER", f_float(number(corpus_summary.get("median_cer_norm")), 6)],
        ["Mean normalised WER", f_float(number(corpus_summary.get("mean_wer_norm")), 6)],
        ["Logged S1 basic transcription anomalies", f_int(stage_overview.get("S1", 0))],
        ["Logged S3 orthographic violations", f_int(stage_overview.get("S3", 0))],
    ]


def corpus_distribution_rows(dist_rows: list[dict]) -> list[list[object]]:
    return [[row["style"], f_int(row["train_docs"]), f_pct(row["pct_train"])] for row in dist_rows]


def geometry_rows(rows: list[dict]) -> list[list[object]]:
    out = []
    for row in rows:
        out.append([
            row["style"],
            f_int(row["docs"]),
            f_int(row["total_lines"]),
            f_int(row["total_tokens"]),
            f_int(row["total_chars"]),
            f_int(row["doc_lines_p25"]),
            f_int(row["doc_lines_median"]),
            f_int(row["doc_lines_p75"]),
            f_int(row["doc_tokens_p25"]),
            f_int(row["doc_tokens_median"]),
            f_int(row["doc_tokens_p75"]),
            f_int(row["line_len_p25"]),
            f_int(row["line_len_median"]),
            f_int(row["line_len_p75"]),
        ])
    return out


def issue_stage_rows(stage_overview: dict) -> list[list[object]]:
    return [
        ["S1", "Basic transcription / surface anomalies", f_int(stage_overview.get("S1", 0)), f_pct(stage_overview.get("S1_pct", 0))],
        ["S2", "GT-HTR comparison issues", f_int(stage_overview.get("S2", 0)), f_pct(stage_overview.get("S2_pct", 0))],
        ["S3", "Orthographic violations", f_int(stage_overview.get("S3", 0)), f_pct(stage_overview.get("S3_pct", 0))],
        ["TOTAL", "All logged issues", f_int(stage_overview.get("total_issues", 0)), "100.00%"],
    ]


def issue_by_style_rows(rows: list[dict]) -> list[list[object]]:
    out = []
    for row in rows:
        out.append([
            row.get("style", ""),
            f_int(row.get("train_docs", 0)),
            f_pct(row.get("pct_train", 0)),
            f_int(row.get("issues", 0)),
            f_pct(row.get("pct_issues", 0)),
            f_pp(row.get("diff_pp", 0)),
            row.get("weight", ""),
            f_int(row.get("S1", 0)),
            f_int(row.get("S2", 0)),
            f_int(row.get("S3", 0)),
        ])
    return out


def style_intelligence_rows(rows: list[dict]) -> list[list[object]]:
    ordered = sorted(rows, key=lambda r: number(r.get("median_cer")), reverse=True)
    out = []
    for row in ordered:
        out.append([
            row["style"],
            f_int(row["docs"]),
            f_metric(row["median_cer"]),
            f_metric(row["p90_cer"]),
            f_metric(row["median_wer"]),
            f_metric(row["median_idr"]),
            f_metric(row["median_continuity"]),
            f_metric(row["median_adjusted_line_delta"]),
            f_pct(row.get("stable_line_structure_pct", 0)),
            f_pct(row.get("severe_line_structure_pct", 0)),
            f_density(row["median_basic_anomaly_density"]),
            f_density(row["median_orthographic_violation_density"]),
            f_pct(row["clean_lt_1_pct"]),
            f_pct(row["clean_lt_2_pct"]),
            f_pct(row["clean_lt_5_pct"]),
            f_pct(row["sub_pct"]),
            f_pct(row["del_pct"]),
            f_pct(row["ins_pct"]),
        ])
    return out


def concentration_rows(rows: list[dict]) -> list[list[object]]:
    out = []
    for row in rows:
        out.append([
            row["style"],
            f_int(row["docs"]),
            f_int(row["total_edits"]),
            f_float(row["gini"], 4),
            f_pct(row["top_10pct_docs_share"]),
            f_pct(row["top_5_docs_share"]),
        ])
    return out


def risk_band_summary_rows(doc_rows: list[dict]) -> list[list[object]]:
    by_style = defaultdict(Counter)
    for row in doc_rows:
        by_style[row["style"]][row.get("risk_band", "Unassigned")] += 1

    out = []
    for style in sorted(by_style):
        counts = by_style[style]
        total = sum(counts.values())
        out.append([
            style,
            f_int(total),
            f_int(counts.get("High", 0)),
            f_pct(safe_div(counts.get("High", 0), total)),
            f_int(counts.get("Medium", 0)),
            f_pct(safe_div(counts.get("Medium", 0), total)),
            f_int(counts.get("Low", 0)),
            f_pct(safe_div(counts.get("Low", 0), total)),
        ])
    return out


def operational_document_rows(rows: list[dict]) -> list[list[object]]:
    ordered = sorted(rows, key=lambda r: (str(r.get("style", "")), number(r.get("risk_rank_within_style", 999999))))
    out = []
    for row in ordered:
        out.append([
            row.get("doc_id", ""),
            row.get("style", ""),
            f_int(row.get("ref_chars_norm", 0)),
            f_int(row.get("token_count", 0)),
            f_int(row.get("line_count", 0)),
            f_density(row.get("basic_anomaly_density", 0)),
            f_density(row.get("orthographic_violation_density", 0)),
            f_metric(row.get("cer_norm", 0)),
            f_metric(row.get("wer_norm", 0)),
            f_metric(row.get("indel_disruption_rate", 0)),
            f_metric(row.get("continuity", 0)),
            f_int(row.get("adjusted_line_delta_abs", row.get("adjusted_line_delta", 0))),
            row.get("line_structure_band", line_structure_band(row.get("adjusted_line_delta_abs", 0))),
            row.get("line_structure_flag", ""),
            f_score(row.get("risk_score", 0)),
            f_int(row.get("risk_rank_within_style", 0)),
            row.get("risk_band", ""),
            row.get("escalation_flags", ""),
            row.get("governance_status", ""),
        ])
    return out


def top_risk_rows(rows: list[dict], top_n: int = 50) -> list[list[object]]:
    ordered = sorted(rows, key=lambda r: number(r.get("risk_score")), reverse=True)[:top_n]
    out = []
    for row in ordered:
        out.append([
            row.get("style", ""),
            f_int(row.get("risk_rank_within_style", 0)),
            row.get("doc_id", ""),
            row.get("risk_band", ""),
            f_score(row.get("risk_score", 0)),
            f_metric(row.get("cer_norm", 0)),
            f_metric(row.get("indel_disruption_rate", 0)),
            f_metric(row.get("continuity", 0)),
            f_density(row.get("basic_anomaly_density", 0)),
            f_density(row.get("orthographic_violation_density", 0)),
            f_int(row.get("adjusted_line_delta_abs", row.get("adjusted_line_delta", 0))),
            row.get("line_structure_band", line_structure_band(row.get("adjusted_line_delta_abs", 0))),
            row.get("governance_status", ""),
        ])
    return out


def confusion_rows(items: list[dict], kind: str) -> list[list[object]]:
    out = []
    if kind == "char":
        for row in items:
            out.append([row.get("gt", ""), row.get("htr", ""), f_int(row.get("count", 0)), f_pct(row.get("pct_style_char_confusions", 0))])
    elif kind == "bigram":
        for row in items:
            out.append([row.get("gt_bigram", ""), row.get("htr_out", ""), f_int(row.get("count", 0)), f_pct(row.get("pct_style_bigram_confusions", 0))])
    elif kind == "word":
        for row in items:
            out.append([row.get("gt_word", ""), row.get("htr_word", ""), f_int(row.get("count", 0)), f_pct(row.get("pct_style_word_confusions", 0))])
    return out


# ---------------------------------------------------------------------
# Lightweight Plotly visualisations
# ---------------------------------------------------------------------

def plotly_div(div_id: str, traces: list[dict[str, Any]], layout: dict[str, Any]) -> str:
    return f"""
<div class="plot-box">
  <div id="{escape(div_id)}" style="height:420px;"></div>
  <script>
    Plotly.newPlot(
      {json.dumps(div_id)},
      {json.dumps(traces, ensure_ascii=False)},
      {json.dumps(layout, ensure_ascii=False)},
      {{responsive: true}}
    );
  </script>
</div>
"""


def boxplot_by_style(doc_rows: list[dict[str, Any]], metric: str, title: str, y_title: str) -> str:
    by_style: dict[str, list[float]] = defaultdict(list)
    for row in doc_rows:
        by_style[str(row.get("style", "UNKNOWN"))].append(metric_value_for_index(row, metric) if metric == "continuity_risk" else number(row.get(metric)))
    traces = []
    for style in sorted(by_style):
        traces.append({
            "type": "box",
            "name": style,
            "y": by_style[style],
            "boxpoints": "outliers",
        })
    return plotly_div(
        f"box_{metric}",
        traces,
        {"title": title, "yaxis": {"title": y_title}, "margin": {"t": 50, "l": 60, "r": 20, "b": 90}},
    )


def bar_from_style_rows(rows: list[dict[str, Any]], key: str, title: str, y_title: str, pct: bool = True) -> str:
    ordered = sorted(rows, key=lambda r: number(r.get(key)), reverse=True)
    y = [number(r.get(key)) * 100 if pct else number(r.get(key)) for r in ordered]
    traces = [{"type": "bar", "x": [r["style"] for r in ordered], "y": y}]
    return plotly_div(
        f"bar_{key}",
        traces,
        {"title": title, "yaxis": {"title": y_title}, "margin": {"t": 50, "l": 60, "r": 20, "b": 90}},
    )


def grouped_clean_subset_bars(style_rows: list[dict[str, Any]]) -> str:
    styles = [r["style"] for r in sorted(style_rows, key=lambda r: r["style"])]
    traces = [
        {"type": "bar", "name": "CER < 1%", "x": styles, "y": [number(r.get("clean_lt_1_pct")) * 100 for r in sorted(style_rows, key=lambda r: r["style"])]},
        {"type": "bar", "name": "CER < 2%", "x": styles, "y": [number(r.get("clean_lt_2_pct")) * 100 for r in sorted(style_rows, key=lambda r: r["style"])]},
        {"type": "bar", "name": "CER < 5%", "x": styles, "y": [number(r.get("clean_lt_5_pct")) * 100 for r in sorted(style_rows, key=lambda r: r["style"])]},
    ]
    return plotly_div(
        "clean_subset_bars",
        traces,
        {"title": "Clean-subset potential by style", "barmode": "group", "yaxis": {"title": "% of documents"}, "margin": {"t": 50, "l": 60, "r": 20, "b": 90}},
    )


def concentration_bars(concentration: list[dict[str, Any]]) -> str:
    return bar_from_style_rows(concentration, "top_10pct_docs_share", "Error concentration: share of edits in top 10% of documents", "% of edits", pct=True)


def top_char_confusion_bars(char_conf: dict[str, list[dict[str, Any]]], top_n: int = 10) -> str:
    parts = []
    for style in sorted(char_conf):
        rows = char_conf.get(style, [])[:top_n]
        if not rows:
            continue
        labels = [f"{r.get('gt', '')} → {r.get('htr', '')}" for r in rows]
        values = [number(r.get("count")) for r in rows]
        parts.append(plotly_div(
            f"char_conf_{style}",
            [{"type": "bar", "orientation": "h", "x": values, "y": labels}],
            {"title": f"{style}: top character confusions", "xaxis": {"title": "Count"}, "margin": {"t": 50, "l": 120, "r": 20, "b": 60}},
        ))
    return "".join(parts)



# ---------------------------------------------------------------------
# Generic helpers for issue dictionaries
# ---------------------------------------------------------------------

def first_existing(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Return the first non-empty value found in an issue/document row."""
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def norm_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def issue_style(issue: dict[str, Any]) -> str:
    return norm_text(first_existing(issue, ["style", "hand", "script", "style_name"], "UNKNOWN")) or "UNKNOWN"


def issue_operation(issue: dict[str, Any]) -> str | None:
    """
    Normalise issue operation labels to substitution/insertion/deletion.

    Important: processing.py logs Step-2 edit tags as S2I/S2D/S2X, where:
      - S2I = insertion
      - S2D = deletion
      - S2X = substitution / replacement

    Earlier versions of this report parser looked for words such as
    "insert" or suffixes such as "_i", which meant compact tags like S2I
    were not recognised and topology sections incorrectly reported that no
    usable relative-position fields were available.
    """
    # Prefer explicit word/edit operation fields where present.
    op_raw = norm_text(first_existing(
        issue,
        ["word_op", "operation", "op", "edit_type", "type", "kind"],
        "",
    )).lower()

    if op_raw in {"sub", "subs", "substitution", "replace", "replacement", "s", "x"}:
        return "substitution"
    if op_raw in {"ins", "insert", "insertion", "i"}:
        return "insertion"
    if op_raw in {"del", "delete", "deletion", "d"}:
        return "deletion"

    if "sub" in op_raw or "replace" in op_raw:
        return "substitution"
    if "insert" in op_raw:
        return "insertion"
    if "delete" in op_raw or "delet" in op_raw:
        return "deletion"

    # Fall back to compact pipeline tags.
    tag = norm_text(first_existing(issue, ["tag"], "")).upper()

    if tag in {"S2X", "S2R", "S2S"} or tag.endswith("X") or tag.endswith("R"):
        return "substitution"
    if tag in {"S2I"} or tag.endswith("I"):
        return "insertion"
    if tag in {"S2D"} or tag.endswith("D"):
        return "deletion"

    # Avoid treating S1/S3 non-alignment diagnostics as edit operations.
    return None


def issue_gt(issue: dict[str, Any]) -> str:
    return norm_text(first_existing(issue, ["gt", "gt_char", "expected", "source", "ref", "gt_text", "gt_value"], ""))


def issue_htr(issue: dict[str, Any]) -> str:
    return norm_text(first_existing(issue, ["htr", "htr_char", "observed", "target", "hyp", "htr_text", "htr_value"], ""))


def issue_gt_bigram(issue: dict[str, Any]) -> str:
    return norm_text(first_existing(issue, ["gt_bigram", "bigram_gt", "gt_pair", "ref_bigram"], ""))


def issue_htr_bigram(issue: dict[str, Any]) -> str:
    return norm_text(first_existing(issue, ["htr_bigram", "htr_out", "bigram_htr", "htr_pair", "hyp_bigram"], ""))


def issue_token_context(issue: dict[str, Any]) -> str:
    """Extract the most useful token/lexical context available for a confusion."""
    for keys in (
        ["gt_token", "token_gt", "ref_token", "gt_word", "word_gt"],
        ["token", "word", "context_token", "lexical_context"],
        ["htr_token", "token_htr", "htr_word", "word_htr"],
    ):
        val = norm_text(first_existing(issue, keys, ""))
        if val:
            return val

    # Last-resort: use a compact text context if present, but avoid long dumps.
    context = norm_text(first_existing(issue, ["context", "line_context", "gt_line", "line_gt", "line"], ""))
    if context:
        return context[:120]
    return ""


def relative_value(issue: dict[str, Any], kind: str) -> float | None:
    """Extract a 0..1 relative position for line or document topology."""
    if kind == "line":
        keys = [
            "relative_line_position", "line_relative_position", "relative_position_in_line",
            "char_relative_position", "relative_char_position", "line_pos", "rel_line_pos",
            "position_in_line_ratio", "within_line_position",
        ]
        numerator_keys = ["char_index", "char_pos", "position_in_line", "line_char_index", "gt_char_index_in_line"]
        denominator_keys = ["line_length", "gt_line_length", "line_len"]
    else:
        keys = [
            "relative_document_position", "document_relative_position", "relative_position_in_document",
            "doc_pos", "rel_doc_pos", "position_in_document_ratio", "within_document_position",
        ]
        numerator_keys = ["doc_char_index", "document_char_index", "gt_doc_char_index", "global_char_index"]
        denominator_keys = ["doc_length", "document_length", "gt_doc_length", "ref_chars_norm"]

    raw = first_existing(issue, keys, None)
    if raw is not None:
        try:
            v = float(raw)
            if v > 1 and v <= 100:
                v = v / 100.0
            if 0 <= v <= 1:
                return v
        except Exception:
            pass

    numerator = first_existing(issue, numerator_keys, None)
    denominator = first_existing(issue, denominator_keys, None)
    try:
        n = float(numerator)
        d = float(denominator)
        if d > 0:
            v = n / d
            return min(1.0, max(0.0, v))
    except Exception:
        return None
    return None


def position_bin(value: float, bins: int = TOPOLOGY_BINS) -> int:
    """Convert a 0..1 relative position into a zero-based bin index."""
    return min(bins - 1, max(0, int(value * bins)))


def parse_logged_position_bin(raw: Any, bins: int = TOPOLOGY_BINS) -> int | None:
    """
    Read a bin value already logged by processing.py.

    Supports either zero-based integers, one-based integers, or labels such as
    ``0-10%``.  If no usable bin is present, return None so the caller can
    fall back to relative-position values.
    """
    if raw is None:
        return None

    if isinstance(raw, int):
        # Prefer zero-based, but tolerate one-based bins if present.
        if 0 <= raw < bins:
            return raw
        if 1 <= raw <= bins:
            return raw - 1
        return None

    text = str(raw).strip()
    if not text:
        return None

    try:
        val = int(text)
        if 0 <= val < bins:
            return val
        if 1 <= val <= bins:
            return val - 1
    except Exception:
        pass

    # Label form, e.g. "20-30%".
    if "-" in text:
        try:
            start = float(text.split("-", 1)[0].replace("%", "").strip())
            return min(bins - 1, max(0, int((start / 100.0) * bins)))
        except Exception:
            return None

    return None


def issue_position_bin(issue: dict[str, Any], kind: str, bins: int = TOPOLOGY_BINS) -> int | None:
    """
    Prefer the explicit line/document position bins written by processing.py;
    fall back to relative-position fields where needed.
    """
    if kind == "line":
        raw_bin = first_existing(issue, ["line_position_bin", "relative_line_position_bin"], None)
    else:
        raw_bin = first_existing(issue, ["document_position_bin", "doc_position_bin", "relative_document_position_bin"], None)

    parsed = parse_logged_position_bin(raw_bin, bins)
    if parsed is not None:
        return parsed

    pos = relative_value(issue, kind)
    if pos is None:
        return None
    return position_bin(pos, bins)


def bin_label(bin_index: int, bins: int = TOPOLOGY_BINS) -> str:
    start = int((bin_index / bins) * 100)
    end = int(((bin_index + 1) / bins) * 100)
    return f"{start}-{end}%"


# ---------------------------------------------------------------------
# Recurrent confusion context inventories
# ---------------------------------------------------------------------

def selected_confusion_pairs(confusions: dict[str, list[dict[str, Any]]], kind: str, top_n: int) -> dict[str, set[tuple[str, str]]]:
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for style, rows in (confusions or {}).items():
        for row in rows[:top_n]:
            if kind == "char":
                gt = norm_text(row.get("gt"))
                htr = norm_text(row.get("htr"))
            else:
                gt = norm_text(row.get("gt_bigram"))
                htr = norm_text(row.get("htr_out"))
            if gt or htr:
                out[style].add((gt, htr))
    return out


def recurrent_confusion_contexts(
    issues: list[dict[str, Any]],
    char_conf: dict[str, list[dict[str, Any]]],
    bigram_conf: dict[str, list[dict[str, Any]]],
    top_n_confusions: int = TOP_N_CONFUSIONS,
    top_n_contexts: int = TOP_N_CONTEXTS,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    For the top character and bigram confusions, list the tokens/contexts in
    which those confusions occur most often. This is intentionally selective:
    it is a recurrent-context inventory, not a full lexical dump.
    """
    selected_char = selected_confusion_pairs(char_conf, "char", top_n_confusions)
    selected_bigram = selected_confusion_pairs(bigram_conf, "bigram", top_n_confusions)

    counters: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for issue in issues:
        style = issue_style(issue)
        token = issue_token_context(issue)
        if not token:
            continue

        gt = issue_gt(issue)
        htr = issue_htr(issue)
        if (gt, htr) in selected_char.get(style, set()):
            counters[style][f"char::{gt}→{htr}"][token] += 1

        gt_bi = issue_gt_bigram(issue)
        htr_bi = issue_htr_bigram(issue)
        if (gt_bi, htr_bi) in selected_bigram.get(style, set()):
            counters[style][f"bigram::{gt_bi}→{htr_bi}"][token] += 1

    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for style, by_confusion in counters.items():
        for confusion, counter in by_confusion.items():
            out[style][confusion] = [
                {"context": context, "count": count}
                for context, count in counter.most_common(top_n_contexts)
            ]
    return out


def confusion_context_rows(contexts: dict[str, list[dict[str, Any]]]) -> list[list[object]]:
    rows: list[list[object]] = []
    for confusion, items in sorted(contexts.items()):
        kind, pair = confusion.split("::", 1) if "::" in confusion else ("", confusion)
        for item in items:
            rows.append([kind, pair, item["context"], f_int(item["count"])])
    return rows


# ---------------------------------------------------------------------
# Structural topology of errors
# ---------------------------------------------------------------------

def topology_by_style(
    issues: list[dict[str, Any]],
    doc_rows: list[dict[str, Any]],
    kind: str,
    bins: int = TOPOLOGY_BINS,
) -> dict[str, dict[str, list[float]]]:
    """
    Compare substitution/deletion/insertion density by structural position.

    The preferred input is the explicit ``line_position_bin`` or
    ``document_position_bin`` now written by processing.py.  The function also
    tolerates older issue logs by falling back to relative-position fields.

    Returned values are normalised as events per document within style. This
    avoids visually privileging styles that simply have more documents.
    """
    styles = sorted({str(r.get("style", "UNKNOWN")) for r in doc_rows})
    counts: dict[str, dict[str, list[int]]] = {
        style: {op: [0] * bins for op in ["substitution", "deletion", "insertion"]}
        for style in styles
    }

    totals_by_style = Counter(str(r.get("style", "UNKNOWN")) for r in doc_rows)
    usable = 0
    for issue in issues:
        style = issue_style(issue)
        op = issue_operation(issue)
        if op not in {"substitution", "deletion", "insertion"}:
            continue

        b = issue_position_bin(issue, kind, bins)
        if b is None:
            continue

        if style not in counts:
            counts[style] = {o: [0] * bins for o in ["substitution", "deletion", "insertion"]}
        counts[style][op][b] += 1
        usable += 1

    if usable < TOPOLOGY_MIN_EVENTS:
        return {}

    densities: dict[str, dict[str, list[float]]] = {}
    for style, by_op in counts.items():
        denom = max(1, totals_by_style.get(style, 1))
        densities[style] = {op: [v / denom for v in vals] for op, vals in by_op.items()}
    return densities


def topology_rows(topology: dict[str, dict[str, list[float]]], bins: int = TOPOLOGY_BINS) -> list[list[object]]:
    rows: list[list[object]] = []
    for style in sorted(topology):
        for op in ["substitution", "deletion", "insertion"]:
            for i, value in enumerate(topology[style].get(op, [])):
                rows.append([style, op, bin_label(i, bins), f_float(value, 4)])
    return rows


def topology_summary_rows(
    line_topology: dict[str, dict[str, list[float]]],
    document_topology: dict[str, dict[str, list[float]]],
    bins: int = TOPOLOGY_BINS,
) -> list[list[object]]:
    """
    Compact interpretive summary: for each style and operation, identify the
    peak line-position and document-position bins.
    """
    rows: list[list[object]] = []
    styles = sorted(set(line_topology) | set(document_topology))
    for style in styles:
        for op in ["substitution", "deletion", "insertion"]:
            line_vals = line_topology.get(style, {}).get(op, [])
            doc_vals = document_topology.get(style, {}).get(op, [])

            if line_vals:
                line_peak_idx = max(range(len(line_vals)), key=lambda i: line_vals[i])
                line_peak = bin_label(line_peak_idx, bins)
                line_peak_value = f_float(line_vals[line_peak_idx], 4)
            else:
                line_peak = ""
                line_peak_value = ""

            if doc_vals:
                doc_peak_idx = max(range(len(doc_vals)), key=lambda i: doc_vals[i])
                doc_peak = bin_label(doc_peak_idx, bins)
                doc_peak_value = f_float(doc_vals[doc_peak_idx], 4)
            else:
                doc_peak = ""
                doc_peak_value = ""

            if line_peak or doc_peak:
                rows.append([style, op, line_peak, line_peak_value, doc_peak, doc_peak_value])
    return rows


def topology_profiles_json(
    line_topology: dict[str, dict[str, list[float]]],
    document_topology: dict[str, dict[str, list[float]]],
    bins: int = TOPOLOGY_BINS,
) -> dict[str, Any]:
    """Machine-readable topology profile for RAG/fine-tuning handoff."""
    out: dict[str, Any] = {}
    for style in sorted(set(line_topology) | set(document_topology)):
        out[style] = {}
        for op in ["substitution", "deletion", "insertion"]:
            op_profile: dict[str, Any] = {}
            for kind, topo in [("line", line_topology), ("document", document_topology)]:
                vals = topo.get(style, {}).get(op, [])
                if vals:
                    peak_idx = max(range(len(vals)), key=lambda i: vals[i])
                    op_profile[kind] = {
                        "peak_bin": bin_label(peak_idx, bins),
                        "peak_events_per_document": vals[peak_idx],
                        "bins": [
                            {"bin": bin_label(i, bins), "events_per_document": v}
                            for i, v in enumerate(vals)
                        ],
                    }
            if op_profile:
                out[style][op] = op_profile
    return out


def topology_plot(topology: dict[str, dict[str, list[float]]], title: str, div_prefix: str, bins: int = TOPOLOGY_BINS) -> str:
    if not topology:
        return html_note(
            "No usable relative-position fields were found in the issue logs for this topology view. "
            "The section is retained so the report can use it automatically once position metadata is logged."
        )

    labels = [bin_label(i, bins) for i in range(bins)]
    parts: list[str] = []
    for style in sorted(topology):
        traces = []
        for op in ["substitution", "deletion", "insertion"]:
            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "name": op,
                "x": labels,
                "y": topology[style].get(op, [0] * bins),
            })
        parts.append(plotly_div(
            f"{div_prefix}_{style}",
            traces,
            {
                "title": f"{style}: {title}",
                "xaxis": {"title": "Relative position"},
                "yaxis": {"title": "Events per document"},
                "margin": {"t": 50, "l": 70, "r": 20, "b": 80},
            },
        ))
    return "".join(parts)

# ---------------------------------------------------------------------
# Appendices
# ---------------------------------------------------------------------

def appendix_a_metric_definitions() -> str:
    return """
<section id="appendix-a">
<h2>Appendix A. Metric definitions and interpretation</h2>

<p>
This appendix defines the principal metrics used in the report.  The report
computes more diagnostic information than is used in the operational risk index.
This distinction is deliberate: diagnostic metrics help explain corpus behaviour,
whereas the risk index uses a smaller set of metrics for transparent subset
construction.
</p>

<hr>

<h3>Standard transcription disagreement</h3>
<details open>
<summary><strong>Character Error Rate (CER)</strong></summary>
<p><code>CER = (S + I + D) / N</code></p>
<p>
CER measures the proportion of characters that must be edited to transform the
HTR text into the GT text.  <code>S</code> is substitutions, <code>I</code> is
insertions, <code>D</code> is deletions, and <code>N</code> is the number of
normalised GT characters.
</p>
</details>

<details>
<summary><strong>Word Error Rate (WER)</strong></summary>
<p><code>WER = (S_w + I_w + D_w) / N_w</code></p>
<p>
WER measures word-level edit disagreement. It is useful as a higher-level
usability signal, but it is less diagnostic than CER and alignment metrics for
character-level HTR failure.
</p>
</details>

<hr>

<h3>Alignment and drift</h3>
<details open>
<summary><strong>Indel Disruption Rate (IDR)</strong></summary>
<p><code>IDR = (I + D) / (M + S + I + D)</code></p>
<p>
IDR measures the proportion of aligned positions disrupted by insertions or
deletions. It is used as the principal drift metric in the operational index.
</p>
</details>

<details>
<summary><strong>Coverage</strong></summary>
<p><code>Coverage = (M + S) / (M + S + D)</code></p>
<p>
Coverage measures the proportion of GT characters represented in the aligned HTR
output. Low coverage suggests omitted regions or severe alignment failure.
</p>
</details>

<details open>
<summary><strong>Continuity</strong></summary>
<p><code>Continuity = sum(length(span_k ≥ MIN_COHERENT_SPAN_CHARS)) / N</code></p>
<p>
Continuity measures the proportion of a document contained in coherent aligned
spans. Low continuity indicates fragmented alignment and is especially relevant
to model training and RAG chunk stability.
</p>
</details>

<hr>

<h3>Line and surface pathology</h3>
<details open>
<summary><strong>Adjusted line delta</strong></summary>
<p><code>Adjusted_line_delta = abs(gt_adjusted_line_count − htr_adjusted_line_count)</code></p>
<p>
Adjusted line delta measures disagreement between GT and HTR line structure after
blank-line adjustment. It is included in the operational index because line
mismatch may compromise alignment, chunking, and training-pair reliability.
</p>
</details>

<details open>
<summary><strong>Basic anomaly density</strong></summary>
<p><code>Basic_anomaly_density = S1_issue_count / ref_chars_norm × 1000</code></p>
<p>
Basic anomaly density captures surface-level transcription irregularity, such as
unexpected whitespace, punctuation inside words, or non-Latin glyphs, normalised
by document length.
</p>
</details>

<details open>
<summary><strong>Orthographic violation density</strong></summary>
<p><code>Orthographic_violation_density = S3_issue_count / ref_chars_norm × 1000</code></p>
<p>
Orthographic violation density captures suspicious graphemic or orthographic
behaviour, normalised by document length. It is interpreted cautiously because
historical spelling variation and transcription errors may overlap.
</p>
</details>

<hr>

<h3>Diagnostic-only metrics</h3>
<p>
The wider report may also expose diagnostic-only measures such as windowed CER,
fragmentation, LASR, boundary burden, and confusion frequencies. These metrics
are useful for corpus intelligence and forensic analysis, but they are not all
used in the operational risk index.
</p>
</section>
"""


def appendix_b_risk_index() -> str:
    weights_rows = "".join(
        f"<tr><td><code>{escape(metric)}</code></td><td>{weight}</td></tr>"
        for metric, weight in RISK_INDEX_METRICS.items()
    )
    return f"""
<section id="appendix-b">
<h2>Appendix B. Index-Derived Risk Stratification</h2>

<h3>B1. Purpose</h3>
<p>
The index is used to produce an initial, style-relative risk band for each
document. It is not a universal quality score and it is not the final downstream
decision. Its role is to answer a narrower question: which documents are
relatively more or less unstable when compared with other documents in the same
calligraphic style?
</p>

<h3>B2. Metrics used</h3>
<table class="report-table">
<thead><tr><th>Dimension</th><th>Metric</th><th>Plain interpretation</th></tr></thead>
<tbody>
<tr><td>Transcription disagreement</td><td>CER</td><td>How much character-level editing is needed overall.</td></tr>
<tr><td>Drift</td><td>IDR</td><td>How much disruption is caused by insertions and deletions.</td></tr>
<tr><td>Alignment coherence</td><td>Continuity risk</td><td>How fragmented the aligned text has become.</td></tr>
<tr><td>Line structure</td><td>Adjusted line delta</td><td>Whether GT and HTR line structure still broadly agree.</td></tr>
</tbody>
</table>
<p>
Other diagnostics, such as WER, anomaly density, orthographic violations,
confusion inventories, and topology, remain important for interpretation but are
not included in the composite score. This keeps the index small enough to explain
and defend.
</p>

<h3>B3. Standardisation within style</h3>
<p>
Each metric is standardised within style using the style median and median
absolute deviation:
</p>
<p><code>z_robust = (x − median_style) / MAD_style</code></p>
<p>
This prevents generally difficult styles from being automatically treated as
worse than cleaner styles. The index identifies documents that are unusually
unstable for their own style.
</p>

<h3>B4. Weighted score</h3>
<p>
The index score is a weighted sum of the style-standardised metrics:
</p>
<p><code>Risk_score = Σ(z_i × w_i)</code></p>
<table class="report-table">
<thead><tr><th>Metric</th><th>Weight</th></tr></thead>
<tbody>{weights_rows}</tbody>
</table>

<h3>B5. Risk bands</h3>
<p>
Within each style, documents are assigned to Low, Medium, and High index-derived
risk bands using terciles of the risk score. These bands are used for
stratification, review sampling, and the first step of document-status
assignment.
</p>

<h3>B6. From risk band to document status</h3>
<p>
The final Document Status is derived in two steps. First, the index assigns a
style-relative risk band. Second, the document is checked for structural danger
signals: severe line instability, very low continuity, or high IDR. If any of
these are present, the status is escalated by one level.
</p>
<table class="report-table">
<thead><tr><th>Starting point</th><th>No escalation</th><th>With escalation</th></tr></thead>
<tbody>
<tr><td>Low index-derived risk</td><td>Stable</td><td>Review</td></tr>
<tr><td>Medium index-derived risk</td><td>Review</td><td>Exclude</td></tr>
<tr><td>High index-derived risk</td><td>Exclude</td><td>Exclude</td></tr>
</tbody>
</table>
<p>
This gives a single, plain downstream status: Stable, Review, or Exclude.
</p>
</section>
"""

def appendix_c_human_review_protocol() -> str:
    return """
<section id="appendix-c">
<h2>Appendix C. Human Review Protocol</h2>

<h3>C1. Purpose of human review</h3>
<p>
The computational pipeline identifies where transcription instability is likely
to exist, but computational signals alone cannot decide whether a transcription
is still usable for scholarly work or downstream AI systems.
</p>
<p>
Human review therefore checks the practical effect of the instability. Reviewers
are not asked to tag every character error. Instead, they judge whether the HTR
remains reliable, synchronised, structurally coherent, and usable.
</p>

<h3>C2. Construction of the review packets</h3>
<p>
Review packets are constructed from three layers:
</p>
<table class="report-table">
<thead><tr><th>Layer</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td>Stratified review sample</td><td>Samples across style, index-derived risk band, and Document Status so the framework can be validated across the full governance structure.</td></tr>
<tr><td>Random control sample</td><td>Includes documents selected independently of the index to reduce confirmation bias and test whether the framework misses important instability.</td></tr>
<tr><td>Triggered exemplars</td><td>Includes rare or severe cases, such as catastrophic drift or major structural instability, for failure-mode analysis.</td></tr>
</tbody>
</table>

<h3>C3. Reviewer workflow</h3>
<p>
Each reviewer receives the manuscript reference, GT transcription, HTR output,
and a short review sheet. The sheet asks for judgements on overall reliability,
drift/synchronisation, omissions, structural integrity, scholarly usability, and
reviewer confidence.
</p>
<p>
Reviewers are not shown risk scores, CER/WER, or other machine-generated metrics.
This keeps human judgement independent from the computational assessment.
</p>

<h3>C4. Feedback into the process</h3>
<p>
Human review is used to validate and refine the Document Status Index. If human
judgements agree with the assigned statuses, the governance framework is
supported. If reviewers find serious problems in supposedly Stable documents, or
judge many Review/Exclude documents to be usable, the thresholds and escalation
rules can be revised.
</p>
<p>
After review, corrected transcriptions and review judgements can be stored as a
separate reviewed layer. The original HTR is preserved.
</p>
</section>
"""

# ---------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------

def build_report() -> str:
    POSTHOC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    train_pairs = load_json_if_exists(LOGS_DIR / "meta" / "train_pairs.json", [])
    test_pairs = load_json_if_exists(LOGS_DIR / "meta" / "test_pairs.json", [])
    issues = load_all_issues()

    # -----------------------------------------------------------------
    # Compute core data
    # -----------------------------------------------------------------
    dist_rows = train_distribution_by_style(train_pairs)
    geom_rows = geometry_by_style(train_pairs)
    stage_overview = issue_stage_overview(issues)

    try:
        issue_style_raw = issue_distribution_by_style(issues, train_pairs)
    except NameError:
        issue_style_raw = []

    doc_rows = compute_doc_metrics(train_pairs, issues)
    line_count_rows = load_line_count_diagnostics()
    doc_rows = merge_line_count_diagnostics(doc_rows, line_count_rows)
    doc_rows = apply_simplified_risk_index(doc_rows)
    doc_rows = apply_document_status_governance(doc_rows)

    raw_style_bundle = aggregate_style_metrics(doc_rows)
    style_bundle = safe_style_bundle(doc_rows, raw_style_bundle)
    corpus_summary = raw_style_bundle.get("corpus_summary", {})
    style_rows_raw = style_bundle["style_rows"]
    concentration_raw = style_bundle.get("concentration_rows", concentration_from_doc_rows(doc_rows))
    line_structure_raw = line_structure_style_summary(doc_rows)

    # Legacy-style error ecology outputs: use when available.
    try:
        stopwords = load_stopwords()
    except Exception:
        stopwords = []

    try:
        char_conf = char_confusions_by_style(issues, top_n=TOP_N_CONFUSIONS)
    except NameError:
        char_conf = {}
    try:
        bigram_conf = bigram_confusions_by_style(issues, top_n=TOP_N_CONFUSIONS)
    except NameError:
        bigram_conf = {}
    try:
        word_conf = word_confusions_by_style(issues, stopwords=stopwords, top_n=TOP_N_CONFUSIONS)
    except NameError:
        word_conf = {}

    confusion_contexts = recurrent_confusion_contexts(
        issues,
        char_conf,
        bigram_conf,
        top_n_confusions=TOP_N_CONFUSIONS,
        top_n_contexts=TOP_N_CONTEXTS,
    )
    line_topology = topology_by_style(issues, doc_rows, kind="line", bins=TOPOLOGY_BINS)
    document_topology = topology_by_style(issues, doc_rows, kind="document", bins=TOPOLOGY_BINS)

    # -----------------------------------------------------------------
    # CSV outputs
    # -----------------------------------------------------------------
    operational_columns = [
        "doc_id", "style", "ref_chars_norm", "token_count", "line_count",
        "basic_anomaly_density", "orthographic_violation_density", "cer_norm",
        "wer_norm", "indel_disruption_rate", "continuity", "adjusted_line_delta_abs",
        "line_structure_band", "line_structure_flag", "risk_score", "risk_rank_within_style", "risk_band", "escalation_flags", "governance_status",
    ]
    write_dict_csv("operational_document_metrics", doc_rows, operational_columns)

    meta_headers = ["Metric", "Value"]
    meta_table_rows = metadata_rows(generated, train_pairs, test_pairs, corpus_summary, stage_overview)

    dist_headers = ["Style", "Train docs", "% of train corpus"]
    dist_table_rows = corpus_distribution_rows(dist_rows)
    write_csv_table("training_corpus_by_style", dist_headers, dist_table_rows)

    geom_headers = [
        "Style", "Docs", "Total lines", "Total tokens", "Total chars",
        "Doc lines P25", "Doc lines Median", "Doc lines P75",
        "Doc tokens P25", "Doc tokens Median", "Doc tokens P75",
        "Line len P25", "Line len Median", "Line len P75",
    ]
    geom_table_rows = geometry_rows(geom_rows)
    write_csv_table("training_geometry_by_style", geom_headers, geom_table_rows)

    stage_headers = ["Stage", "Meaning", "Count", "% of all issues"]
    stage_table_rows = issue_stage_rows(stage_overview)
    write_csv_table("logged_issue_stage_overview", stage_headers, stage_table_rows)

    style_headers = [
        "Style", "Docs", "Median CER", "P90 CER", "Median WER",
        "Median IDR", "Median continuity", "Median adjusted line delta",
        "Stable line structure %", "Severe line instability %",
        "Median basic anomaly density", "Median orthographic violation density",
        "Clean <1%", "Clean <2%", "Clean <5%",
        "Sub %", "Del %", "Ins %",
    ]
    style_table_rows = style_intelligence_rows(style_rows_raw)
    write_csv_table("style_intelligence_overview", style_headers, style_table_rows)

    concentration_headers = [
        "Style", "Docs", "Total edits", "Gini",
        "Top 10% docs share of edits", "Top 5 docs share of edits",
    ]
    concentration_table_rows = concentration_rows(concentration_raw)
    write_csv_table("error_concentration_by_style", concentration_headers, concentration_table_rows)

    risk_band_headers = ["Style", "Docs", "High", "High %", "Medium", "Medium %", "Low", "Low %"]
    risk_band_rows_out = risk_band_summary_rows(doc_rows)
    write_csv_table("risk_band_summary_by_style", risk_band_headers, risk_band_rows_out)

    status_headers = ["Style", "Docs", "Stable", "Stable %", "Review", "Review %", "Exclude", "Exclude %"]
    status_table_rows = governance_status_summary_rows(doc_rows)
    write_csv_table("document_status_summary_by_style", status_headers, status_table_rows)

    escalation_headers = ["Escalation flag", "Documents", "% of documents"]
    escalation_table_rows = escalation_summary_rows(doc_rows)
    write_csv_table("document_status_escalation_flags", escalation_headers, escalation_table_rows)

    document_status_headers = [
        "doc_id", "style", "Index-derived risk band", "Escalation flags", "Document status",
        "Risk score", "CER", "IDR", "Continuity", "Adjusted line delta", "Line-structure band",
    ]
    document_status_table_rows = document_status_index_rows(doc_rows)
    write_csv_table("document_status_index", document_status_headers, document_status_table_rows)

    op_headers = [
        "doc_id", "style", "ref_chars_norm", "token_count", "line_count",
        "Basic anomaly density", "Orthographic violation density", "CER", "WER",
        "IDR", "Continuity", "Adjusted line delta", "Line-structure band", "Line-structure flag",
        "Risk score", "Risk rank within style", "Risk band", "Escalation flags", "Document status",
    ]
    op_table_rows = operational_document_rows(doc_rows)
    write_csv_table("operational_document_metrics_display", op_headers, op_table_rows)

    top_risk_headers = [
        "Style", "Risk rank within style", "doc_id", "Risk band", "Risk score",
        "CER", "IDR", "Continuity", "Basic anomaly density",
        "Orthographic violation density", "Adjusted line delta", "Line-structure band", "Document status",
    ]
    top_risk_table_rows = top_risk_rows(doc_rows, top_n=50)
    write_csv_table("top_50_documents_by_simplified_risk", top_risk_headers, top_risk_table_rows)

    line_structure_headers = [
        "Style", "Docs", "Median adjusted line delta", "P90 adjusted line delta",
        "Stable %", "Minor mismatch %", "Moderate instability %", "Severe instability %",
        "Docs with unmatched HTR blanks %",
    ]
    line_structure_table_rows = line_structure_style_rows(line_structure_raw)
    write_csv_table("line_structure_by_style", line_structure_headers, line_structure_table_rows)

    line_doc_headers = [
        "doc_id", "style", "GT adjusted lines", "HTR adjusted lines",
        "Adjusted delta GT-HTR", "Absolute adjusted delta", "Line-structure band",
        "Unmatched HTR blank count", "Line-structure flag", "Risk band", "Risk score",
    ]
    line_doc_table_rows = line_structure_document_rows(doc_rows)
    write_csv_table("line_structure_document_flags", line_doc_headers, line_doc_table_rows)

    # Machine-readable report-table exports used by downstream governance scripts.
    rag_quality_columns = [
        "doc_id", "style", "risk_band", "risk_score", "risk_rank_within_style", "escalation_flags", "governance_status",
        "cer_norm", "wer_norm", "indel_disruption_rate", "continuity",
        "adjusted_line_delta_abs", "line_structure_band", "line_structure_flag",
        "basic_anomaly_density", "orthographic_violation_density",
    ]
    write_dict_csv("rag_document_quality", doc_rows, rag_quality_columns)
    with (TABLE_DIR / "rag_confusion_contexts.json").open("w", encoding="utf-8") as f:
        json.dump(confusion_contexts, f, ensure_ascii=False, indent=2)
    with (TABLE_DIR / "rag_style_line_structure_profiles.json").open("w", encoding="utf-8") as f:
        json.dump(line_structure_raw, f, ensure_ascii=False, indent=2)

    # Confusion CSVs
    for style in sorted(set(char_conf) | set(bigram_conf) | set(word_conf)):
        char_rows = confusion_rows(char_conf.get(style, []), "char")
        bigram_rows = confusion_rows(bigram_conf.get(style, []), "bigram")
        word_rows = confusion_rows(word_conf.get(style, []), "word")
        if char_rows:
            write_csv_table(f"{style}_char_confusions", ["GT", "HTR", "Count", "% of style char confusions"], char_rows)
        if bigram_rows:
            write_csv_table(f"{style}_bigram_confusions", ["GT bigram", "HTR output", "Count", "% of style bigram confusions"], bigram_rows)
        if word_rows:
            write_csv_table(f"{style}_word_confusions", ["GT word", "HTR word", "Count", "% of style word confusions"], word_rows)

    for style, contexts in confusion_contexts.items():
        rows = confusion_context_rows(contexts)
        if rows:
            write_csv_table(
                f"{style}_recurrent_confusion_contexts",
                ["Kind", "Confusion", "Token/context", "Count"],
                rows,
            )

    line_topology_table_rows = topology_rows(line_topology, TOPOLOGY_BINS)
    document_topology_table_rows = topology_rows(document_topology, TOPOLOGY_BINS)
    topology_summary_table_rows = topology_summary_rows(line_topology, document_topology, TOPOLOGY_BINS)
    if topology_summary_table_rows:
        write_csv_table(
            "topology_summary_by_style",
            ["Style", "Operation", "Peak line-position bin", "Peak line events/doc", "Peak document-position bin", "Peak document events/doc"],
            topology_summary_table_rows,
        )
    if line_topology_table_rows:
        write_csv_table(
            "topology_line_position_by_style",
            ["Style", "Operation", "Relative position bin", "Events per document"],
            line_topology_table_rows,
        )
    if document_topology_table_rows:
        write_csv_table(
            "topology_document_position_by_style",
            ["Style", "Operation", "Relative position bin", "Events per document"],
            document_topology_table_rows,
        )
    topology_profiles = topology_profiles_json(line_topology, document_topology, TOPOLOGY_BINS)
    with (TABLE_DIR / "rag_topology_profiles.json").open("w", encoding="utf-8") as f:
        json.dump(topology_profiles, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------
    # HTML sections
    # -----------------------------------------------------------------
    sections: list[str] = []

    sections.append(section(
        "Metadata and analytical scope",
        html_note(
            "This report has two layers: a corpus-intelligence layer describing style behaviour and error ecology, "
            "and an operational governance layer for style-aware document status assignment. "
            "The risk-weighted stratification uses a deliberately smaller metric set than the full diagnostics table."
        ) + html_table(meta_headers, meta_table_rows, datatable=False),
        open_by_default=True,
    ))

    corpus_section = (
        subsection(
            "Training corpus composition by style",
            html_note("How to read this section: large styles will dominate aggregate corpus behaviour, so style-level comparisons are needed before making corpus-wide claims.")
            + html_table(dist_headers, dist_table_rows, datatable=True, csv_name="training_corpus_by_style"),
        )
        + subsection("Training corpus geometry by style", html_table(geom_headers, geom_table_rows, datatable=True, csv_name="training_geometry_by_style"))
        + subsection(
            "Logged issue context",
            html_note("How to read this section: if one style contributes more logged issues than its corpus share, that style may require closer scrutiny; however, issue counts are context, not the main quality measure.")
            + html_table(stage_headers, stage_table_rows, datatable=True, csv_name="logged_issue_stage_overview"),
        )
    )
    if issue_style_raw:
        issue_style_headers = ["Style", "Train docs", "% train", "Issues", "% issues", "Issue vs corpus", "Weight", "S1", "S2", "S3"]
        issue_style_table_rows = issue_by_style_rows(issue_style_raw)
        write_csv_table("logged_issues_by_style", issue_style_headers, issue_style_table_rows)
        corpus_section += subsection(
            "Logged issues by style",
            html_note("This contextual view shows whether logged issue burden is over- or under-represented relative to corpus composition.")
            + html_table(issue_style_headers, issue_style_table_rows, datatable=True, csv_name="logged_issues_by_style"),
        )
    sections.append(section("Corpus shape and issue context", corpus_section, open_by_default=True))

    style_section = (
        html_note(
            "How to read this section: compare styles across several signals, not CER alone. "
            "For example, a style with lower CER but low continuity may still be structurally risky for downstream use."
        )
        + subsection("Style intelligence overview", html_table(style_headers, style_table_rows, datatable=True, csv_name="style_intelligence_overview"))
        + subsection("CER distribution by style", boxplot_by_style(doc_rows, "cer_norm", "CER distribution by style", "CER"))
        + subsection("Drift distribution by style", boxplot_by_style(doc_rows, "indel_disruption_rate", "Indel disruption by style", "IDR"))
        + subsection("Alignment coherence by style", boxplot_by_style(doc_rows, "continuity", "Continuity distribution by style", "Continuity"))
        + subsection("Clean-subset potential", grouped_clean_subset_bars(style_rows_raw))
    )
    sections.append(section("Style behaviour and clean-subset potential", style_section, open_by_default=True))

    concentration_section = (
        html_note(
            "How to read this section: high concentration means a small number of documents carry a large share of the error burden. "
            "Those documents are strong candidates for review or exclusion testing even if the style as a whole looks manageable."
        )
        + subsection("Concentration summary by style", html_table(concentration_headers, concentration_table_rows, datatable=True, csv_name="error_concentration_by_style"))
        + subsection("Top-decile edit burden by style", concentration_bars(concentration_raw))
    )
    sections.append(section("Error concentration and high-burden subsets", concentration_section, open_by_default=True))

    line_structure_section = (
        html_note(
            "How to read this section: line-count deltas are structural integrity signals, not bookkeeping. "
            "A severe adjusted line mismatch can indicate unstable segmentation, which may compromise both training supervision and retrieval chunking."
        )
        + subsection("Line-structure bands by style", line_structure_bars(line_structure_raw))
        + subsection("Line-structure summary by style", html_table(line_structure_headers, line_structure_table_rows, datatable=True, csv_name="line_structure_by_style"))
        + subsection(
            "Documents with line-structure flags",
            html_note(
                "This table excludes documents with stable line structure and no unmatched HTR blanks. "
                "Use it as a review queue for segmentation risk, especially when line mismatch co-occurs with high IDR or low continuity."
            )
            + html_table(line_doc_headers, line_doc_table_rows, datatable=True, csv_name="line_structure_document_flags"),
        )
    )
    sections.append(section("Line-structure integrity and segmentation risk", line_structure_section, open_by_default=True))

    error_ecology_parts = [html_note(
        "How to read this section: recurrent confusions show what kinds of textual corruption repeat within styles. "
        "They are diagnostic evidence for later correction, model evaluation, and retrieval normalisation, but they are not part of the risk-weighted stratification."
    )]
    if char_conf:
        error_ecology_parts.append(subsection("Top character confusions as bars", top_char_confusion_bars(char_conf, top_n=TOP_N_CONFUSIONS)))

    for style in sorted(set(char_conf) | set(bigram_conf) | set(word_conf)):
        block = ""
        char_rows = confusion_rows(char_conf.get(style, []), "char")
        bigram_rows = confusion_rows(bigram_conf.get(style, []), "bigram")
        word_rows = confusion_rows(word_conf.get(style, []), "word")
        if char_rows:
            block += subsection(f"{style} — character confusions", html_table(["GT", "HTR", "Count", "% of style char confusions"], char_rows, datatable=True, csv_name=f"{style}_char_confusions"))
        if bigram_rows:
            block += subsection(f"{style} — bigram confusions", html_table(["GT bigram", "HTR output", "Count", "% of style bigram confusions"], bigram_rows, datatable=True, csv_name=f"{style}_bigram_confusions"))
        context_rows = confusion_context_rows(confusion_contexts.get(style, {}))
        if context_rows:
            block += subsection(
                f"{style} — recurrent confusion contexts",
                html_note(
                    "For the top character and bigram confusions, this table lists the tokens or compact text contexts in which the confusion most often occurs. "
                    "This is a selective inventory for downstream fine-tuning and RAG review, not an exhaustive lexical dump."
                )
                + html_table(
                    ["Kind", "Confusion", "Token/context", "Count"],
                    context_rows,
                    datatable=True,
                    csv_name=f"{style}_recurrent_confusion_contexts",
                ),
            )
        if word_rows:
            block += subsection(f"{style} — word confusions", html_table(["GT word", "HTR word", "Count", "% of style word confusions"], word_rows, datatable=True, csv_name=f"{style}_word_confusions"))
        if block:
            error_ecology_parts.append(subsection(f"Style block: {style}", block))

    sections.append(section("Error ecology: character, bigram, word, and recurrent-context behaviour", "".join(error_ecology_parts), open_by_default=False))

    topology_section = (
        html_note(
            "How to read this section: topology shows where errors occur within lines and documents. "
            "If deletions or insertions peak near line endings or late-document regions, the problem is structural and local rather than just a global CER issue."
        )
        + (subsection(
            "Topology summary by style",
            html_table(
                ["Style", "Operation", "Peak line-position bin", "Peak line events/doc", "Peak document-position bin", "Peak document events/doc"],
                topology_summary_table_rows,
                datatable=True,
                csv_name="topology_summary_by_style",
            ),
        ) if topology_summary_table_rows else "")
        + subsection("Operation density by relative line position", topology_plot(line_topology, "operation density by relative line position", "topology_line", TOPOLOGY_BINS))
        + (subsection("Line-position topology table", html_table(["Style", "Operation", "Relative position bin", "Events per document"], line_topology_table_rows, datatable=True, csv_name="topology_line_position_by_style")) if line_topology_table_rows else "")
        + subsection("Operation density by relative document position", topology_plot(document_topology, "operation density by relative document position", "topology_document", TOPOLOGY_BINS))
        + (subsection("Document-position topology table", html_table(["Style", "Operation", "Relative position bin", "Events per document"], document_topology_table_rows, datatable=True, csv_name="topology_document_position_by_style")) if document_topology_table_rows else "")
    )
    sections.append(section("Structural topology of transcription instability", topology_section, open_by_default=False))

    risk_section = (
        html_note(
            "How to read this section: the risk-weighted stratification is the starting point, not the final decision. "
            "Documents are first ranked within style as Low, Medium, or High index-derived risk using CER, IDR, continuity risk, and adjusted line delta. "
            "Supporting diagnostics such as confusions and topology explain behaviour but do not enter the composite score."
        )
        + subsection("Index-derived risk-band summary by style", html_table(risk_band_headers, risk_band_rows_out, datatable=True, csv_name="risk_band_summary_by_style"))
        + subsection("Operational document metrics table", html_table(op_headers, op_table_rows, datatable=True, csv_name="operational_document_metrics_display"))
        + subsection("Highest-risk documents under index-derived risk", html_table(top_risk_headers, top_risk_table_rows, datatable=True, csv_name="top_50_documents_by_simplified_risk"))
    )
    sections.append(section("Risk Weighted Stratification by Style", risk_section, open_by_default=True))

    governance_section = (
        html_note(
            "How to read this section: Document Status is the plain operational outcome. "
            "A document starts with its style-relative risk band. It is then checked for structural danger signals: severe line instability, very low continuity, or high IDR. "
            "If any of these are present, the document is escalated one level: Stable to Review, or Review to Exclude."
        )
        + subsection("Document status summary by style", html_table(status_headers, status_table_rows, datatable=True, csv_name="document_status_summary_by_style"))
        + subsection("Escalation flags", html_table(escalation_headers, escalation_table_rows, datatable=True, csv_name="document_status_escalation_flags"))
        + subsection(
            "Document Status Index",
            html_note(
                "The Document Status Index is the machine-readable handoff table for downstream use. "
                "It records each document's style, index-derived risk band, structural escalation flags, and final status: Stable, Review, or Exclude."
            )
            + html_table(document_status_headers, document_status_table_rows, datatable=True, csv_name="document_status_index"),
        )
    )
    sections.append(section("Document Status Governance", governance_section, open_by_default=True))

    downstream_section = html_note(
        "How to read this section: downstream ML and RAG processes should consume the same Document Status Index rather than separate selection systems. "
        "Stable documents are provisionally suitable for downstream use; Review documents require caution or human review; Exclude documents are structurally unreliable pending correction or adjudication."
    ) + """
<p>
The status layer is intentionally task-agnostic. A machine-learning workflow may use Stable documents for clean supervision and Review documents for robustness experiments. A RAG workflow may index Stable documents first and treat Review documents as chunk-review candidates. In both cases, Exclude documents should be withheld from routine downstream use until reviewed or corrected.
</p>
<p>
This report does not directly model semantic corruption. It therefore does not yet measure whether names, places, quantities, or events have been damaged in ways that affect retrieval answers. Instead, it identifies structural and transcriptional conditions under which semantic corruption is more likely: omission-heavy drift, low continuity, severe line mismatch, and recurrent character confusions.
</p>
"""
    sections.append(section("Downstream use of the Document Status Index", downstream_section, open_by_default=True))

    sections.append(section("Appendix A. Metric definitions", appendix_a_metric_definitions(), open_by_default=False))
    sections.append(section("Appendix B. Index-Derived Risk Stratification", appendix_b_risk_index(), open_by_default=False))
    sections.append(section("Appendix C. Human Review Protocol", appendix_c_human_review_protocol(), open_by_default=False))

    body = PLOTLY_CDN + "\n" + "\n".join(sections)
    page = html_page(REPORT_TITLE, body)

    out_path = POSTHOC_DIR / "corpus_report.html"
    safe_write_text(page, out_path)
    return str(out_path)


if __name__ == "__main__":
    print(build_report())
