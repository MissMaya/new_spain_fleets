"""
build_document_subsets.py

Exports simple document-subset CSVs from the Document Status Index.

This script is designed to run AFTER build_corpus_report.py has produced the
Document Status Index / document-level status table. It does not create reviewer
packets and does not touch outputs/human_review/. Reviewer packet construction
is handled by build_human_review_sampling.py.

Default output directory:
    outputs/ml_outputs/

Generated files:
    stable_documents.csv
    review_documents.csv
    exclude_documents.csv
    document_subsets_summary.csv

The subset files are intentionally simple handoff artefacts for downstream
machine-learning and RAG experimentation. The full methodological record remains
in the source Document Status Index.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# User-controlled parameters
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("outputs") / "ml_outputs"

# Leave as None to auto-detect from DEFAULT_STATUS_INDEX_CANDIDATES.
DOCUMENT_STATUS_INDEX_CSV: Path | None = None

DEFAULT_STATUS_INDEX_CANDIDATES = [
    Path("logs/posthoc/corpus_report_tables/document_status_index.csv"),
    Path("outputs/ml_outputs/document_status_index.csv"),
    Path("logs/posthoc/corpus_report_tables/operational_document_metrics.csv"),
]

STATUS_COLUMN_CANDIDATES = [
    "document_status",
    "governance_status",
    "status",
]

REQUIRED_BASE_COLUMNS = [
    "doc_id",
    "style",
    "risk_band",
    "document_status",
]

# Keep useful provenance/diagnostic columns if present.
OPTIONAL_EXPORT_COLUMNS = [
    "risk_score",
    "risk_rank_within_style",
    "status_reason",
    "escalation_flags",
    "human_review_selected",
    "human_review_layer",
    "cer_norm",
    "wer_norm",
    "indel_disruption_rate",
    "continuity",
    "adjusted_line_delta_abs",
    "line_structure_band",
    "line_structure_flag",
    "basic_anomaly_density",
    "orthographic_violation_density",
    "image_path_or_url",
    "gt_path_or_url",
    "htr_path_or_url",
    "gt_path",
    "htr_path",
    "filename",
    "pair_id",
]

STATUS_ALIASES = {
    "stable": "Stable",
    "review": "Review",
    "exclude": "Exclude",
    "stable_documents": "Stable",
    "review_documents": "Review",
    "exclude_documents": "Exclude",
    "low": "Stable",
    "medium": "Review",
    "high": "Exclude",
}


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_default_status_index_csv() -> Path:
    found = find_first_existing(DEFAULT_STATUS_INDEX_CANDIDATES)
    if found:
        return found
    raise SystemExit(
        "Could not find a Document Status Index CSV. Provide --status-index-csv. Tried: "
        + ", ".join(str(p) for p in DEFAULT_STATUS_INDEX_CANDIDATES)
    )


# ---------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------


def normalise_status(value: Any) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace(" ", "_").replace("-", "_")
    return STATUS_ALIASES.get(key, raw if raw in {"Stable", "Review", "Exclude"} else "Unassigned")


def find_status_column(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    columns = set(rows[0].keys())
    for col in STATUS_COLUMN_CANDIDATES:
        if col in columns:
            return col
    return None


def normalise_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Normalise likely column variants and ensure document_status exists."""
    if not rows:
        return []

    status_col = find_status_column(rows)
    out: list[dict[str, Any]] = []

    aliases = {
        "Document ID": "doc_id",
        "Doc ID": "doc_id",
        "Style": "style",
        "Risk band": "risk_band",
        "Risk score": "risk_score",
        "Risk rank within style": "risk_rank_within_style",
        "Document status": "document_status",
        "Governance status": "governance_status",
        "Status reason": "status_reason",
        "Escalation flags": "escalation_flags",
        "Human review selected": "human_review_selected",
        "Human review layer": "human_review_layer",
        "CER": "cer_norm",
        "WER": "wer_norm",
        "IDR": "indel_disruption_rate",
        "Continuity": "continuity",
        "Adjusted line delta": "adjusted_line_delta_abs",
        "Line-structure band": "line_structure_band",
        "Line-structure flag": "line_structure_flag",
        "Image path": "image_path_or_url",
        "GT path": "gt_path",
        "HTR path": "htr_path",
        "GT file": "gt_path",
        "HTR file": "htr_path",
    }

    for raw in rows:
        row: dict[str, Any] = dict(raw)

        for source, target in aliases.items():
            if source in row and not str(row.get(target, "")).strip():
                row[target] = row[source]

        row["doc_id"] = str(row.get("doc_id") or row.get("id") or row.get("filename") or "").strip()
        row["style"] = str(row.get("style") or "UNKNOWN").strip() or "UNKNOWN"

        if status_col:
            row["document_status"] = normalise_status(row.get(status_col))
        elif "risk_band" in row:
            # Fallback only for older report outputs. This should not be the
            # preferred path once the Document Status Index is in place.
            row["document_status"] = normalise_status(row.get("risk_band"))
        else:
            row["document_status"] = "Unassigned"

        if row["doc_id"]:
            out.append(row)

    return out


def export_columns(rows: list[dict[str, Any]]) -> list[str]:
    available = set()
    for row in rows:
        available.update(row.keys())

    cols: list[str] = []
    for col in REQUIRED_BASE_COLUMNS + OPTIONAL_EXPORT_COLUMNS:
        if col in available and col not in cols:
            cols.append(col)
    return cols


# ---------------------------------------------------------------------
# Output logic
# ---------------------------------------------------------------------


def write_subset_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = export_columns(rows)

    stable = [r for r in rows if r.get("document_status") == "Stable"]
    review = [r for r in rows if r.get("document_status") == "Review"]
    exclude = [r for r in rows if r.get("document_status") == "Exclude"]

    write_csv_dicts(output_dir / "stable_documents.csv", stable, columns)
    write_csv_dicts(output_dir / "review_documents.csv", review, columns)
    write_csv_dicts(output_dir / "exclude_documents.csv", exclude, columns)

    status_counts = Counter(str(r.get("document_status", "Unassigned")) for r in rows)
    style_counts = Counter(str(r.get("style", "UNKNOWN")) for r in rows)

    summary_rows = [
        {"category": "total_documents", "value": len(rows)},
        {"category": "stable_documents", "value": len(stable)},
        {"category": "review_documents", "value": len(review)},
        {"category": "exclude_documents", "value": len(exclude)},
    ]
    for status, count in sorted(status_counts.items()):
        summary_rows.append({"category": f"status::{status}", "value": count})
    for style, count in sorted(style_counts.items()):
        summary_rows.append({"category": f"style::{style}", "value": count})

    write_csv_dicts(output_dir / "document_subsets_summary.csv", summary_rows, ["category", "value"])


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stable/Review/Exclude document subset CSVs from the Document Status Index.")
    parser.add_argument(
        "--status-index-csv",
        type=Path,
        default=DOCUMENT_STATUS_INDEX_CSV,
        help="Path to document_status_index.csv. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory. Default: outputs/ml_outputs",
    )
    return parser.parse_args()


def build_document_subsets() -> None:
    args = parse_args()
    source_csv = args.status_index_csv or find_default_status_index_csv()

    rows = normalise_rows(read_csv_dicts(source_csv))
    if not rows:
        raise SystemExit(f"No usable rows found in {source_csv}")

    write_subset_outputs(rows, args.output_dir)

    counts = Counter(str(r.get("document_status", "Unassigned")) for r in rows)
    print(f"Document status source: {source_csv}")
    print(f"Output directory: {args.output_dir}")
    print("Subset counts:")
    for status in ["Stable", "Review", "Exclude", "Unassigned"]:
        if counts.get(status, 0):
            print(f"  {status}: {counts[status]}")


if __name__ == "__main__":
    build_document_subsets()
