"""
build_corpus_report.py

Script to generate a full interactive corpus diagnostics report

Creates the report by:

- loading train/test metadata
- calling analytical helpers from utils.report_metrics
- calling rendering helpers from utils.report_html
- writing HTML and CSV outputs to logs/posthoc

Note
------------
True CER is always computed from GT/HTR text directly 
rather than being inferred from logged issue counts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv

from utils.config import LOGS_DIR
from utils.file_io import load_json_if_exists, safe_write_text
from utils.report_metrics import *
from pipeline.report_html import *


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

POSTHOC_DIR = LOGS_DIR / "posthoc"
TABLE_DIR = POSTHOC_DIR / "corpus_report_tables"
CORPUS_NAME = "New Spain Fleets"
PIPELINE_VERSION = "v1.0"


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def write_csv_table(
    name: str,
    headers: list[str],
    rows: list[list[object]],
) -> None:
    """
    Write a CSV companion file for any rendered table

    Parameters
    ----------
    name:
        Filename stem for the CSV output
    headers:
        Column headers
    rows:
        Table rows already formatted for rendering
    """
    TABLE_DIR.mkdir(parents = True, exist_ok = True)
    path = TABLE_DIR / f"{name}.csv"

    with open(path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_ready_rows(rows))


# ---------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------

def metadata_rows(
    generated: str,
    train_pairs: list[dict],
    test_pairs: list[dict],
    total_issues: int,
    corpus_mean_cer: float,
) -> list[list[object]]:
    """
    Build the metadata table rows
    """
    return [
        ["Generated", generated],
        ["Corpus", CORPUS_NAME],
        ["Pipeline version", PIPELINE_VERSION],
        ["Training documents", f_int(len(train_pairs))],
        ["Test documents", f_int(len(test_pairs))],
        ["Total logged issues", f_int(total_issues)],
        ["Corpus mean normalised CER (train)", f_float(corpus_mean_cer, 6)],
    ]


def corpus_distribution_rows(dist_rows: list[dict]) -> list[list[object]]:
    """
    Render training corpus composition by style
    """
    return [
        [
            row["style"],
            f_int(row["train_docs"]),
            f_pct(row["pct_train"]),
        ]
        for row in dist_rows
    ]


def geometry_rows(rows: list[dict]) -> list[list[object]]:
    """
    Render text-geometry-by-style rows
    """
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
    """
    Render overall logged issue distribution by stage
    """
    return [
        ["S1", f_int(stage_overview["S1"]), f_pct(stage_overview["S1_pct"])],
        ["S2", f_int(stage_overview["S2"]), f_pct(stage_overview["S2_pct"])],
        ["S3", f_int(stage_overview["S3"]), f_pct(stage_overview["S3_pct"])],
        ["TOTAL", f_int(stage_overview["total_issues"]), "100.00%"],
    ]


def issue_by_style_rows(rows: list[dict]) -> list[list[object]]:
    """
    Render issue-distribution-by-style rows.
    """
    out = []

    for row in rows:
        out.append([
            row["style"],
            f_int(row["train_docs"]),
            f_pct(row["pct_train"]),
            f_int(row["issues"]),
            f_pct(row["pct_issues"]),
            f_pp(row["diff_pp"]),
            row["weight"],
            f_int(row["S1"]),
            f_int(row["S2"]),
            f_int(row["S3"]),
        ])

    return out


def style_comparison_rows(rows: list[dict]) -> list[list[object]]:
    """
    Render the primary style-comparison rows

    Sorted by mean normalised CER descending so that the hardest
    styles appear first
    """
    ordered = sorted(rows, key = lambda r: r["mean_cer"], reverse = True)

    out = []

    for row in ordered:
        out.append([
            row["style"],
            f_int(row["docs"]),
            f_float(row["mean_cer"], 6),
            f_float(row["median_cer"], 6),
            f_float(row["p90_cer"], 6),
            f_float(row["gini"], 4),
            f_float(row["split_rate"], 2),
            f_float(row["merge_rate"], 2),
            f_pct(row["boundary_pct"]),
            f_pct(row["drift_rate"]),
            f_pct(row["clean_lt_1_pct"]),
            f_pct(row["clean_lt_2_pct"]),
            f_pct(row["clean_lt_5_pct"]),
            f_pct(row["sub_pct"]),
            f_pct(row["del_pct"]),
            f_pct(row["ins_pct"]),
            f_float(row["delta_cer"], 6),
            f_pp(row["delta_boundary_pct"] * 100),
            f_float(row["delta_split_rate"], 2),
            f_float(row["delta_merge_rate"], 2),
            f_pp(row["delta_drift_rate"] * 100),
        ])

    return out


def concentration_rows(rows: list[dict]) -> list[list[object]]:
    """
    Render style-level concentration metrics
    """
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


def top_doc_rows(rows: list[dict]) -> list[list[object]]:
    """
    Render overall top-document burden rows
    """
    out = []

    for row in rows:
        out.append([
            row["doc_id"],
            row["style"],
            f_int(row["edits"]),
            f_float(row["cer_norm"], 6),
            f_int(row["logged_issues"]),
            f_pct(row["boundary_pct_of_edits"]),
            "YES" if row["drift_flag"] else "NO",
        ])

    return out


def confusion_rows(items: list[dict], kind: str) -> list[list[object]]:
    """
    Render confusion tables for one style block

    Parameters
    ----------
    items:
        Confusion rows for one style.
    kind:
        One of: "char", "bigram", "word"
    """
    out = []

    if kind == "char":
        for row in items:
            out.append([
                row["gt"],
                row["htr"],
                f_int(row["count"]),
                f_pct(row["pct_style_char_confusions"]),
            ])

    elif kind == "bigram":
        for row in items:
            out.append([
                row["gt_bigram"],
                row["htr_out"],
                f_int(row["count"]),
                f_pct(row["pct_style_bigram_confusions"]),
            ])

    elif kind == "word":
        for row in items:
            out.append([
                row["gt_word"],
                row["htr_word"],
                f_int(row["count"]),
                f_pct(row["pct_style_word_confusions"]),
            ])

    return out


def doc_block_rows(rows: list[dict], block_type: str) -> list[list[object]]:
    """
    Render rows for a per-style document diagnostic block

    Parameters
    ----------
    rows:
        Document rows already filtered/ranked for one block
    block_type:
        One of:
            - "edit"
            - "issues"
            - "boundary"
            - "drift"
    """
    out = []

    if block_type == "edit":
        for row in rows:
            out.append([
                row["doc_id"],
                row["style"],
                f_int(row["edits"]),
                f_float(row["cer_norm"], 6),
                f_int(row["logged_issues"]),
                f_pct(row["boundary_pct_of_edits"]),
                "YES" if row["drift_flag"] else "NO",
            ])

    elif block_type == "issues":
        for row in rows:
            out.append([
                row["doc_id"],
                row["style"],
                f_int(row["logged_issues"]),
                f_int(row["S1_issues"]),
                f_int(row["S2_issues"]),
                f_int(row["S3_issues"]),
                f_int(row["edits"]),
                f_float(row["cer_norm"], 6),
            ])

    elif block_type == "boundary":
        for row in rows:
            out.append([
                row["doc_id"],
                row["style"],
                f_pct(row["boundary_pct_of_edits"]),
                f_int(row["boundary_events"]),
                f_int(row["split_count"]),
                f_int(row["merge_count"]),
                f_int(row["edits"]),
                f_float(row["cer_norm"], 6),
            ])

    elif block_type == "drift":
        for row in rows:
            out.append([
                row["doc_id"],
                row["style"],
                f_int(row["s2_total"]),
                f_pct(row["drift_insert_ratio"]),
                f_pct(row["drift_delete_ratio"]),
                f_int(row["edits"]),
                f_float(row["cer_norm"], 6),
            ])

    return out


# ---------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------

def build_corpus_report() -> None:
    """
    Build the complete corpus diagnostics report

    Sections:
        1. corpus description to set the scene
        2. stage/issue overview
        3. primary style comparison
        4. concentration and problematic documents
        5. style-specific error drivers
        6. per-style document diagnostics
        7. appendix
    """
    POSTHOC_DIR.mkdir(parents = True, exist_ok = True)
    TABLE_DIR.mkdir(parents = True, exist_ok = True)

    meta_dir = LOGS_DIR / "meta"

    train_pairs = load_json_if_exists(meta_dir / "train_pairs.json", [])
    test_pairs = load_json_if_exists(meta_dir / "test_pairs.json", [])

    issues = load_all_issues()
    stopwords = load_stopwords()

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # -----------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------

    dist_rows = train_distribution_by_style(train_pairs)
    geom_rows = geometry_by_style(train_pairs)

    stage_overview = issue_stage_overview(issues)
    issue_style_rows = issue_distribution_by_style(issues, train_pairs)

    doc_rows = compute_doc_metrics(train_pairs, issues)
    style_bundle = aggregate_style_metrics(doc_rows)

    char_conf = char_confusions_by_style(issues, top_n = 20)
    bigram_conf = bigram_confusions_by_style(issues, top_n = 20)
    word_conf = word_confusions_by_style(issues, stopwords = stopwords, top_n = 20)

    overall_top_docs = top_documents_overall(doc_rows, top_n = 20)
    per_style_blocks = per_style_document_blocks(doc_rows, top_n = 20)

    # -----------------------------------------------------------------
    # Build rows and write CSV companions
    # -----------------------------------------------------------------

    meta_headers = ["Metric", "Value"]
    meta_rows = metadata_rows(
        generated = generated,
        train_pairs = train_pairs,
        test_pairs = test_pairs,
        total_issues = len(issues),
        corpus_mean_cer = style_bundle["corpus_summary"]["mean_cer_norm"],
    )

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

    stage_headers = ["Stage", "Count", "% of all issues"]
    stage_table_rows = issue_stage_rows(stage_overview)
    write_csv_table("logged_issue_stage_overview", stage_headers, stage_table_rows)

    issue_style_headers = [
        "Style", "Train docs", "% train", "Issues", "% issues",
        "Issue vs corpus", "Weight", "S1", "S2", "S3",
    ]
    issue_style_table_rows = issue_by_style_rows(issue_style_rows)
    write_csv_table("logged_issues_by_style", issue_style_headers, issue_style_table_rows)

    style_headers = [
        "Style", "Docs", "Mean CER", "Median CER", "P90 CER", "Gini",
        "Splits/1k GT tokens", "Merges/1k GT tokens", "Boundary % edits",
        "Drift %", "Clean <1%", "Clean <2%", "Clean <5%",
        "Sub %", "Del %", "Ins %",
        "Δ CER", "Δ Boundary", "Δ Split rate", "Δ Merge rate", "Δ Drift",
    ]
    style_table_rows = style_comparison_rows(style_bundle["style_rows"])
    write_csv_table("primary_style_comparison", style_headers, style_table_rows)

    concentration_headers = [
        "Style", "Docs", "Total edits", "Gini",
        "Top 10% docs share of edits", "Top 5 docs share of edits",
    ]
    concentration_table_rows = concentration_rows(style_bundle["concentration_rows"])
    write_csv_table("error_concentration_by_style", concentration_headers, concentration_table_rows)

    top_docs_headers = [
        "Document", "Style", "Total edits", "CER norm",
        "Logged issues", "Boundary % of edits", "Drift flag",
    ]
    top_docs_table_rows = top_doc_rows(overall_top_docs)
    write_csv_table("top_documents_by_edit_burden_overall", top_docs_headers, top_docs_table_rows)

    # -----------------------------------------------------------------
    # Build HTML sections
    # -----------------------------------------------------------------

    sections = []

    # -----------------------------------------------------------------
    # Metadata and scope
    # -----------------------------------------------------------------

    meta_section = (
       html_table(meta_headers, meta_rows, datatable = False)
    )
    sections.append(section("Metadata", meta_section, open_by_default = True))

    # -----------------------------------------------------------------
    # Corpus description and issue overview
    # -----------------------------------------------------------------

    overview_section = (
        subsection(
            "Training corpus composition by style",
            html_table(
                dist_headers,
                dist_table_rows,
                csv_name = "training_corpus_by_style",
            )
        )
        + subsection(
            "Training corpus geometry by style",
            html_table(
                geom_headers,
                geom_table_rows,
                csv_name = "training_geometry_by_style",
            )
        )
        + subsection(
            "Overall logged issue distribution",
            html_note(
                "This table describes the overall stage mix of logged issues across the training corpus."
            )
            + html_table(
                stage_headers,
                stage_table_rows,
                csv_name = "logged_issue_stage_overview",
            )
        )
        + subsection(
            "Logged issues by style",
            html_note(
                "This table compares each style's share of logged issues against its share of the training corpus."
            )
            + html_table(
                issue_style_headers,
                issue_style_table_rows,
                csv_name = "logged_issues_by_style",
            )
        )
    )
    sections.append(section("Corpus description and issue overview", overview_section, open_by_default = True))

    # -----------------------------------------------------------------
    # Primary style comparison
    # -----------------------------------------------------------------

    style_section = (
        html_note(
            "This is the primary comparison layer for downstream ML: style difficulty, concentration, "
            "boundary burden, drift burden, clean-subset potential, and delta-vs-corpus views are all shown together."
        )
        + html_table(
            style_headers,
            style_table_rows,
            csv_name = "primary_style_comparison",
        )
    )
    sections.append(section("Primary style comparison", style_section, open_by_default = True))

    # -----------------------------------------------------------------
    # Error concentration and problematic documents
    # -----------------------------------------------------------------

    concentration_section = (
        subsection(
            "Concentration summary by style",
            html_table(
                concentration_headers,
                concentration_table_rows,
                csv_name = "error_concentration_by_style",
            )
        )
        + subsection(
            "Top documents by edit burden overall",
            html_note(
                "Documents are always shown with style so that particularly problematic documents remain interpretable in their stylistic context."
            )
            + html_table(
                top_docs_headers,
                top_docs_table_rows,
                csv_name = "top_documents_by_edit_burden_overall",
            )
        )
        + subsection(
            "Lorenz curves by style",
            lorenz_plot_block(style_bundle["lorenz_data"])
        )
    )
    sections.append(section("Error concentration and problematic documents", concentration_section))

    # -----------------------------------------------------------------
    # Error drivers by style
    # -----------------------------------------------------------------

    driver_parts = [
        html_note(
            "Character, bigram, and word confusion signals are separated deliberately because they answer different modelling questions: "
            "glyph-level confusion, transition-level confusion, and lexical confusion."
        )
    ]

    for style in sorted(set(char_conf) | set(bigram_conf) | set(word_conf)):
        char_rows = confusion_rows(char_conf.get(style, []), kind = "char")
        bigram_rows = confusion_rows(bigram_conf.get(style, []), kind = "bigram")
        word_rows = confusion_rows(word_conf.get(style, []), kind = "word")

        if char_rows:
            write_csv_table(
                f"{style}_char_confusions",
                ["GT", "HTR", "Count", "% of style char confusions"],
                char_rows
            )
        if bigram_rows:
            write_csv_table(
                f"{style}_bigram_confusions",
                ["GT bigram", "HTR output", "Count", "% of style bigram confusions"],
                bigram_rows
            )
        if word_rows:
            write_csv_table(
                f"{style}_word_confusions",
                ["GT word", "HTR word", "Count", "% of style word confusions"],
                word_rows
            )

        style_block = ""

        if char_rows:
            style_block += subsection(
                f"{style} — character confusions",
                html_table(
                    ["GT", "HTR", "Count", "% of style char confusions"],
                    char_rows,
                    csv_name = f"{style}_char_confusions",
                )
            )

        if bigram_rows:
            style_block += subsection(
                f"{style} — bigram confusions",
                html_table(
                    ["GT bigram", "HTR output", "Count", "% of style bigram confusions"],
                    bigram_rows,
                    csv_name = f"{style}_bigram_confusions",
                )
            )

        if word_rows:
            style_block += subsection(
                f"{style} — word confusions",
                html_table(
                    ["GT word", "HTR word", "Count", "% of style word confusions"],
                    word_rows,
                    csv_name = f"{style}_word_confusions",
                )
            )

        if style_block:
            driver_parts.append(subsection(f"Style block: {style}", style_block))

    sections.append(section("Error drivers by style: character, bigram, and word", "".join(driver_parts)))

    # -----------------------------------------------------------------
    # Per-style document diagnostics
    # -----------------------------------------------------------------

    per_style_parts = [
        html_note(
            "Each style gets its own document-diagnostic block so that particularly problematic documents can be compared within style rather than in a single mixed corpus-wide list."
        )
    ]

    for style in sorted(per_style_blocks):
        blocks = per_style_blocks[style]

        edit_rows = doc_block_rows(blocks["top_edit_burden"], "edit")
        issue_rows = doc_block_rows(blocks["top_issue_count"], "issues")
        boundary_rows = doc_block_rows(blocks["top_boundary_burden"], "boundary")
        drift_rows = doc_block_rows(blocks["drift_docs"], "drift")

        write_csv_table(
            f"{style}_top_docs_by_edit_burden",
            ["Document", "Style", "Edits", "CER norm", "Logged issues", "Boundary % edits", "Drift"],
            edit_rows
        )
        write_csv_table(
            f"{style}_top_docs_by_issue_count",
            ["Document", "Style", "Logged issues", "S1", "S2", "S3", "Edits", "CER norm"],
            issue_rows
        )
        write_csv_table(
            f"{style}_top_docs_by_boundary_burden",
            ["Document", "Style", "Boundary % edits", "Boundary events", "Splits", "Merges", "Edits", "CER norm"],
            boundary_rows
        )
        write_csv_table(
            f"{style}_drift_docs",
            ["Document", "Style", "S2 total", "Insert ratio", "Delete ratio", "Edits", "CER norm"],
            drift_rows
        )

        block_html = (
            subsection(
                f"{style} — top documents by edit burden",
                html_table(
                    ["Document", "Style", "Edits", "CER norm", "Logged issues", "Boundary % edits", "Drift"],
                    edit_rows,
                    csv_name = f"{style}_top_docs_by_edit_burden",
                )
            )
            + subsection(
                f"{style} — top documents by logged issue count",
                html_table(
                    ["Document", "Style", "Logged issues", "S1", "S2", "S3", "Edits", "CER norm"],
                    issue_rows,
                    csv_name = f"{style}_top_docs_by_issue_count",
                )
            )
            + subsection(
                f"{style} — top documents by boundary burden",
                html_table(
                    ["Document", "Style", "Boundary % edits", "Boundary events", "Splits", "Merges", "Edits", "CER norm"],
                    boundary_rows,
                    csv_name = f"{style}_top_docs_by_boundary_burden",
                )
            )
        )

        if drift_rows:
            block_html += subsection(
                f"{style} — drift-flagged documents",
                html_table(
                    ["Document", "Style", "S2 total", "Insert ratio", "Delete ratio", "Edits", "CER norm"],
                    drift_rows,
                    csv_name = f"{style}_drift_docs",
                )
            )

        per_style_parts.append(subsection(f"Per-style diagnostic block: {style}", block_html))

    sections.append(section("Per-style document diagnostics", "".join(per_style_parts)))

    # -----------------------------------------------------------------
    # Write HTML
    # -----------------------------------------------------------------

    body = "\n".join(sections)
    page = html_page("HTR Corpus Diagnostics Report", body)

    output = POSTHOC_DIR / "corpus_report.html"
    safe_write_text(page, output)

    print(f"Corpus report written: {output}")
    print(f"Companion CSV tables written: {TABLE_DIR}")


if __name__ == "__main__":
    build_corpus_report()