"""
posthoc_analysis.py

Posthoc analytical utilities to assess overlap between pipeline stages.

Inputs:
- logs/posthoc/s1_s2_overlap.json
- logs/posthoc/s1_s3_overlap.json
- logs/posthoc/s2_s3_overlap.json
- per-document issue logs under logs/<style>/<doc_id>/issues.json

Outputs:
- logs/posthoc/posthoc_summary.json
- logs/posthoc/posthoc_summary.csv
- logs/posthoc/posthoc_overlap_rates.png

Purpose:
Quantify how many Step 1 and Step 2 detections are explainable by later stages.
Generate totals, rates, and unexplained residuals.

"""

from pathlib import Path
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.file_io import load_json_if_exists, safe_write_json
from utils.config import LOGS_DIR
from utils.file_io import issues_json_path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sum_nested_counts(d):
    """
    Sum counts in a dict-of-dicts structure.
    """
    return sum(
        sum(inner.values())
        for inner in d.values()
        if isinstance(inner, dict)
    )


def count_tags_from_logs(prefix: str):
    """
    Count total issues whose tag starts with prefix (S1, S2, S3),
    then aggregate these per style and globally.
    """
    totals_by_style = defaultdict(int)
    total = 0

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir() or style_dir.name == "meta" or style_dir.name == "posthoc":
            continue

        style = style_dir.name

        for doc_dir in style_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            doc_id = doc_dir.name
            issues_path = issues_json_path(doc_dir, doc_id)
            if not issues_path.exists():
                continue

            issues = load_json_if_exists(issues_path, [])
            for issue in issues:
                if issue.get("tag", "").startswith(prefix):
                    totals_by_style[style] += 1
                    total += 1

    return total, dict(totals_by_style)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_posthoc_analysis():
    posthoc_dir = LOGS_DIR / "posthoc"
    posthoc_dir.mkdir(parents = True, exist_ok = True)

    # Load overlap logs
    s1_s2 = load_json_if_exists(posthoc_dir / "s1_s2_overlap.json", {})
    s1_s3 = load_json_if_exists(posthoc_dir / "s1_s3_overlap.json", {})
    s2_s3 = load_json_if_exists(posthoc_dir / "s2_s3_overlap.json", {})

    styles = set(s1_s2.keys()) | set(s1_s3.keys()) | set(s2_s3.keys())

    # Totals from logs
    total_s1, s1_by_style = count_tags_from_logs("S1")
    total_s2, s2_by_style = count_tags_from_logs("S2")
    total_s3, s3_by_style = count_tags_from_logs("S3")

    summary = {
        "totals": {
            "s1": total_s1,
            "s2": total_s2,
            "s3": total_s3,
        },
        "by_style": {},
    }

    # --------------------------------------------------------------
    # Per-style aggregation
    # --------------------------------------------------------------

    for style in sorted(styles):
        s1_s2_counts = s1_s2.get(style, {})
        s1_s3_counts = s1_s3.get(style, {})
        s2_s3_counts = s2_s3.get(style, {})

        s1_explained_by_s2 = sum_nested_counts(s1_s2_counts)
        s1_explained_by_s3 = sum_nested_counts(s1_s3_counts)
        s2_explained_by_s3 = sum_nested_counts(s2_s3_counts)

        s1_total_style = s1_by_style.get(style, 0)
        s2_total_style = s2_by_style.get(style, 0)
        s3_total_style = s3_by_style.get(style, 0)

        summary["by_style"][style] = {
            "totals": {
                "s1": s1_total_style,
                "s2": s2_total_style,
                "s3": s3_total_style,
            },
            "explained": {
                "s1_by_s2": s1_explained_by_s2,
                "s1_by_s3": s1_explained_by_s3,
                "s2_by_s3": s2_explained_by_s3,
            },
            "rates": {
                "s1_by_s2": s1_explained_by_s2 / s1_total_style if s1_total_style else 0.0,
                "s1_by_s3": s1_explained_by_s3 / s1_total_style if s1_total_style else 0.0,
                "s2_by_s3": s2_explained_by_s3 / s2_total_style if s2_total_style else 0.0,
            },
            "unexplained": {
                "s1": max(s1_total_style - (s1_explained_by_s2 + s1_explained_by_s3), 0),
                "s2": max(s2_total_style - s2_explained_by_s3, 0),
            },
        }

    # --------------------------------------------------------------
    # Write JSON
    # --------------------------------------------------------------

    safe_write_json(summary, posthoc_dir / "posthoc_summary.json")

    # --------------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------------

    csv_path = posthoc_dir / "posthoc_summary.csv"
    with open(csv_path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "style",
            "total_s1", "total_s2", "total_s3",
            "s1_by_s2", "s1_by_s3", "s2_by_s3",
            "rate_s1_by_s2", "rate_s1_by_s3", "rate_s2_by_s3",
            "unexplained_s1", "unexplained_s2",
        ])

        for style, data in summary["by_style"].items():
            writer.writerow([
                style,
                data["totals"]["s1"],
                data["totals"]["s2"],
                data["totals"]["s3"],
                data["explained"]["s1_by_s2"],
                data["explained"]["s1_by_s3"],
                data["explained"]["s2_by_s3"],
                round(data["rates"]["s1_by_s2"], 3),
                round(data["rates"]["s1_by_s3"], 3),
                round(data["rates"]["s2_by_s3"], 3),
                data["unexplained"]["s1"],
                data["unexplained"]["s2"],
            ])

    # --------------------------------------------------------------
    # Plot: overlap rates by style
    # --------------------------------------------------------------

    # TODO FLAG: MOVE THIS TO visualise.py
    styles_sorted = list(summary["by_style"].keys())
    s1_s2_rates = [summary["by_style"][s]["rates"]["s1_by_s2"] for s in styles_sorted]
    s1_s3_rates = [summary["by_style"][s]["rates"]["s1_by_s3"] for s in styles_sorted]

    x = range(len(styles_sorted))

    plt.figure(figsize = (10, 6))
    plt.bar(x, s1_s2_rates, label = "S1 explained by S2")
    plt.bar(x, s1_s3_rates, bottom = s1_s2_rates, label = "S1 explained by S3")

    plt.xticks(x, styles_sorted, rotation = 45)
    plt.ylabel("Explanation rate")
    plt.title("Posthoc explanation rates by calligraphy style")
    plt.legend()
    plt.tight_layout()

    plt.savefig(posthoc_dir / "posthoc_overlap_rates.png")
    plt.close()

    print("Posthoc analysis complete.")