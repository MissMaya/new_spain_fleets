"""
posthoc_analysis.py

Posthoc analytical utilities for interpreting overlap between pipeline stages.

Consumes:
- logs/posthoc/s1_s2_overlap.json
- logs/posthoc/s1_s3_overlap.json
- logs/posthoc/s2_s3_overlap.json

Produces:
- logs/posthoc/posthoc_summary.csv
- logs/posthoc/posthoc_summary.json
- logs/posthoc/posthoc_overlap_rates.png

Purpose:
Quantify how many Step 1 and Step 3 detections are explainable by OCR alignment
(Step 2), per calligraphy style.

This module performs NO detection. It only aggregates existing artifacts.
"""

from pathlib import Path
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.file_io import load_json_if_exists, safe_write_json
from utils.config import LOGS_DIR


def run_posthoc_analysis():
    posthoc_dir = LOGS_DIR / "posthoc"

    s1_s2 = load_json_if_exists(posthoc_dir / "s1_s2_overlap.json", {})
    s1_s3 = load_json_if_exists(posthoc_dir / "s1_s3_overlap.json", {})
    s2_s3 = load_json_if_exists(posthoc_dir / "s2_s3_overlap.json", {})

    styles = set(s1_s2.keys()) | set(s1_s3.keys()) | set(s2_s3.keys())

    summary = {}

    for style in sorted(styles):
        s1_s2_counts = s1_s2.get(style, {})
        s1_s3_counts = s1_s3.get(style, {})
        s2_s3_counts = s2_s3.get(style, {})

        s1_explained_by_s2 = sum(s1_s2_counts.values())
        s3_explained_by_s2 = sum(s2_s3_counts.values())
        s3_explained_by_s1 = sum(s1_s3_counts.values())

        summary[style] = {
            "s1_explained_by_s2": s1_explained_by_s2,
            "s3_explained_by_s2": s3_explained_by_s2,
            "s3_explained_by_s1": s3_explained_by_s1,
        }

    # --------------------------------------------------------------
    # Write JSON summary
    # --------------------------------------------------------------

    safe_write_json(summary, posthoc_dir / "posthoc_summary.json")

    # --------------------------------------------------------------
    # Write CSV summary
    # --------------------------------------------------------------

    csv_path = posthoc_dir / "posthoc_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["style", "s1_explained_by_s2", "s3_explained_by_s2", "s3_explained_by_s1"])

        for style, vals in summary.items():
            writer.writerow([
                style,
                vals["s1_explained_by_s2"],
                vals["s3_explained_by_s2"],
                vals["s3_explained_by_s1"],
            ])

    # --------------------------------------------------------------
    # Bar chart
    # --------------------------------------------------------------

    styles_sorted = list(summary.keys())
    s1_vals = [summary[s]["s1_explained_by_s2"] for s in styles_sorted]
    s3_s2_vals = [summary[s]["s3_explained_by_s2"] for s in styles_sorted]
    s3_s1_vals = [summary[s]["s3_explained_by_s1"] for s in styles_sorted]

    x = range(len(styles_sorted))

    plt.figure(figsize=(10, 6))
    plt.bar(x, s1_vals, label="S1 explained by S2")
    plt.bar(x, s3_s2_vals, bottom=s1_vals, label="S3 explained by S2")
    plt.bar(x, s3_s1_vals, bottom=[a + b for a, b in zip(s1_vals, s3_s2_vals)], label="S3 explained by S1")

    plt.xticks(x, styles_sorted, rotation=45)
    plt.ylabel("Overlap count")
    plt.title("Posthoc overlap attribution by calligraphy style")
    plt.legend()
    plt.tight_layout()

    plt.savefig(posthoc_dir / "posthoc_overlap_rates.png")
    plt.close()

    print("Posthoc analysis complete.")
