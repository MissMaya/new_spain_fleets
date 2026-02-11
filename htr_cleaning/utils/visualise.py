"""
visualise.py

Utilities for plotting pipeline outputs

Supports:

Step 1:
- JSON summaries
- CSV summaries
- Per-style bar charts

Step 2:
- Unified Ø confusion matrices (GT x HTR)
- CSV exports per style
- PNG heatmaps per style

No interactive display - all plots are written to output files. 
"""

from pathlib import Path
from typing import Dict
import json
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NULL_CHAR = "Ø"


# ----------------------------------------------------------------------
# STEP 1 VISUALS
# ----------------------------------------------------------------------

def write_step_summary_json(error_counts_by_style: Dict, output_dir: Path, step_name: str):
    output_dir.mkdir(parents = True, exist_ok = True)
    path = output_dir / f"{step_name}_summary.json"

    with open(path, "w", encoding = "utf-8") as f:
        json.dump(error_counts_by_style, f, indent = 2, ensure_ascii = False)


def write_step_summary_csv(error_counts_by_style: Dict, output_dir: Path, step_name: str):
    output_dir.mkdir(parents = True, exist_ok = True)
    path = output_dir / f"{step_name}_summary.csv"

    styles = sorted(error_counts_by_style.keys())
    tags = sorted(next(iter(error_counts_by_style.values())).keys())

    with open(path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["style"] + tags)

        for style in styles:
            row = [style] + [error_counts_by_style[style][tag] for tag in tags]
            writer.writerow(row)


def plot_step_summary(error_counts_by_style: Dict, output_dir: Path, step_name: str):
    output_dir.mkdir(parents = True, exist_ok = True)

    for style, tag_counts in error_counts_by_style.items():
        tags = list(tag_counts.keys())
        counts = [tag_counts[t] for t in tags]

        plt.figure(figsize = (10, 4))
        plt.bar(tags, counts)
        plt.title(f"{step_name.upper()} - {style}")
        plt.xlabel("Tag")
        plt.ylabel("Count")
        plt.tight_layout()

        out_path = output_dir / f"{step_name}_{style}.png"
        plt.savefig(out_path)
        plt.close()


def generate_all_outputs(error_counts_by_style: Dict, step_name: str, output_dir: Path):
    write_step_summary_json(error_counts_by_style, output_dir, step_name)
    write_step_summary_csv(error_counts_by_style, output_dir, step_name)
    plot_step_summary(error_counts_by_style, output_dir, step_name)


# ----------------------------------------------------------------------
# STEP 2 VISUALS
# ----------------------------------------------------------------------

def write_confusion_matrices(confusion_by_style: Dict, output_dir: Path):
    """
    Write per style confusion matrices (HTR x GT) to CSV + PNG.
    """

    output_dir.mkdir(parents = True, exist_ok = True)

    for style, matrix in confusion_by_style.items():

        gt_chars = set(matrix.keys())
        htr_chars = set()

        for g in matrix:
            htr_chars.update(matrix[g].keys())

        gt_chars = sorted(gt_chars)
        htr_chars = sorted(htr_chars)

        # Ensure Ø present
        if NULL_CHAR not in gt_chars:
            gt_chars.append(NULL_CHAR)
        if NULL_CHAR not in htr_chars:
            htr_chars.append(NULL_CHAR)

        # Build dense matrix
        dense = []
        for g in gt_chars:
            row = []
            for h in htr_chars:
                row.append(matrix[g].get(h, 0))
            dense.append(row)

        # --------------------------------------------------------------
        # CSV
        # --------------------------------------------------------------

        csv_path = output_dir / f"step2_confusion_{style}.csv"
        with open(csv_path, "w", newline = "", encoding = "utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["GT\\HTR"] + htr_chars)
            for g, row in zip(gt_chars, dense):
                writer.writerow([g] + row)

        # --------------------------------------------------------------
        # PNG heatmap
        # --------------------------------------------------------------

        plt.figure(figsize = (12, 10))
        plt.imshow(dense)
        plt.colorbar()
        plt.xticks(range(len(htr_chars)), htr_chars, rotation = 90)
        plt.yticks(range(len(gt_chars)), gt_chars)
        plt.title(f"Step 2 Confusion Matrix - {style}")
        plt.tight_layout()

        png_path = output_dir / f"step2_confusion_{style}.png"
        plt.savefig(png_path)
        plt.close()
