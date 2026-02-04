"""
visualise.py

Utilities for producing summaries from pipeline outputs.

This module converts structured error counts into:

- CSV tables
- JSON summaries
- Bar plots

"""

from pathlib import Path
from typing import Dict

import json
import csv

import matplotlib
matplotlib.use("Agg")  # Headless backend for non-interactive environments
import matplotlib.pyplot as plt


def write_step_summary_json(error_counts_by_style: Dict, output_dir: Path, step_name: str):
    """
    Write aggregate error counts to a JSON summary file.
    """
    output_dir.mkdir(parents = True, exist_ok = True)
    path = output_dir / f"{step_name}_summary.json"

    with open(path, "w", encoding = "utf-8") as f:
        json.dump(error_counts_by_style, f, indent = 2, ensure_ascii = False)


def write_step_summary_csv(error_counts_by_style: Dict, output_dir: Path, step_name: str):
    """
    Write aggregate error counts to a CSV file (one row per style).
    """
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
    """
    Generate bar plot per calligraphy style showing tag counts.
    """
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


def generate_all_outputs(
    error_counts_by_style: Dict,
    step_name: str,
    output_dir: Path,
):
    """
    Generate:

    - JSON summary
    - CSV summary
    - Per-style bar plots
    """

    write_step_summary_json(error_counts_by_style, output_dir, step_name)
    write_step_summary_csv(error_counts_by_style, output_dir, step_name)
    plot_step_summary(error_counts_by_style, output_dir, step_name)
