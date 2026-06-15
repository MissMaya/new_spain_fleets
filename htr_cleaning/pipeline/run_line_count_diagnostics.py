"""
run_line_count_diagnostics.py

Script produces per-document lines counts comparing the number of lines 
in a GT file with the number of lines in the corresponding HTR script.

Runs after run_split() on files in the train set, using logs/meta/train_pairs.json

Outputs
-------
1. logs/line_counts/line_count_diagnostics.jsonl
   Detailed one-record-per-document line count info to be included in final reporting

2. logs/line_counts/line_count_diagnostics.csv
   CSV of line count info for quick inspection 

Logic
-------
Before comparing GT/HTR line counts:

- Any blank lines in the GT are removed and their positions noted
- HTR blank lines are removed ONLY if they occur in the exact same
  location as a GT blank line. This removal is also noted.
- HTR blank lines without a matching GT blank are preserved and logged

"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import LOGS_DIR
from utils.file_io import load_json_if_exists, read_text


META_DIR = LOGS_DIR / "meta"
LINE_COUNT_DIR = LOGS_DIR / "line_counts"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_lines_including_blanks(path: str | Path) -> list[str]:
    """
    Read text and preserve blank lines.
    """
    return read_text(Path(path)).splitlines()


def is_blank_line(line: str) -> bool:
    """
    Treat empty and whitespace-only lines as blank.
    """
    return not line.strip()


# ---------------------------------------------------------------------
# Line count analysis
# ---------------------------------------------------------------------

def analyse_line_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """
    Function to analyse one GT/HTR document pair.

    Rules
    -----
    - Count all original lines, including blank ones.
    - Remove blank GT lines before adjusted delta calculation.
    - Remove HTR blank lines ONLY if GT is also blank at the same line number.
    - Preserve/log HTR blank lines where GT is not blank at the same line number.
    """
    gt_path = Path(pair["gt_path"])
    htr_path = Path(pair["htr_path"])

    gt_lines = read_lines_including_blanks(gt_path)
    htr_lines = read_lines_including_blanks(htr_path)

    gt_blank_lines_removed: list[int] = []
    htr_blank_lines_removed_because_gt_blank: list[int] = []
    htr_blank_lines_preserved_no_matching_gt_blank: list[int] = []

    adjusted_gt_lines: list[str] = []
    adjusted_htr_lines: list[str] = []

    max_line_count = max(len(gt_lines), len(htr_lines))

    for idx in range(max_line_count):
        line_number = idx + 1

        gt_exists = idx < len(gt_lines)
        htr_exists = idx < len(htr_lines)

        gt_line = gt_lines[idx] if gt_exists else None
        htr_line = htr_lines[idx] if htr_exists else None

        gt_is_blank = gt_exists and is_blank_line(gt_line)
        htr_is_blank = htr_exists and is_blank_line(htr_line)

        # -------------------------------------------------------------
        # GT blank line
        # -------------------------------------------------------------

        if gt_is_blank:
            gt_blank_lines_removed.append(line_number)

            # Matching HTR blank line at exact same position
            if htr_is_blank:
                htr_blank_lines_removed_because_gt_blank.append(line_number)

            # HTR not blank while GT is blank -> preserve
            elif htr_exists:
                adjusted_htr_lines.append(htr_line)

            continue

        # -------------------------------------------------------------
        # GT non-blank line
        # -------------------------------------------------------------

        if gt_exists:
            adjusted_gt_lines.append(gt_line)

        if htr_exists:

            # HTR blank with no matching GT blank
            if htr_is_blank:
                htr_blank_lines_preserved_no_matching_gt_blank.append(line_number)

            adjusted_htr_lines.append(htr_line)

    gt_original_line_count = len(gt_lines)
    htr_original_line_count = len(htr_lines)

    gt_adjusted_line_count = len(adjusted_gt_lines)
    htr_adjusted_line_count = len(adjusted_htr_lines)

    return {

        # -------------------------------------------------------------
        # JSON metadata fields
        # -------------------------------------------------------------

        "pair_id": pair["id"],
        "doc_id": htr_path.stem,
        "filename": htr_path.name,
        "style": pair["style"],

        "gt_path": str(gt_path),
        "htr_path": str(htr_path),

        # -------------------------------------------------------------
        # Original counts
        # -------------------------------------------------------------

        "gt_original_line_count": gt_original_line_count,
        "htr_original_line_count": htr_original_line_count,

        "original_delta_gt_minus_htr":
            gt_original_line_count - htr_original_line_count,

        # -------------------------------------------------------------
        # Adjusted counts
        # -------------------------------------------------------------

        "gt_adjusted_line_count": gt_adjusted_line_count,
        "htr_adjusted_line_count": htr_adjusted_line_count,

        "adjusted_delta_gt_minus_htr":
            gt_adjusted_line_count - htr_adjusted_line_count,

        # -------------------------------------------------------------
        # Blank-line diagnostics
        # -------------------------------------------------------------

        "gt_blank_lines_removed":
            gt_blank_lines_removed,

        "htr_blank_lines_removed_because_gt_blank":
            htr_blank_lines_removed_because_gt_blank,

        "htr_blank_lines_preserved_no_matching_gt_blank":
            htr_blank_lines_preserved_no_matching_gt_blank,

        # -------------------------------------------------------------
        # Summary counts
        # -------------------------------------------------------------

        "gt_blank_removed_count":
            len(gt_blank_lines_removed),

        "htr_blank_removed_matching_gt_count":
            len(htr_blank_lines_removed_because_gt_blank),

        "htr_blank_preserved_unmatched_count":
            len(htr_blank_lines_preserved_no_matching_gt_blank),
    }


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write one JSON object per line.
    """
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("w", encoding = "utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii = False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    CSV file to allow quick inspection/debugging.
    """
    path.parent.mkdir(parents = True, exist_ok = True)

    fieldnames = [
        "pair_id",
        "doc_id",
        "filename",
        "style",

        "gt_original_line_count",
        "htr_original_line_count",
        "original_delta_gt_minus_htr",

        "gt_adjusted_line_count",
        "htr_adjusted_line_count",
        "adjusted_delta_gt_minus_htr",

        "gt_blank_removed_count",
        "htr_blank_removed_matching_gt_count",
        "htr_blank_preserved_unmatched_count",

        "gt_blank_lines_removed",
        "htr_blank_lines_removed_because_gt_blank",
        "htr_blank_lines_preserved_no_matching_gt_blank",

        "gt_path",
        "htr_path",
    ]

    with path.open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        for row in rows:
            flat = dict(row)

            for key in [
                "gt_blank_lines_removed",
                "htr_blank_lines_removed_because_gt_blank",
                "htr_blank_lines_preserved_no_matching_gt_blank",
            ]:
                flat[key] = ";".join(str(n) for n in flat[key])

            writer.writerow({
                key: flat.get(key, "")
                for key in fieldnames
            })


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_line_count_diagnostics() -> list[dict[str, Any]]:
    """
    Run line count diagnostics on pairs in the training set only 
    """
    pairs = load_json_if_exists(META_DIR / "train_pairs.json", [])

    rows = [
        analyse_line_pair(pair)
        for pair in pairs
    ]

    LINE_COUNT_DIR.mkdir(parents = True, exist_ok = True)

    jsonl_path = LINE_COUNT_DIR / "line_count_diagnostics.jsonl"
    csv_path = LINE_COUNT_DIR / "line_count_diagnostics.csv"

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    print(f"Line-count diagnostics written: {len(rows):,} documents")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")

    return rows


def main() -> None:
    run_line_count_diagnostics()


if __name__ == "__main__":
    main()
