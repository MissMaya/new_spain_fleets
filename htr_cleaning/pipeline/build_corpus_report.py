"""
build_corpus_report.py

Script generates a Markdown report for pipeline outputs.

Output:
    logs/posthoc/corpus_report.md
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from rapidfuzz.distance import Levenshtein

import statistics

from utils.config import LOGS_DIR, SCHEMAS_DIR
from utils.file_io import load_json_if_exists, read_json, read_text


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

POSTHOC_DIR = LOGS_DIR / "posthoc"
CORPUS_NAME = "New Spain Fleets"
PIPELINE_VERSION = "v1.0"

# These are based on previously run corpus diagnostics
LINE_BUCKETS = [
    (1, 5, "1-5"),
    (6, 10, "6-10"),
    (11, 15, "11-15"),
    (16, 20, "16-20"),
    (21, 25, "21-25"),
    (26, 30, "26-30"),
    (31, 35, "31-35"),
    (36, 40, "36-40"),
    (41, 45, "41-45"),
    (46, 50, "46-50"),
    (51, 10_000, "51+"),
]

CHAR_BUCKETS = [
    (0.00, 0.25, "0-25%"),
    (0.25, 0.50, "25-50%"),
    (0.50, 0.75, "50-75%"),
    (0.75, 1.00, "75-100%"),
]


# ---------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------

def h1(text: str) -> str:
    return f"# {text}\n"


def h2(text: str) -> str:
    return f"\n## {text}\n"


def h3(text: str) -> str:
    return f"\n### {text}\n"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_pct(part: int | float, total: int | float) -> str:
    if total == 0:
        return "0.00%"
    return f"{(part / total) * 100:.2f}%"


def fmt_pp(diff: float) -> str:
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.2f} pp"


def weight_label(diff: float, threshold: float = 1.0) -> str:
    if diff > threshold:
        return "Overweight"
    if diff < -threshold:
        return "Underweight"
    return "Balanced"


def fmt_count_pct(count: int, total: int) -> str:
    return f"{fmt_int(count)} ({fmt_pct(count, total)})"


def make_table(headers, rows):
    """
    Generate clean Markdown tables with alignment markers.
    This renders correctly in GitHub / VSCode preview.
    """

    # header row
    lines = []
    lines.append("| " + " | ".join(headers) + " |")

    # alignment row
    align = []
    for h in headers:
        if "Count" in h or "Total" in h or "%" in h or "Docs" in h:
            align.append("---:")
        else:
            align.append(":---")

    lines.append("| " + " | ".join(align) + " |")

    # body rows
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")

    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_stopwords() -> set[str]:
    """
    Load stopwords from the first path that exists.
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


def load_all_issues() -> list[dict]:
    """
    Load all per-document issues and inject style/doc_id from directory structure.
    """
    issues: list[dict] = []

    for style_dir in LOGS_DIR.iterdir():
        if not style_dir.is_dir():
            continue
        if style_dir.name in ("meta", "posthoc", "review", "step_summaries"):
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
                issue = dict(issue)
                issue["style"] = style
                issue["doc_id"] = doc_id
                issues.append(issue)

    return issues


def build_line_length_lookup(train_pairs: list[dict]) -> dict[str, dict[int, int]]:
    """
    Map doc_id -> {line_number: line_length}, using HTR text.
    Line numbers are 1-based to match logged issue line numbers.
    """
    lookup: dict[str, dict[int, int]] = {}

    for pair in train_pairs:
        doc_id = pair["id"]
        text = read_text(pair["htr_path"])
        lines = text.splitlines()

        lookup[doc_id] = {
            idx + 1: len(line)
            for idx, line in enumerate(lines)
        }

    return lookup


# ---------------------------------------------------------------------
# Corpus geometry
# ---------------------------------------------------------------------

def percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    idx = int(len(sorted_values) * p)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def compute_geometry_by_style(train_pairs: list[dict]) -> list[list[object]]:
    """
    TRAIN corpus geometry only, by style.
    Lower-tail and upper-tail percentiles show skew.
    """
    line_lengths_by_style: dict[str, list[int]] = defaultdict(list)
    doc_lengths_by_style: dict[str, list[int]] = defaultdict(list)

    for pair in train_pairs:
        style = pair["style"]
        text = read_text(pair["htr_path"])
        lines = text.splitlines()

        doc_lengths_by_style[style].append(len(lines))

        for line in lines:
            line_lengths_by_style[style].append(len(line))

    rows: list[list[object]] = []

    for style in sorted(line_lengths_by_style):
        line_lengths = sorted(line_lengths_by_style[style])
        doc_lengths = sorted(doc_lengths_by_style[style])

        rows.append([
            style,
            fmt_int(len(doc_lengths)),
            percentile(line_lengths, 0.10),
            int(statistics.median(line_lengths)),
            percentile(line_lengths, 0.90),
            percentile(doc_lengths, 0.10),
            int(statistics.median(doc_lengths)),
            percentile(doc_lengths, 0.90),
            max(doc_lengths) if doc_lengths else 0,
        ])

    return rows


# ---------------------------------------------------------------------
# Distribution tables
# ---------------------------------------------------------------------

def train_test_by_style(train_pairs: list[dict], test_pairs: list[dict]) -> list[list[object]]:
    train_counts = Counter(p["style"] for p in train_pairs)
    test_counts = Counter(p["style"] for p in test_pairs)

    styles = sorted(set(train_counts) | set(test_counts))
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())

    rows: list[list[object]] = []

    for style in styles:
        train_n = train_counts.get(style, 0)
        test_n = test_counts.get(style, 0)
        rows.append([
            style,
            f"{fmt_int(train_n)} ({fmt_pct(train_n, total_train)})",
            f"{fmt_int(test_n)} ({fmt_pct(test_n, total_test)})",
            fmt_int(train_n + test_n),
        ])

    return rows


def issue_distribution_by_stage(issues: list[dict]) -> list[list[object]]:
    counts = Counter(i["tag"][:2] for i in issues)
    total = sum(counts.values())

    rows: list[list[object]] = []
    for stage in ["S1", "S2", "S3"]:
        count = counts.get(stage, 0)
        rows.append([stage, fmt_int(count), fmt_pct(count, total)])

    rows.append(["TOTAL", fmt_int(total), "100.00%"])
    return rows


def issue_distribution_by_style(
    issues: list[dict],
    train_pairs: list[dict],
) -> list[list[object]]:
    issue_counts = Counter(i["style"] for i in issues)
    train_counts = Counter(p["style"] for p in train_pairs)

    total_issues = sum(issue_counts.values())
    total_train = sum(train_counts.values())

    styles = sorted(set(issue_counts) | set(train_counts))
    rows: list[list[object]] = []

    for style in styles:
        train_n = train_counts.get(style, 0)
        issue_n = issue_counts.get(style, 0)

        corpus_pct = (train_n / total_train) * 100 if total_train else 0.0
        issue_pct = (issue_n / total_issues) * 100 if total_issues else 0.0
        diff = issue_pct - corpus_pct

        rows.append([
            style,
            f"{fmt_int(train_n)} ({corpus_pct:.2f}%)",
            f"{fmt_int(issue_n)} ({issue_pct:.2f}%)",
            fmt_pp(diff),
            weight_label(diff),
        ])

    rows.append([
        "TOTAL",
        f"{fmt_int(total_train)} (100.00%)",
        f"{fmt_int(total_issues)} (100.00%)",
        "0.00 pp",
        "—",
    ])

    return rows


def stage_distribution_by_style(issues: list[dict]) -> list[list[object]]:
    table = defaultdict(Counter)

    for issue in issues:
        table[issue["style"]][issue["tag"][:2]] += 1

    rows: list[list[object]] = []

    total_s1 = total_s2 = total_s3 = 0

    for style in sorted(table):
        s1 = table[style].get("S1", 0)
        s2 = table[style].get("S2", 0)
        s3 = table[style].get("S3", 0)
        total = s1 + s2 + s3

        total_s1 += s1
        total_s2 += s2
        total_s3 += s3

        rows.append([
            style,
            fmt_int(s1),
            fmt_int(s2),
            fmt_int(s3),
            fmt_int(total),
        ])

    rows.append([
        "TOTAL",
        fmt_int(total_s1),
        fmt_int(total_s2),
        fmt_int(total_s3),
        fmt_int(total_s1 + total_s2 + total_s3),
    ])

    return rows


def alignment_ops_by_style(issues: list[dict]) -> list[list[object]]:
    table = defaultdict(Counter)

    for issue in issues:
        tag = issue["tag"]
        if tag not in ("S2X", "S2I", "S2D"):
            continue

        table[issue["style"]][tag] += 1

    rows: list[list[object]] = []

    total_x = total_i = total_d = 0

    for style in sorted(table):
        x = table[style].get("S2X", 0)
        ins = table[style].get("S2I", 0)
        d = table[style].get("S2D", 0)
        total = x + ins + d

        total_x += x
        total_i += ins
        total_d += d

        rows.append([
            style,
            fmt_int(ins),
            fmt_int(d),
            fmt_int(x),
            fmt_int(total),
        ])

    rows.append([
        "TOTAL",
        fmt_int(total_i),
        fmt_int(total_d),
        fmt_int(total_x),
        fmt_int(total_i + total_d + total_x),
    ])

    return rows


# ---------------------------------------------------------------------
# Substitution tables
# ---------------------------------------------------------------------

def top_word_subs_by_style(
    issues: list[dict],
    stopwords: set[str],
    top_n: int = 10,
) -> list[list[object]]:
    """
    Lowercase both GT and HTR before counting.
    Skip stopwords and pure capitalisation-only switches.
    """
    table = defaultdict(Counter)

    for issue in issues:
        if issue["tag"] != "S2X":
            continue

        gt = (issue.get("word_gt") or "").strip().lower()
        htr = (issue.get("word_htr") or "").strip().lower()

        if not gt or not htr:
            continue

        if gt in stopwords:
            continue

        if gt == htr:
            # ignore pure capitalisation-only differences after normalisation
            continue

        table[issue["style"]][(gt, htr)] += 1

    rows: list[list[object]] = []

    for style in sorted(table):
        for (gt, htr), count in table[style].most_common(top_n):
            rows.append([style, gt, htr, fmt_int(count)])

    return rows


def top_char_subs_by_style(
    issues: list[dict],
    top_n: int = 10,
) -> list[list[object]]:
    """
    Do NOT lowercase characters.
    """
    table = defaultdict(Counter)

    for issue in issues:
        if issue["tag"] != "S2X":
            continue

        gt = issue.get("gt_text", "") or ""
        htr = issue.get("htr_text", "") or ""

        for g, h in zip(gt, htr):
            if g == h:
                continue
            table[issue["style"]][(g, h)] += 1

    rows: list[list[object]] = []

    for style in sorted(table):
        for (g, h), count in table[style].most_common(top_n):
            rows.append([style, g, h, fmt_int(count)])

    return rows


# ---------------------------------------------------------------------
# Word-level contributions to CER
# ---------------------------------------------------------------------

def cer_contributing_words_by_style(issues, stopwords, top_n=10):

    style_stats = defaultdict(lambda: defaultdict(lambda: {
        "errors":0,
        "chars":0,
        "occurrences":0
    }))

    style_total_errors = defaultdict(int)

    for i in issues:

        if i["tag"] != "S2X":
            continue

        style = i["style"]

        gt = (i.get("word_gt") or "").lower()
        htr = (i.get("word_htr") or "").lower()

        if not gt:
            continue

        if gt in stopwords:
            continue

        errors = Levenshtein.distance(gt, htr)

        style_stats[style][gt]["errors"] += errors
        style_stats[style][gt]["chars"] += len(gt)
        style_stats[style][gt]["occurrences"] += 1

        style_total_errors[style] += errors

    rows = []

    for style in sorted(style_stats):

        ranked = sorted(
            style_stats[style].items(),
            key=lambda x: x[1]["errors"],
            reverse=True
        )

        for word, stats in ranked[:top_n]:

            cer = stats["errors"] / max(stats["chars"],1)
            contribution = stats["errors"] / max(style_total_errors[style],1)

            rows.append([
                style,
                word,
                fmt_int(stats["occurrences"]),
                fmt_int(stats["errors"]),
                f"{cer:.3f}",
                f"{contribution*100:.2f}%"
            ])

    return rows

# ---------------------------------------------------------------------
# Positional distributions
# ---------------------------------------------------------------------

def line_distribution_by_style(issues: list[dict]) -> list[list[object]]:
    """
    Step 2 only.
    Report count and within-style percentage in each line bucket.
    """
    table = defaultdict(Counter)

    for issue in issues:
        if not issue["tag"].startswith("S2"):
            continue

        line = issue.get("line")
        if not line:
            continue

        for lo, hi, label in LINE_BUCKETS:
            if lo <= line <= hi:
                table[issue["style"]][label] += 1
                break

    rows: list[list[object]] = []

    labels = [label for _, _, label in LINE_BUCKETS]

    grand_totals = Counter()

    for style in sorted(table):
        style_total = sum(table[style].values())
        row: list[object] = [style]

        for label in labels:
            count = table[style][label]
            grand_totals[label] += count
            row.append(fmt_count_pct(count, style_total))

        row.append(fmt_int(style_total))
        rows.append(row)

    total_all = sum(grand_totals.values())
    total_row: list[object] = ["TOTAL"]
    for label in labels:
        total_row.append(fmt_count_pct(grand_totals[label], total_all))
    total_row.append(fmt_int(total_all))
    rows.append(total_row)

    return rows


def char_position_distribution_by_style(
    issues: list[dict],
    line_length_lookup: dict[str, dict[int, int]],
) -> list[list[object]]:
    """
    Step 2 only.
    Correct relative character position calculation:
        relative_position = char_start / line_length

    Uses actual HTR line lengths, not char_end.
    """
    table = defaultdict(Counter)

    for issue in issues:
        if not issue["tag"].startswith("S2"):
            continue

        doc_id = issue["doc_id"]
        line_no = issue.get("line")
        char_start = issue.get("char_start")

        if not line_no or not char_start:
            continue

        line_length = line_length_lookup.get(doc_id, {}).get(line_no, 0)
        if line_length <= 0:
            continue

        rel = char_start / line_length
        rel = max(0.0, min(rel, 1.0))

        for lo, hi, label in CHAR_BUCKETS:
            if hi == 1.00:
                cond = lo <= rel <= hi
            else:
                cond = lo <= rel < hi

            if cond:
                table[issue["style"]][label] += 1
                break

    rows: list[list[object]] = []

    labels = [label for _, _, label in CHAR_BUCKETS]
    grand_totals = Counter()

    for style in sorted(table):
        style_total = sum(table[style].values())
        row: list[object] = [style]

        for label in labels:
            count = table[style][label]
            grand_totals[label] += count
            row.append(fmt_count_pct(count, style_total))

        row.append(fmt_int(style_total))
        rows.append(row)

    total_all = sum(grand_totals.values())
    total_row: list[object] = ["TOTAL"]
    for label in labels:
        total_row.append(fmt_count_pct(grand_totals[label], total_all))
    total_row.append(fmt_int(total_all))
    rows.append(total_row)

    return rows


# ---------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------

def build_corpus_report() -> None:
    POSTHOC_DIR.mkdir(parents=True, exist_ok=True)

    meta_dir = LOGS_DIR / "meta"

    train_pairs = load_json_if_exists(meta_dir / "train_pairs.json", [])
    test_pairs = load_json_if_exists(meta_dir / "test_pairs.json", [])

    issues = load_all_issues()
    stopwords = load_stopwords()
    line_length_lookup = build_line_length_lookup(train_pairs)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report: list[str] = []

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    report.append(h1("HTR Corpus Diagnostics Report"))

    report.append(make_table(
        ["Metadata", "Value"],
        [
            ["Generated", generated],
            ["Corpus", CORPUS_NAME],
            ["Pipeline version", PIPELINE_VERSION],
            ["Total documents", fmt_int(len(train_pairs) + len(test_pairs))],
            ["Training documents", fmt_int(len(train_pairs))],
            ["Test documents", fmt_int(len(test_pairs))],
            ["Total issues", fmt_int(len(issues))],
        ]
    ))

    # -----------------------------------------------------------------
    # Train / test by style
    # -----------------------------------------------------------------

    report.append(h2("Train/Test Distribution by Style"))
    report.append(make_table(
        ["Style", "Train", "Test", "Total"],
        train_test_by_style(train_pairs, test_pairs)
    ))

    # -----------------------------------------------------------------
    # Train corpus geometry only, by style
    # -----------------------------------------------------------------

    report.append(h2("Training Corpus Geometry by Style"))
    report.append(make_table(
        ["Style", "Docs", "Line P10", "Line Median", "Line P90", "Doc P10", "Doc Median", "Doc P90", "Doc Max"],
        compute_geometry_by_style(train_pairs)
    ))

    # -----------------------------------------------------------------
    # Issue distributions
    # -----------------------------------------------------------------

    report.append(h2("Issue Distribution by Pipeline Stage"))
    report.append(make_table(
        ["Stage", "Count", "% of all issues"],
        issue_distribution_by_stage(issues)
    ))

    report.append(h2("Issue Distribution by Style"))
    report.append(make_table(
        ["Style", "Train docs (% of train)", "Issues (% of issues)", "Issue vs corpus", "Weight"],
        issue_distribution_by_style(issues, train_pairs)
    ))

    report.append(h2("Stage Distribution by Style"))
    report.append(make_table(
        ["Style", "S1", "S2", "S3", "Total"],
        stage_distribution_by_style(issues)
    ))

    # -----------------------------------------------------------------
    # Step 2 alignment breakdown
    # -----------------------------------------------------------------

    report.append(h2("Alignment Operations by Style"))
    report.append(make_table(
        ["Style", "Insertions", "Deletions", "Substitutions", "Total"],
        alignment_ops_by_style(issues)
    ))

    # -----------------------------------------------------------------
    # Substitutions
    # -----------------------------------------------------------------

    report.append(h2("Top Word Substitutions by Style"))
    report.append(make_table(
        ["Style", "GT", "HTR", "Count"],
        top_word_subs_by_style(issues, stopwords)
    ))

    report.append(h2("Top Character Substitutions by Style"))
    report.append(make_table(
        ["Style", "GT char", "HTR char", "Count"],
        top_char_subs_by_style(issues)
    ))

    # -----------------------------------------------------------------
    # Largest contributions to CER by style
    # -----------------------------------------------------------------

    report.append(h2("Words Contributing Most to CER (by Style)"))

    report.append(make_table(
        ["Style","GT Word","Occurrences","Char Errors","Avg CER","% of Style CER"],
        cer_contributing_words_by_style(issues, stopwords)
        ))


    # -----------------------------------------------------------------
    # Positional diagnostics (Step 2 only)
    # -----------------------------------------------------------------

    report.append(h2("Error Distribution by Line Number (Step 2 only)"))
    report.append(make_table(
        ["Style", "1-10", "11-20", "21-30", "31-40", "41-50", "51+", "Total"],
        line_distribution_by_style(issues)
    ))

    report.append(h2("Error Distribution by Relative Character Position (Step 2 only)"))
    report.append(make_table(
        ["Style", "0-25%", "25-50%", "50-75%", "75-100%", "Total"],
        char_position_distribution_by_style(issues, line_length_lookup)
    ))

    output = POSTHOC_DIR / "corpus_report.md"

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Corpus report written: {output}")


if __name__ == "__main__":
    build_corpus_report()