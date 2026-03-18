"""
text_diagnostics.py

Diagnostic script used to understand:
- line length distribution
- document length distribution

These are used to work out the appropriate bucket sizes for
use in build_corpus_report

"""

from collections import Counter
from statistics import median
from utils.config import LOGS_DIR
from utils.file_io import load_json_if_exists, read_text


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def percentile(data, p):

    data = sorted(data)

    if not data:
        return 0

    idx = int(len(data) * p)

    if idx >= len(data):
        idx = len(data) - 1

    return data[idx]


def print_histogram(counter, title, top=20):

    print(f"\n--- {title} ---\n")

    for value, count in counter.most_common(top):

        print(f"{value:>5}  |  {count:,}")


# ---------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------

meta_dir = LOGS_DIR / "meta"

train_pairs = load_json_if_exists(meta_dir / "train_pairs.json", [])

line_lengths = []
doc_line_counts = []

for pair in train_pairs:

    text = read_text(pair["htr_path"])

    lines = [l for l in text.splitlines() if l.strip()]

    doc_line_counts.append(len(lines))

    for line in lines:

        line_lengths.append(len(line))


# ---------------------------------------------------------------------
# Line length statistics
# ---------------------------------------------------------------------

print("\n==============================")
print("LINE LENGTH STATISTICS")
print("==============================\n")

print("Total lines:", f"{len(line_lengths):,}")

print("Min length:", min(line_lengths))
print("Median length:", median(line_lengths))
print("90th percentile:", percentile(line_lengths, 0.90))
print("95th percentile:", percentile(line_lengths, 0.95))
print("Max length:", max(line_lengths))

line_length_hist = Counter(line_lengths)

print_histogram(line_length_hist, "Most common line lengths")


# ---------------------------------------------------------------------
# Document length statistics
# ---------------------------------------------------------------------

print("\n==============================")
print("DOCUMENT LENGTH STATISTICS")
print("==============================\n")

print("Total documents:", f"{len(doc_line_counts):,}")

print("Min lines:", min(doc_line_counts))
print("Median lines:", median(doc_line_counts))
print("90th percentile:", percentile(doc_line_counts, 0.90))
print("95th percentile:", percentile(doc_line_counts, 0.95))
print("Max lines:", max(doc_line_counts))

doc_length_hist = Counter(doc_line_counts)

print_histogram(doc_length_hist, "Most common document lengths")


# ---------------------------------------------------------------------
# Bucket exploration
# ---------------------------------------------------------------------

print("\n==============================")
print("LINE BUCKET EXPLORATION")
print("==============================\n")

percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

for p in percentiles:

    value = percentile(doc_line_counts, p)

    print(f"{int(p*100):>2}th percentile: {value}")

print("\nUse these values to choose sensible line-number buckets for the report.")


print("\n==============================")
print("CHARACTER POSITION BUCKET GUIDANCE")
print("==============================\n")

print("Character position should be relative to line length.")
print("Recommended quartile buckets:")

print("0-25%")
print("25-50%")
print("50-75%")
print("75-100%")

print("\nDiagnostics complete.\n")

print("\n==============================")
print("LINE BUCKET TESTING")
print("==============================\n")

candidate_buckets = [
    (1,1),
    (2,3),
    (4,6),
    (7,10),
    (11,15),
    (16,25),
    (26,999)
]

bucket_counts = Counter()

for length in doc_line_counts:

    for low, high in candidate_buckets:

        if low <= length <= high:

            label = f"{low}-{high}" if low != high else str(low)

            bucket_counts[label] += 1

            break

for bucket, count in bucket_counts.items():

    print(f"{bucket:>6} lines  |  {count:,} documents")