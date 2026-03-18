"""
allocate_reviews.py

Allocate sampled issues to individual reviewers.

Reads:
- logs/review/review_master.csv
- logs/review/sampling_metadata.json

Writes:
- logs/review/review_master_with_allocations.csv
- logs/review/allocations/review_<INITIALS>.csv
- logs/review/review_tracking.csv   (appends a new "round" of assignments)

Allocation properties:
- Deterministic (seeded)
- Balanced across calligraphy types per reviewer (as far as possible)
- Reviewer column auto-filled in individual sheets
- Reviewer sheets are simplified and use the column name "filename"
  (this is a rename of doc_id; the underlying value is unchanged).
- Multi-round safe: each allocation appends to review_tracking.csv with round_number
"""

from datetime import datetime, timezone
import json
import pandas as pd

from utils.config import LOGS_DIR
from utils.file_io import protect_for_excel


# ------------------------------------------------------------------
# CONFIGURATION (EDITABLE)
# ------------------------------------------------------------------

REVIEWER_INITIALS = [
    # Placeholders for now - will replace with actual reviewer initials 
    "REVIEWER_1", 
    "REVIEWER_2", 
    "REVIEWER_3", 
    "REVIEWER_4", 
    "REVIEWER_5", 
    "REVIEWER_6", 
    "REVIEWER_7", 
    "REVIEWER_8", 
    "REVIEWER_9", 
    "REVIEWER_10"
]

RANDOM_SEED = 42

# Allowed statuses for reviewer sheet (CSV can't enforce; validated on import)
DEFAULT_REVIEW_STATUS = "UNREVIEWED"

# Reviewer-facing output columns (minimal + clear)
REVIEWER_COLUMNS = [
    "issue_id",
    "calligraphy_type",
    "filename",          # renamed from doc_id
    "step",
    "tag",
    "description",
    "line",
    "line_char_start",   # renamed from char_start
    "line_char_end",     # renamed from char_end
    "htr_text",
    "gt_text",
    "word_gt",
    "word_htr",
    "reviewer",
    "review_status",
    "correction",
    "notes",
]

# ------------------------------------------------------------------

REVIEW_DIR = LOGS_DIR / "review"
ALLOC_DIR = REVIEW_DIR / "allocations"


def allocate_reviews():
    master_path = REVIEW_DIR / "review_master.csv"
    metadata_path = REVIEW_DIR / "sampling_metadata.json"

    if not master_path.exists():
        raise FileNotFoundError(
            "review_master.csv not found. Run rank_and_sample_reviews.py first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "sampling_metadata.json not found. Run rank_and_sample_reviews.py first."
        )

    df = pd.read_csv(master_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_reviewers = int(meta.get("reviewers", len(REVIEWER_INITIALS)))
    per_reviewer = int(meta.get("issues_per_reviewer", 0))

    if len(REVIEWER_INITIALS) != expected_reviewers:
        raise ValueError(
            f"REVIEWER_INITIALS has {len(REVIEWER_INITIALS)} entries, "
            f"but sampling_metadata.json expects {expected_reviewers} reviewers."
        )

    ALLOC_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure required columns exist in review_master
    required = {
        "issue_id",
        "calligraphy_type",
        "doc_id",
        "step",
        "tag",
        "description",
        "line",
        "char_start",
        "char_end",
        "htr_text",
        "gt_text",
        "word_gt",
        "word_htr",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"review_master.csv missing required columns: {sorted(missing)}")

    # Ensure assignment column exists
    if "assigned_reviewer" not in df.columns:
        df["assigned_reviewer"] = ""

    styles = sorted(df["calligraphy_type"].unique())
    n_reviewers = len(REVIEWER_INITIALS)

    # Prepare empty buckets
    buckets = {r: [] for r in REVIEWER_INITIALS}

    # Allocate within each style block to balance styles per reviewer
    for style in styles:
        df_style = df[df["calligraphy_type"] == style].copy()

        # Deterministic shuffle
        df_style = df_style.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        # Split as evenly as possible across reviewers
        base = len(df_style) // n_reviewers
        rem = len(df_style) % n_reviewers

        start = 0
        for i, reviewer in enumerate(REVIEWER_INITIALS):
            extra = 1 if i < rem else 0
            end = start + base + extra

            if end > start:
                chunk = df_style.iloc[start:end].copy()
                buckets[reviewer].append(chunk)

            start = end

    # Concatenate reviewer buckets and stamp assigned reviewer
    allocated_frames = []
    for reviewer, parts in buckets.items():
        if not parts:
            continue
        reviewer_df = pd.concat(parts, ignore_index=True)
        reviewer_df["assigned_reviewer"] = reviewer
        allocated_frames.append(reviewer_df)

    if not allocated_frames:
        raise ValueError("No allocations produced (review_master may be empty).")

    allocated = pd.concat(allocated_frames, ignore_index=True)

    # Final deterministic shuffle (keeps determinism but mixes styles/stages)
    allocated = allocated.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Write master-with-allocations (full; useful for audit/debug)
    allocated = protect_for_excel(allocated)
    master_out = REVIEW_DIR / "review_master_with_allocations.csv"
    allocated.to_csv(master_out, index = False, encoding = "utf-8-sig")

    # ------------------------------------------------------------
    # Write simplified per-reviewer sheets
    # ------------------------------------------------------------

    # Build a reviewer-facing view (rename columns only for reviewer sheets)
    # - "filename" is just a header rename of doc_id (value unchanged)
    # - "line_char_start/end" are header renames of char_start/end
    # - "review_status" default = UNREVIEWED
    for reviewer in REVIEWER_INITIALS:
        df_r = allocated[allocated["assigned_reviewer"] == reviewer].copy()

        # Rename headers for reviewer clarity
        df_r = df_r.rename(columns={
            "doc_id": "filename",
            "char_start": "line_char_start",
            "char_end": "line_char_end",
        })

        # Ensure required reviewer columns exist
        df_r["reviewer"] = reviewer
        if "review_status" not in df_r.columns:
            df_r["review_status"] = DEFAULT_REVIEW_STATUS
        else:
            df_r["review_status"] = df_r["review_status"].fillna(DEFAULT_REVIEW_STATUS).replace("", DEFAULT_REVIEW_STATUS)

        if "correction" not in df_r.columns:
            df_r["correction"] = ""
        if "notes" not in df_r.columns:
            df_r["notes"] = ""

        # Drop decision column if it exists for any reason
        if "decision" in df_r.columns:
            df_r = df_r.drop(columns=["decision"])

        # Keep only the simplified columns (in the requested order)
        missing_cols = [c for c in REVIEWER_COLUMNS if c not in df_r.columns]
        if missing_cols:
            raise ValueError(
                f"Reviewer sheet is missing expected columns after renaming: {missing_cols}. "
                f"Check review_master.csv columns."
            )

        df_r = df_r[REVIEWER_COLUMNS]

        out_path = ALLOC_DIR / f"review_{reviewer}.csv"
        df_r.to_csv(out_path, index=False)

    # ------------------------------------------------------------
    # Update tracking (multi-round allocations)
    # ------------------------------------------------------------

    tracking_path = REVIEW_DIR / "review_tracking.csv"

    round_number = 1
    if tracking_path.exists():
        existing = pd.read_csv(tracking_path)
        if not existing.empty and "round_number" in existing.columns:
            round_number = int(existing["round_number"].max()) + 1

    tracking_rows = allocated[["issue_id", "assigned_reviewer"]].copy()
    tracking_rows["round_number"] = round_number
    tracking_rows["assignment_timestamp"] = datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M UTC")
    tracking_rows["review_status"] = "assigned"
    tracking_rows["review_status_reviewer"] = DEFAULT_REVIEW_STATUS  # optional but useful
    tracking_rows["review_timestamp"] = ""

    if tracking_path.exists():
        tracking_rows.to_csv(tracking_path, mode="a", header=False, index=False)
    else:
        tracking_rows.to_csv(tracking_path, index=False)

    # ------------------------------------------------------------
    # CLI summary
    # ------------------------------------------------------------

    print("\nAllocation complete.")
    print(f"Round number: {round_number}")
    print(f"Master with allocations: {master_out}")
    print(f"Per-reviewer files written to: {ALLOC_DIR}")
    print(f"Tracking updated: {tracking_path}")

    counts = allocated["assigned_reviewer"].value_counts().to_dict()
    print("\nIssues per reviewer:")
    for r in REVIEWER_INITIALS:
        print(f"  {r}: {counts.get(r, 0)}")

    if per_reviewer:
        print(f"\nExpected per reviewer (from metadata): {per_reviewer}")
        print("(Not enforced as a hard constraint; style splits can cause +/- 1 variance.)")


if __name__ == "__main__":
    allocate_reviews()