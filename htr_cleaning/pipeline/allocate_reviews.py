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
- Uses seed so deterministic
- Balanced as far as possible across calligraphy types
- In individual sheets, the reviewer initials are auto-filled
- If there are multiple rounds of reviews, each allocation appends to review_tracking.csv with round_number
"""

from datetime import datetime, timezone
import json
import pandas as pd

from utils.config import LOGS_DIR


# ------------------------------------------------------------------
# REVIEWER CONFIGURATION (EDITABLE )
# ------------------------------------------------------------------

REVIEWER_INITIALS = [
    # Placeholders only for the moment
    "JM", "LR", "AC", "MP", "RT", "SG", "DL", "NV", "CF", "EP"
]

RANDOM_SEED = 42

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

    with open(metadata_path, "r", encoding = "utf-8") as f:
        meta = json.load(f)

    expected_reviewers = int(meta.get("reviewers", len(REVIEWER_INITIALS)))
    per_reviewer = int(meta.get("issues_per_reviewer", 0))

    if len(REVIEWER_INITIALS) != expected_reviewers:
        raise ValueError(
            f"REVIEWER_INITIALS has {len(REVIEWER_INITIALS)} entries, "
            f"but sampling_metadata.json expects {expected_reviewers} reviewers."
        )

    ALLOC_DIR.mkdir(parents = True, exist_ok = True)

    # Ensure required columns exist
    required = {"issue_id", "calligraphy_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"review_master.csv missing required columns: {sorted(missing)}")

    # Create assigned_reviewer column if it doesn't exist
    if "assigned_reviewer" not in df.columns:
        df["assigned_reviewer"] = ""

    styles = sorted(df["calligraphy_type"].unique())
    n_reviewers = len(REVIEWER_INITIALS)

    # Prepare empty buckets
    buckets = {r: [] for r in REVIEWER_INITIALS}

    # Balance styles per reviewer by allocating within each style block
    for style in styles:
        df_style = df[df["calligraphy_type"] == style].copy()

        # Deterministic shuffle
        df_style = df_style.sample(frac=1, random_state = RANDOM_SEED).reset_index(drop = True)

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

    # Concatenate reviewer buckets
    allocated_frames = []
    for reviewer, parts in buckets.items():
        if not parts:
            continue
        reviewer_df = pd.concat(parts, ignore_index = True)
        reviewer_df["assigned_reviewer"] = reviewer
        allocated_frames.append(reviewer_df)

    if not allocated_frames:
        raise ValueError("No allocations produced (review_master may be empty).")

    allocated = pd.concat(allocated_frames, ignore_index=True)

    # Final deterministic shuffle
    allocated = allocated.sample(frac = 1, random_state = RANDOM_SEED).reset_index(drop=True)

    # Write out the master allocation sheet
    master_out = REVIEW_DIR / "review_master_with_allocations.csv"
    allocated.to_csv(master_out, index = False)

    # Write per-reviewer sheets
    for reviewer in REVIEWER_INITIALS:
        df_r = allocated[allocated["assigned_reviewer"] == reviewer].copy()

        # Ensure reviewer column is auto-filled for each sheet
        df_r["reviewer"] = reviewer

        out_path = ALLOC_DIR / f"review_{reviewer}.csv"
        df_r.to_csv(out_path, index = False)

    # ------------------------------------------------------------
    # Update tracking (in the case of multi-round allocations)
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
    tracking_rows["decision"] = ""
    tracking_rows["review_timestamp"] = ""

    if tracking_path.exists():
        tracking_rows.to_csv(tracking_path, mode = "a", header = False, index = False)
    else:
        tracking_rows.to_csv(tracking_path, index = False)

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


if __name__ == "__main__":
    allocate_reviews()