"""
rank_and_sample_reviews.py

Deterministic stratified sampling across styles and steps
Includes Step 2 token frequency analysis and line/character position diagnostics

Configuration is set at top of file.

The sampling is:
- Balanced across calligraphy styles
- Balanced across stages (S1/S2/S3)
- Deterministic through use of a seed

Outputs (written to logs/review/):
1) review_master.csv
2) sampling_metadata.json
3) diagnostics_full_pool_*  (token + stage + position diagnostics on full pool)
4) diagnostics_sampled_master_* (same diagnostics on sampled master)

Notes:
- "char_start" and "line" are absolute positions
- Line length isn't being tracked at the moment but could be added later 
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import LOGS_DIR


# ------------------------------------------------------------------
# CONFIGURATION (EDITABLE)
# ------------------------------------------------------------------

N_REVIEWERS = 10
ISSUES_PER_REVIEWER = 120

STAGE_WEIGHTS = {
    "S1": 0.30,
    "S2": 0.40,
    "S3": 0.30,
}

RANDOM_SEED = 42

TOP_N_DIAGNOSTICS = 50  # top-N tokens to write out for lexical diagnostics

# ------------------------------------------------------------------

REVIEW_DIR = LOGS_DIR / "review"
TOTAL_SAMPLE_SIZE = N_REVIEWERS * ISSUES_PER_REVIEWER

# ------------------------------------------------------------------
# Exclude any issues that have already been allocated previously 
# ------------------------------------------------------------------

def load_tracked_issue_ids() -> set:
    tracking_path = REVIEW_DIR / "review_tracking.csv"
    if not tracking_path.exists():
        return set()

    df_track = pd.read_csv(tracking_path)
    return set(df_track["issue_id"].dropna().astype(str))

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_weights(weights: dict):
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Stage weights must sum to 1.0 (got {total})")
    for k in ["S1", "S2", "S3"]:
        if k not in weights:
            raise ValueError("Stage weights must include S1, S2, S3")


# ------------------------------------------------------------------
# Stage distribution diagnostics
# ------------------------------------------------------------------

def _stage_distribution(df: pd.DataFrame) -> dict:
    """
    Return counts and proportions by stage (S1/S2/S3).
    """
    counts = df["step"].value_counts().to_dict()
    total = int(len(df))
    props = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {"counts": {k: int(v) for k, v in counts.items()},
            "proportions": {k: float(v) for k, v in props.items()},
            "total": total}


def print_stage_distribution(label: str, df: pd.DataFrame):
    dist = _stage_distribution(df)
    print(f"\n--- Stage Distribution: {label} ---")
    for stage in ["S1", "S2", "S3"]:
        c = dist["counts"].get(stage, 0)
        p = dist["proportions"].get(stage, 0.0)
        print(f"  {stage}: {c} ({p:.2%})")
    print(f"  TOTAL: {dist['total']}")


# ------------------------------------------------------------------
# Step 2 token diagnostics (deletions/insertions)
# ------------------------------------------------------------------

def _top_counts(series: pd.Series, top_n: int = 50) -> pd.DataFrame:
    """
    Returns a 2-col DataFrame: token, count (top_n).
    """
    if series is None or series.empty:
        return pd.DataFrame(columns = ["token", "count"])
    vc = series.dropna().astype(str).value_counts().head(top_n)
    return vc.reset_index().rename(columns = {"index": "token", 0: "count"})


def write_step2_token_diagnostics(df: pd.DataFrame, out_prefix: Path, top_n: int = 50):
    """
    Writes out the token-level Step 2 diagnostics.

    Deletions: S2D -> count by word_gt
    Insertions: S2I -> count by word_htr

    Produces:
      <prefix>_top_deletions.csv
      <prefix>_top_dropped_gt_tokens.csv
      <prefix>_top_insertions.csv
      <prefix>_summary.json

    Also prints top 10 of each to CLI.
    """
    s2 = df[df["step"] == "S2"].copy()

    del_df = s2[s2["tag"] == "S2D"]
    ins_df = s2[s2["tag"] == "S2I"]

    top_del = _top_counts(
        del_df["word_gt"] if "word_gt" in del_df.columns else pd.Series(dtype = object),
        top_n = top_n,
    )

    # "Dropped GT tokens" = deletions at word-op level (same as above for now)
    top_dropped_gt = top_del.copy()

    top_ins = _top_counts(
        ins_df["word_htr"] if "word_htr" in ins_df.columns else pd.Series(dtype = object),
        top_n = top_n,
    )

    out_prefix.parent.mkdir(parents = True, exist_ok = True)

    top_del.to_csv(f"{out_prefix}_top_deletions.csv", index = False)
    top_dropped_gt.to_csv(f"{out_prefix}_top_dropped_gt_tokens.csv", index = False)
    top_ins.to_csv(f"{out_prefix}_top_insertions.csv", index = False)

    summary = {
        "scope": out_prefix.name,
        "step2_total": int(len(s2)),
        "step2_deletions_total": int(len(del_df)),
        "step2_insertions_total": int(len(ins_df)),
        "top_n": int(top_n),
    }

    with open(f"{out_prefix}_step2_token_summary.json", "w", encoding = "utf-8") as f:
        json.dump(summary, f, indent = 2)

    # CLI preview (top 10)
    print(f"\n--- Step 2 Token Diagnostics: {out_prefix.name} ---")

    print("\nTop deletion words (S2D) — top 10:")
    print(top_del.head(10).to_string(index = False) if not top_del.empty else "  (none)")

    print("\nMost common dropped GT tokens — top 10:")
    print(top_dropped_gt.head(10).to_string(index = False) if not top_dropped_gt.empty else "  (none)")

    print("\nMost common inserted HTR tokens (S2I) — top 10:")
    print(top_ins.head(10).to_string(index = False) if not top_ins.empty else "  (none)")


# ------------------------------------------------------------------
# Line + character position diagnostics (absolute positions)
# ------------------------------------------------------------------

def _bin_counts(df: pd.DataFrame, col: str, bins: list[int]) -> pd.DataFrame:
    """
    Bin a numeric column into ranges and return counts per bin.
    """
    if col not in df.columns:
        return pd.DataFrame(columns=["bin", "count"])

    s = pd.to_numeric(df[col], errors = "coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns = ["bin", "count"])

    binned = pd.cut(s, bins = bins, right = True, include_lowest = True)
    out = binned.value_counts().sort_index().reset_index()
    out.columns = ["bin", "count"]
    return out


def write_line_and_char_position_diagnostics(df: pd.DataFrame, out_prefix: Path):
    """
    Writes line-number and char_start distributions overall and by calligraphy type.

    Uses:
      - line (1-based line number)
      - char_start (1-based column position on line)
    """
    # Bins (can be edited)
    line_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 10_000]
    char_bins = [0, 1, 2, 3, 5, 10, 20, 40, 80, 160, 10_000]

    # Overall
    overall_line = _bin_counts(df, "line", line_bins)
    overall_char = _bin_counts(df, "char_start", char_bins)

    overall_line.to_csv(f"{out_prefix}_line_bins_overall.csv", index = False)
    overall_char.to_csv(f"{out_prefix}_char_bins_overall.csv", index = False)

    # By style
    rows_line = []
    rows_char = []

    if "calligraphy_type" in df.columns:
        for style, df_style in df.groupby("calligraphy_type"):
            lc = _bin_counts(df_style, "line", line_bins)
            lc.insert(0, "calligraphy_type", style)
            rows_line.append(lc)

            cc = _bin_counts(df_style, "char_start", char_bins)
            cc.insert(0, "calligraphy_type", style)
            rows_char.append(cc)

    by_style_line = pd.concat(rows_line, ignore_index = True) if rows_line else pd.DataFrame(
        columns=["calligraphy_type", "bin", "count"]
    )
    by_style_char = pd.concat(rows_char, ignore_index = True) if rows_char else pd.DataFrame(
        columns=["calligraphy_type", "bin", "count"]
    )

    by_style_line.to_csv(f"{out_prefix}_line_bins_by_style.csv", index = False)
    by_style_char.to_csv(f"{out_prefix}_char_bins_by_style.csv", index = False)

    # Quantiles
    def _quantiles(series: pd.Series) -> dict:
        s = pd.to_numeric(series, errors = "coerce").dropna()
        if s.empty:
            return {}
        qs = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        return {str(k): float(v) for k, v in qs.items()}

    quant = {
        "scope": out_prefix.name,
        "overall": {
            "line_quantiles": _quantiles(df["line"]) if "line" in df.columns else {},
            "char_start_quantiles": _quantiles(df["char_start"]) if "char_start" in df.columns else {},
        },
        "by_style": {}
    }

    if "calligraphy_type" in df.columns:
        for style, df_style in df.groupby("calligraphy_type"):
            quant["by_style"][style] = {
                "line_quantiles": _quantiles(df_style["line"]) if "line" in df_style.columns else {},
                "char_start_quantiles": _quantiles(df_style["char_start"]) if "char_start" in df_style.columns else {},
            }

    with open(f"{out_prefix}_position_quantiles.json", "w", encoding = "utf-8") as f:
        json.dump(quant, f, indent = 2)

    # CLI preview
    print(f"\n--- Line/Char Position Diagnostics: {out_prefix.name} ---")
    print("\nOverall line bins (first 10):")
    print(overall_line.head(10).to_string(index = False) if not overall_line.empty else "  (none)")
    print("\nOverall char_start bins (first 10):")
    print(overall_char.head(10).to_string(index = False) if not overall_char.empty else "  (none)")


# ------------------------------------------------------------------
# Sampling per stage
# ------------------------------------------------------------------

def sample_stage_subset(df_stage: pd.DataFrame, quota: int) -> pd.DataFrame:
    """
    Sample within a stage.

    - S2: frequency-prioritised for true substitution pairs (word_gt + word_htr present),
          but DOES NOT suppress insertions/deletions (they simply get freq=0).
    - S1/S3: random.
    """
    if quota <= 0 or df_stage.empty:
        return df_stage.head(0).copy()

    if len(df_stage) <= quota:
        return df_stage.copy()

    # Step 2: frequency-prioritised
    if df_stage["step"].iloc[0] == "S2":
        df_stage = df_stage.copy()

        if "word_gt" in df_stage.columns and "word_htr" in df_stage.columns:
            # Compute frequency only on genuine pairs (avoid NaN/None artifacts)
            sub_df = df_stage.dropna(subset = ["word_gt", "word_htr"]).copy()

            if not sub_df.empty:
                freq = (
                    sub_df.groupby(["word_gt", "word_htr"])
                    .size()
                    .reset_index(name = "freq")
                )
                df_stage = df_stage.merge(freq, on = ["word_gt", "word_htr"], how = "left")
                df_stage["freq"] = df_stage["freq"].fillna(0)
                df_stage = df_stage.sort_values("freq", ascending = False)
            else:
                df_stage["freq"] = 0

        return df_stage.head(quota)

    # Step 1 & Step 3: random
    return df_stage.sample(n = quota, random_state = RANDOM_SEED, replace = False)


def _allocate_stage_quotas(style_quota: int, df_style: pd.DataFrame) -> dict:
    """
    Allocate integer quotas per stage for a given style_quota using STAGE_WEIGHTS.
    Remainder distributed to stages with available issues.
    Caps quotas to stage availability.
    """
    quotas = {stage: int(style_quota * w) for stage, w in STAGE_WEIGHTS.items()}

    avail = {
        "S1": int((df_style["step"] == "S1").sum()),
        "S2": int((df_style["step"] == "S2").sum()),
        "S3": int((df_style["step"] == "S3").sum()),
    }

    # Cap to availability
    for stage in quotas:
        quotas[stage] = min(quotas[stage], avail.get(stage, 0))

    # Redistribute remainder to stages with remaining capacity (weight order)
    assigned = sum(quotas.values())
    remainder = style_quota - assigned

    stage_order = sorted(STAGE_WEIGHTS.keys(), key = lambda s: STAGE_WEIGHTS[s], reverse = True)

    while remainder > 0:
        progressed = False
        for stage in stage_order:
            if remainder <= 0:
                break
            if quotas[stage] < avail.get(stage, 0):
                quotas[stage] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break

    return quotas


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def rank_and_sample():
    validate_weights(STAGE_WEIGHTS)

    pool_path = REVIEW_DIR / "review_pool.csv"
    if not pool_path.exists():
        raise FileNotFoundError("review_pool.csv not found. Run build_review_pool.py first.")

    df = pd.read_csv(pool_path)
    if df.empty:
        raise ValueError("Review pool is empty.")
    
    # Exclude any previously allocated issues (assuming multiple rounds of reviews)

    tracked_ids = load_tracked_issue_ids()

    if tracked_ids:
        print(f"\nExcluding {len(tracked_ids)} previously allocated issues.")
        df = df[~df["issue_id"].isin(tracked_ids)].copy()

        if df.empty:
            raise ValueError("All issues have already been allocated in previous rounds.")

    required_cols = {"issue_id", "calligraphy_type", "step", "tag"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"review_pool.csv missing required columns: {sorted(missing)}")

    styles = sorted(df["calligraphy_type"].unique())
    n_styles = len(styles)

    print("\n--- Capacity Report ---")
    print(f"Total issues in pool: {len(df)}")
    print(f"Requested total sample size: {TOTAL_SAMPLE_SIZE}")
    print(f"Reviewers: {N_REVIEWERS}")
    print(f"Issues per reviewer: {ISSUES_PER_REVIEWER}")
    print(f"Stage weights: {STAGE_WEIGHTS}")
    print(f"Random seed: {RANDOM_SEED}")

    style_counts = df.groupby("calligraphy_type").size()
    for style in styles:
        print(f"  {style}: {int(style_counts.get(style, 0))} issues")

    if TOTAL_SAMPLE_SIZE > len(df):
        print("\nWARNING: Requested sample exceeds available issues in pool.")
        total_sample = len(df)
        print(f"Adjusted total sample size to: {total_sample}")
    else:
        total_sample = TOTAL_SAMPLE_SIZE

    quota_per_style = total_sample // n_styles
    print(f"\nQuota per style (base): {quota_per_style}")

    # Full-pool diagnostics
    print_stage_distribution("full_pool", df)
    write_step2_token_diagnostics(df=df, out_prefix = REVIEW_DIR / "diagnostics_full_pool", top_n = TOP_N_DIAGNOSTICS)
    write_line_and_char_position_diagnostics(df = df, out_prefix = REVIEW_DIR / "diagnostics_full_pool")

    sampled_frames = []

    for style in styles:
        df_style = df[df["calligraphy_type"] == style].copy()

        style_quota = min(quota_per_style, len(df_style))

        stage_quotas = _allocate_stage_quotas(style_quota, df_style)

        print(f"\nStyle: {style}")
        print(f"  style_quota: {style_quota}")
        print(f"  stage_quotas: {stage_quotas} (avail: "
              f"S1 = {int((df_style['step']=='S1').sum())}, "
              f"S2 = {int((df_style['step']=='S2').sum())}, "
              f"S3 = {int((df_style['step']=='S3').sum())})")

        stage_frames = []
        for stage in ["S1", "S2", "S3"]:
            q = int(stage_quotas.get(stage, 0))
            df_stage = df_style[df_style["step"] == stage]
            stage_frames.append(sample_stage_subset(df_stage, q))

        combined = pd.concat(stage_frames) if stage_frames else df_style.head(0).copy()

        # Fill any under-allocation with remaining issues from this style
        remainder = style_quota - len(combined)
        if remainder > 0:
            remaining_pool = df_style[~df_style["issue_id"].isin(combined["issue_id"])]
            extra = remaining_pool.sample(
                n = min(remainder, len(remaining_pool)),
                random_state = RANDOM_SEED,
                replace = False,
            )
            combined = pd.concat([combined, extra])

        sampled_frames.append(combined)

    review_master = pd.concat(sampled_frames)

    # Final remainder adjustment across whole pool
    remainder_total = total_sample - len(review_master)
    if remainder_total > 0:
        remaining_pool = df[~df["issue_id"].isin(review_master["issue_id"])]
        extra = remaining_pool.sample(
            n = min(remainder_total, len(remaining_pool)),
            random_state = RANDOM_SEED,
            replace = False,
        )
        review_master = pd.concat([review_master, extra])

    review_master = review_master.sample(frac = 1, random_state = RANDOM_SEED)

    REVIEW_DIR.mkdir(parents = True, exist_ok = True)
    review_master_path = REVIEW_DIR / "review_master.csv"
    review_master.to_csv(review_master_path, index = False)

    # Sampled-master diagnostics
    print_stage_distribution("sampled_master", review_master)
    write_step2_token_diagnostics(df = review_master, out_prefix = REVIEW_DIR / "diagnostics_sampled_master", top_n = TOP_N_DIAGNOSTICS)
    write_line_and_char_position_diagnostics(df = review_master, out_prefix = REVIEW_DIR / "diagnostics_sampled_master")

    metadata = {
        "reviewers": int(N_REVIEWERS),
        "issues_per_reviewer": int(ISSUES_PER_REVIEWER),
        "requested_total": int(TOTAL_SAMPLE_SIZE),
        "actual_total": int(len(review_master)),
        "stage_weights": STAGE_WEIGHTS,
        "quota_per_style_base": int(quota_per_style),
        "seed": int(RANDOM_SEED),
        "generated_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M UTC"),
        "diagnostics_top_n": int(TOP_N_DIAGNOSTICS),
        "styles": styles,
        "stage_distribution_full_pool": _stage_distribution(df),
        "stage_distribution_sampled_master": _stage_distribution(review_master),
    }

    with open(REVIEW_DIR / "sampling_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nReview master written: {review_master_path}")
    print(f"Actual sampled issues: {len(review_master)}")


if __name__ == "__main__":
    rank_and_sample()