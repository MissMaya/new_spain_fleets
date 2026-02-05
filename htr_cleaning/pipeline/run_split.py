"""
run_split.py

Pairs HTR files with ground truths and maintains a stable train/test split.

Design guarantees:

- Raw datasets are downloaded + extracted only if missing.
- Existing train/test assignments are NEVER changed.
- Newly discovered pairs are assigned deterministically.
- Approximate stratification is achieved via style-aware hashing.
- All artifacts are written to logs/meta/.

Console output provides:

- total pairs
- missing GT / missing HTR
- per-style pairing stats
- train/test split summary
- warning for low-count styles

Typical usage:

    python run_pipeline.py
"""

from pathlib import Path
import csv
import hashlib
from datetime import datetime
import requests
import zipfile
from collections import defaultdict

from utils.config import PROJECT_ROOT, RAW_DIR, LOGS_DIR
from utils.file_io import (
    read_json,
    safe_write_json,
    load_json_if_exists,
    index_txt_files,
)

META_DIR = LOGS_DIR / "meta"

CALLIGRAPHY_TYPES = ["encadenada", "italica_cursiva", "procesal", "redonda"]
LOW_COUNT_THRESHOLD = 10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _stable_assign(key: str, test_ratio=0.2):
    """
    Deterministically assign an item to train or test using hashing.
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return "test" if value < test_ratio else "train"


def _basename(path: Path):
    name = path.stem
    if name.endswith("_HTR"):
        name = name[:-4]
    return name


# ---------------------------------------------------------------------
# Dataset acquisition
# ---------------------------------------------------------------------

def ensure_raw_data():
    """
    Download and extract raw datasets defined in zip_manifest.json.

    ZIPs stored in zips/.
    Extracted into data/raw/<style>/.

    Idempotent:
    - ZIP downloaded only if missing
    - Extraction only if target folder empty/missing
    """

    manifest_path = PROJECT_ROOT / "schemas_and_manifests" / "zip_manifest.json"
    manifest = read_json(manifest_path)

    zips_dir = PROJECT_ROOT / "zips"
    raw_dir = PROJECT_ROOT / "data" / "raw"

    zips_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, entry in manifest.items():
        url = entry["url"]
        unzip_to = PROJECT_ROOT / entry["unzip_to"]

        zip_path = zips_dir / f"{name}.zip"

        if not zip_path.exists():
            print(f"Downloading {name}...")
            r = requests.get(url)
            r.raise_for_status()
            zip_path.write_bytes(r.content)
        else:
            print(f"ZIP exists: {name}")

        if not unzip_to.exists() or not any(unzip_to.iterdir()):
            print(f"Extracting {name}...")
            unzip_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(unzip_to)
        else:
            print(f"Raw data exists: {name}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_split(test_ratio=0.2):
    print("Starting HTR–GT pairing and stratified split...")

    ensure_raw_data()
    META_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Index files
    # --------------------------------------------------------------

    htr_files = {style: index_txt_files(RAW_DIR / style) for style in CALLIGRAPHY_TYPES}
    gt_files = index_txt_files(RAW_DIR / "ground_truths")
    gt_map = {_basename(p): p for p in gt_files}

    paired = []
    missing_gt = []
    missing_htr = []

    # --------------------------------------------------------------
    # Pair HTR with GT
    # --------------------------------------------------------------

    for style, files in htr_files.items():
        for htr in files:
            base = _basename(htr)
            if base in gt_map:
                paired.append(
                    {
                        "id": base,
                        "style": style,
                        "htr_path": str(htr),
                        "gt_path": str(gt_map[base]),
                    }
                )
            else:
                missing_gt.append(str(htr))

    htr_basenames = {_basename(p) for fs in htr_files.values() for p in fs}
    for base, gt in gt_map.items():
        if base not in htr_basenames:
            missing_htr.append(str(gt))

    # Deterministic ordering
    paired.sort(key=lambda p: (p["style"], p["id"]))

    # --------------------------------------------------------------
    # Load existing splits
    # --------------------------------------------------------------

    train_path = META_DIR / "train_pairs.json"
    test_path = META_DIR / "test_pairs.json"

    existing_train = load_json_if_exists(train_path, [])
    existing_test = load_json_if_exists(test_path, [])

    assigned = {p["id"]: "train" for p in existing_train}
    assigned.update({p["id"]: "test" for p in existing_test})

    train_pairs = list(existing_train)
    test_pairs = list(existing_test)

    # --------------------------------------------------------------
    # Assign new pairs
    # --------------------------------------------------------------

    for p in paired:
        if p["id"] in assigned:
            continue

        split = _stable_assign(f'{p["style"]}:{p["id"]}', test_ratio)
        if split == "test":
            test_pairs.append(p)
        else:
            train_pairs.append(p)

    # --------------------------------------------------------------
    # Write artifacts
    # --------------------------------------------------------------

    safe_write_json(paired, META_DIR / "paired_data.json")
    safe_write_json(train_pairs, train_path)
    safe_write_json(test_pairs, test_path)
    safe_write_json(missing_gt, META_DIR / "missing_gt.json")
    safe_write_json(missing_htr, META_DIR / "missing_htr.json")

    # --------------------------------------------------------------
    # Console summaries
    # --------------------------------------------------------------

    per_style = defaultdict(lambda: {"missing GT": 0, "missing HTR": 0, "total pairs": 0})

    for p in paired:
        per_style[p["style"]]["total pairs"] += 1

    for h in missing_gt:
        style = Path(h).parts[-2]
        per_style[style]["missing GT"] += 1

    for g in missing_htr:
        per_style["unknown"]["missing HTR"] += 1

    print(f"\nTotal pairs found: {len(paired)}")
    print(f"Total missing GTs: {len(missing_gt)}")
    print(f"Total missing HTRs: {len(missing_htr)}")
    print(dict(per_style))

    split_summary = {}
    for style in CALLIGRAPHY_TYPES:
        tr = sum(1 for p in train_pairs if p["style"] == style)
        te = sum(1 for p in test_pairs if p["style"] == style)
        split_summary[style] = {"train": tr, "test": te}

    print("\nSplit summary:")
    for style, counts in split_summary.items():
        total = counts["train"] + counts["test"]
        print(f"  - {style}: {total} total ({counts['train']} train / {counts['test']} test)")

    print(f"\nTotal training pairs: {len(train_pairs)}")
    print(f"Total test pairs: {len(test_pairs)}")

    print("\nSome styles have fewer than 10 examples:\n")
    for style, counts in split_summary.items():
        total = counts["train"] + counts["test"]
        if total < LOW_COUNT_THRESHOLD:
            print(f"  - {style}: {total} total ({counts['train']} train, {counts['test']} test)")

    # --------------------------------------------------------------
    # Metadata + CSV
    # --------------------------------------------------------------

    safe_write_json(split_summary, META_DIR / "pairing_summary.json")

    metadata = {
        "generated_at": datetime.utcnow().strftime("%d-%m-%Y %H:%M UTC"),
        "total_pairs": len(paired),
        "train_count": len(train_pairs),
        "test_count": len(test_pairs),
        "test_ratio": test_ratio,
    }

    safe_write_json(metadata, META_DIR / "split_metadata.json")

    csv_path = META_DIR / "htr_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "style", "split"])
        for p in train_pairs:
            writer.writerow([p["id"], p["style"], "train"])
        for p in test_pairs:
            writer.writerow([p["id"], p["style"], "test"])

    print("\nPairing + split complete.")


if __name__ == "__main__":
    run_split()