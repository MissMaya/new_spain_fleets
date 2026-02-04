"""
run_split.py

Pairs HTR files with ground truths and maintains a stable train/test split.

- Existing assignments are never changed - for example, once an HTR file is in the train set, it stays there.
- New files are added deterministically.
- Splits are approximately stratified by calligraphy style.
- Logs track which files are in the training set, which ones are in train and which are incomplete pairs.

Typical usage:

    python pipeline/run_split.py

or via:

    python run_pipeline.py
"""

from pathlib import Path
import csv
import hashlib
from datetime import datetime

from utils.config import RAW_DIR, LOGS_DIR
from utils.file_io import (
    write_json,
    safe_write_json,
    load_json_if_exists,
    index_txt_files,
)


META_DIR = LOGS_DIR / "meta"


CALLIGRAPHY_TYPES = ["encadenada", "italica_cursiva", "procesal", "redonda"]


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


def run_split(test_ratio=0.2):
    print("Starting HTR–GT pairing and stratified split...")

    META_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Index HTR files by style
    # ------------------------------------------------------------------

    htr_files = {}
    for style in CALLIGRAPHY_TYPES:
        htr_files[style] = index_txt_files(RAW_DIR / style)

    gt_files = index_txt_files(RAW_DIR / "ground_truths")

    gt_map = {_basename(p): p for p in gt_files}

    paired = []
    missing_gt = []
    missing_htr = []

    # ------------------------------------------------------------------
    # Pair HTR with GT
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Load existing splits if present
    # ------------------------------------------------------------------

    train_path = META_DIR / "train_pairs.json"
    test_path = META_DIR / "test_pairs.json"

    existing_train = load_json_if_exists(train_path, [])
    existing_test = load_json_if_exists(test_path, [])

    assigned = {p["id"]: "train" for p in existing_train}
    assigned.update({p["id"]: "test" for p in existing_test})

    train_pairs = list(existing_train)
    test_pairs = list(existing_test)

    # ------------------------------------------------------------------
    # Assign new pairs deterministically
    # ------------------------------------------------------------------

    for p in paired:
        if p["id"] in assigned:
            continue

        split = _stable_assign(f'{p["style"]}:{p["id"]}', test_ratio)

        if split == "test":
            test_pairs.append(p)
        else:
            train_pairs.append(p)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------

    safe_write_json(paired, META_DIR / "paired_data.json")
    safe_write_json(train_pairs, train_path)
    safe_write_json(test_pairs, test_path)
    safe_write_json(missing_gt, META_DIR / "missing_gt.json")
    safe_write_json(missing_htr, META_DIR / "missing_htr.json")

    summary = {}
    for style in CALLIGRAPHY_TYPES:
        summary[style] = {
            "train": sum(1 for p in train_pairs if p["style"] == style),
            "test": sum(1 for p in test_pairs if p["style"] == style),
        }

    safe_write_json(summary, META_DIR / "pairing_summary.json")

    metadata = {
        "generated_at": datetime.utcnow().strftime("%d-%m-%Y %H:%M UTC"),
        "total_pairs": len(paired),
        "train_count": len(train_pairs),
        "test_count": len(test_pairs),
        "test_ratio": test_ratio,
    }

    safe_write_json(metadata, META_DIR / "split_metadata.json")

    # ------------------------------------------------------------------
    # Human-readable CSV index
    # ------------------------------------------------------------------

    csv_path = META_DIR / "htr_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "style", "split"])

        for p in train_pairs:
            writer.writerow([p["id"], p["style"], "train"])
        for p in test_pairs:
            writer.writerow([p["id"], p["style"], "test"])

    print("Pairing + split complete.")
    print(f"Train: {len(train_pairs)} | Test: {len(test_pairs)}")


if __name__ == "__main__":
    run_split()
