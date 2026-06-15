"""
run_split.py

Script prepares for HTR cleaning by:

- Downloading + unzipping raw corpora
- Pairing HTR with GT
- Maintaining a stable train/test split
- Adding new files deterministically
- Invalidating cached splits if ZIP sources change
"""

from pathlib import Path
import csv
import hashlib
from datetime import datetime, timezone
import requests
import zipfile
from collections import defaultdict

from utils.config import *
from utils.file_io import read_json, safe_write_json, load_json_if_exists, index_txt_files


LOW_COUNT_THRESHOLD = 10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _stable_assign(key: str, test_ratio=0.2):
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return "test" if value < test_ratio else "train"


def _basename(path: Path):
    name = path.name

    if name.endswith("_HTR.txt"):
        return name[:-8]
    if name.endswith("_GT.txt"):
        return name[:-7]

    return path.stem


def build_htr_prefix_index(htr_files_by_style):
    index = defaultdict(list)

    for style, files in htr_files_by_style.items():
        for htr in files:
            if not htr.name.endswith("_HTR.txt"):
                continue

            stem = htr.name[:-8]
            parts = stem.split("_")

            for i in range(1, len(parts) + 1):
                prefix = "_".join(parts[:i])
                index[prefix].append({
                    "style": style,
                    "path": htr
                })

    return index


# ---------------------------------------------------------------------
# ZIP handling + cache invalidation
# ---------------------------------------------------------------------

def ensure_raw_data():
    manifest = read_json(PROJECT_ROOT / "schemas_and_manifests" / "zip_manifest.json")

    datasets = manifest.get("datasets", {})
    if not datasets:
        raise ValueError("zip_manifest.json must contain a non-empty 'datasets' object")

    zips_dir = PROJECT_ROOT / "zips"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    zips_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_meta_path = META_DIR / "zip_metadata.json"
    zip_meta = load_json_if_exists(zip_meta_path, {})

    pairing_files = [
        META_DIR / "paired_data.json",
        META_DIR / "train_pairs.json",
        META_DIR / "test_pairs.json",
        META_DIR / "missing_gt.json",
        META_DIR / "missing_htr.json",
        META_DIR / "pairing_summary.json",
        META_DIR / "split_metadata.json",
        META_DIR / "htr_index.csv",
    ]

    updated = False

    for name, entry in datasets.items():
        url = entry["url"]
        unzip_to = PROJECT_ROOT / Path(entry["unzip_to"])
        zip_path = zips_dir / f"{name}.zip"

        print(f"Checking {name}...")

        head = requests.head(url)
        head.raise_for_status()
        remote_size = int(head.headers.get("Content-Length", 0))

        prev_size = zip_meta.get(name, {}).get("size")

        if not zip_path.exists() or prev_size != remote_size:
            print(f"Downloading {name}...")
            r = requests.get(url)
            r.raise_for_status()
            zip_path.write_bytes(r.content)

            zip_meta[name] = {
                "size": remote_size,
                "downloaded_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M UTC")
            }

            updated = True
        else:
            print(f"{name} up to date")

        if not unzip_to.exists() or not any(unzip_to.iterdir()) or updated:
            print(f"Extracting {name}...")
            unzip_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(unzip_to)

    if updated:
        print("Detected ZIP updates — invalidating pairing cache")
        for p in pairing_files:
            if p.exists():
                p.unlink()

    safe_write_json(zip_meta, zip_meta_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_split(test_ratio=0.2):
    print("Starting HTR-GT pairing and stratified split...")

    META_DIR.mkdir(parents=True, exist_ok=True)
    ensure_raw_data()

    htr_files = {s: index_txt_files(RAW_DIR / s) for s in CALLIGRAPHY_TYPES}
    gt_files = index_txt_files(RAW_DIR / "ground_truths")

    gt_map = {_basename(p): p for p in gt_files}

    paired = []
    missing_gt = []
    missing_htr = []

    htr_index = build_htr_prefix_index(htr_files)

    used_htr = set()

    for gt in gt_files:
        if not gt.name.endswith("_GT.txt"):
            continue

        gt_stem = gt.name[:-7]
        matches = htr_index.get(gt_stem, [])

        if len(matches) == 0:
            missing_htr.append(str(gt))
            continue

        for match in matches:
            style = match["style"]
            htr_path = match["path"]

            paired.append({
                "id": f"{gt_stem}:{htr_path.name}",
                "style": style,
                "htr_path": str(htr_path),
                "gt_path": str(gt),
            })

            used_htr.add(str(htr_path))

    for style, files in htr_files.items():
        for htr in files:
            if str(htr) not in used_htr:
                missing_gt.append(str(htr))

    paired.sort(key=lambda p: (p["style"], p["id"]))

    train_path = META_DIR / "train_pairs.json"
    test_path = META_DIR / "test_pairs.json"

    train_pairs = load_json_if_exists(train_path, [])
    test_pairs = load_json_if_exists(test_path, [])

    assigned = {p["id"]: "train" for p in train_pairs}
    assigned.update({p["id"]: "test" for p in test_pairs})

    for p in paired:
        if p["id"] in assigned:
            continue

        split = _stable_assign(f"{p['style']}:{p['id']}", test_ratio)
        (test_pairs if split == "test" else train_pairs).append(p)

    safe_write_json(paired, META_DIR / "paired_data.json")
    safe_write_json(train_pairs, train_path)
    safe_write_json(test_pairs, test_path)
    safe_write_json(missing_gt, META_DIR / "missing_gt.json")
    safe_write_json(missing_htr, META_DIR / "missing_htr.json")

    print(f"\nTotal pairs found: {len(paired)}")
    print(f"Total missing GTs: {len(missing_gt)}")
    print(f"Total missing HTRs: {len(missing_htr)}")

    summary = defaultdict(lambda: {"train": 0, "test": 0})

    for p in train_pairs:
        summary[p["style"]]["train"] += 1
    for p in test_pairs:
        summary[p["style"]]["test"] += 1

    print("\nSplit summary:")
    for style in CALLIGRAPHY_TYPES:
        t = summary[style]["train"]
        e = summary[style]["test"]
        print(f"  - {style}: {t + e} total ({t} train / {e} test)")

    print(f"\nTotal training pairs: {len(train_pairs)}")
    print(f"Total test pairs: {len(test_pairs)}")

    for style in CALLIGRAPHY_TYPES:
        total = summary[style]["train"] + summary[style]["test"]
        if total < LOW_COUNT_THRESHOLD:
            print(f"LOW COUNT: {style}: {total}")

    safe_write_json(summary, META_DIR / "pairing_summary.json")

    safe_write_json({
        "generated_at": datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M UTC"),
        "total_pairs": len(paired),
        "train_count": len(train_pairs),
        "test_count": len(test_pairs),
        "test_ratio": test_ratio,
    }, META_DIR / "split_metadata.json")

    csv_path = META_DIR / "htr_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "style", "split"])
        for p in train_pairs:
            w.writerow([p["id"], p["style"], "train"])
        for p in test_pairs:
            w.writerow([p["id"], p["style"], "test"])

    print("\nPairing + split complete.")


if __name__ == "__main__":
    run_split()