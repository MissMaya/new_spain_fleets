from pathlib import Path
import json
from openpyxl import Workbook

# -----------------------------
# Paths
# -----------------------------
META_DIR = Path("logs/meta")

missing_gt_path = META_DIR / "missing_gt.json"
missing_htr_path = META_DIR / "missing_htr.json"

# -----------------------------
# Load JSON
# -----------------------------
with open(missing_gt_path, "r", encoding = "utf-8") as f:
    missing_gt = json.load(f)

with open(missing_htr_path, "r", encoding = "utf-8") as f:
    missing_htr = json.load(f)

print("Missing GT count:", len(missing_gt))
print("Missing HTR count:", len(missing_htr))

# Just take filenames
missing_gt_filenames = sorted(Path(p).name for p in missing_gt)
missing_htr_filenames = sorted(Path(p).name for p in missing_htr)

# -----------------------------
# Export to 2 sheets in Excel
# -----------------------------
wb = Workbook()

# Sheet 1: Missing_GT
ws_gt = wb.active
ws_gt.title = "Missing_GT"
ws_gt.append(["HTR Filename (No matching GT)"])

for name in missing_gt_filenames:
    ws_gt.append([name])

# Sheet 2: Missing_HTR
ws_htr = wb.create_sheet(title="Missing_HTR")
ws_htr.append(["GT Filename (No matching HTR)"])

for name in missing_htr_filenames:
    ws_htr.append([name])

# -----------------------------
# Save file
# -----------------------------
output_path = Path("missing_files_report.xlsx")
wb.save(output_path)

print("\nExcel file written to:", output_path.resolve())