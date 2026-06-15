from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openpyxl import load_workbook
import pandas as pd

from utils.config import LOGS_DIR, OUTPUTS_DIR
from utils.file_io import load_json_if_exists, read_text, protect_for_excel


META_DIR = LOGS_DIR / "meta"

def line_count(path: str | Path) -> int:
    text = read_text(Path(path))
    lines = [l for l in text.splitlines() if l.strip()]
    return len(lines)

def build_line_length_dataframe() -> pd.DataFrame:
    pairs = load_json_if_exists(META_DIR / "paired_data.json", [])

    rows = []

    for pair in pairs:
        gt_path = Path(pair["gt_path"])
        htr_path = Path(pair["htr_path"])

        gt_len = line_count(gt_path)
        htr_len = line_count(htr_path)

        rows.append({
            "Filename": htr_path.name,
            "GT line count": gt_len,
            "HTR line count": htr_len,
            "Delta": gt_len - htr_len,
            "calligraphy type": pair["style"],
        })

    return pd.DataFrame(rows)


def main():
    OUTPUTS_DIR.mkdir(parents = True, exist_ok = True)

    df = build_line_length_dataframe()

    output_path = OUTPUTS_DIR / "gt_htr_line_counts.xlsx"
    df.to_excel(output_path, index = False)

    wb = load_workbook(output_path)
    ws = wb.active

    ws.freeze_panes = "A2"

    wb.save(output_path)

    print(f"Rows written: {len(df):,}")
    print(f"Output: {output_path}")



if __name__ == "__main__":
    main()