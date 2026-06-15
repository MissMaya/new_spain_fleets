"""
config.py

Central configuration for the HTR diagnostic analysis pipeline.

Responsibilities:
- Define project paths relative to this file
- Ensure required directories exist
- Load the ZIP manifest
- Expose calligraphy types as configured in the manifest

All pipeline stages and utilities should import paths and pipeline
settings from this module.
"""

from pathlib import Path
import json


# ----------------------------------------------------------------------
# Project root
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Core directories
# ----------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ZIPS_DIR = PROJECT_ROOT / "zips"

SCHEMAS_DIR = PROJECT_ROOT / "schemas_and_manifests"


# ----------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------

META_DIR = LOGS_DIR / "meta"
STEP_SUMMARIES_DIR = LOGS_DIR / "step_summaries"
GROUND_TRUTHS_DIR = RAW_DIR / "ground_truths"
REVIEW_DIR = LOGS_DIR / "review"
ALLOC_DIR = REVIEW_DIR / "allocations"
POSTHOC_DIR = LOGS_DIR / "posthoc"
TABLE_DIR = POSTHOC_DIR / "corpus_report_tables"


# ----------------------------------------------------------------------
# Load zip manifest
# ----------------------------------------------------------------------

ZIP_MANIFEST_PATH = SCHEMAS_DIR / "zip_manifest.json"

if ZIP_MANIFEST_PATH.exists():
    with open(ZIP_MANIFEST_PATH, "r", encoding = "utf-8") as f:
        ZIP_MANIFEST = json.load(f)
else:
    ZIP_MANIFEST = {}


# ----------------------------------------------------------------------
# Calligraphy types
# ----------------------------------------------------------------------

CALLIGRAPHY_TYPES = ZIP_MANIFEST.get("calligraphy_types", [])


# ----------------------------------------------------------------------
# Style-specific raw directories
# ----------------------------------------------------------------------

HTR_STYLE_DIRS = {
    style: RAW_DIR / style
    for style in CALLIGRAPHY_TYPES
}


# ----------------------------------------------------------------------
# Ensure directory structure exists
# ----------------------------------------------------------------------

ALL_DIRS = [
    DATA_DIR,
    RAW_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    ZIPS_DIR,
    SCHEMAS_DIR,
    META_DIR,
    STEP_SUMMARIES_DIR,
    GROUND_TRUTHS_DIR,
    *HTR_STYLE_DIRS.values(),
]

for d in ALL_DIRS:
    d.mkdir(parents = True, exist_ok = True)