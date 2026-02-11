"""
config.py

Configures directory layout for the HTR cleaning pipeline.

Defines project paths relative to this file and ensures that all
required directories exist.

All pipeline stages and utilities should import paths from this module.
"""

from pathlib import Path
import json


# ----------------------------------------------------------------------
# Project root (htr_cleaning/)
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


# ----------------------------------------------------------------------
# Ensure directory structure exists
# ----------------------------------------------------------------------

# TODO FLAG: THIS COULD BE IMPROVED ON TO GET RID OF HARDCODED LIST
for d in [
    DATA_DIR,
    RAW_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    ZIPS_DIR,
    SCHEMAS_DIR,
    META_DIR,
    STEP_SUMMARIES_DIR,
]:
    d.mkdir(parents = True, exist_ok = True)


# ----------------------------------------------------------------------
# Load zip manifest if present
# ----------------------------------------------------------------------

ZIP_MANIFEST_PATH = SCHEMAS_DIR / "zip_manifest.json"

if ZIP_MANIFEST_PATH.exists():
    with open(ZIP_MANIFEST_PATH, "r", encoding = "utf-8") as f:
        ZIP_MANIFEST = json.load(f)
else:
    ZIP_MANIFEST = {}
