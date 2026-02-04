"""
bootstrap.py

Project initialisation script for the HTR cleaning pipeline.

This script creates a fresh workspace so the full 
pipeline can run reproducibly on any machine.

It does the following:

- Determines the project root dynamically
- Creates the core directory structure:
    * data/
    * logs/
    * outputs/
    * zips/
    * schemas_and_manifests/
    * utils/
- Writes utils/config.py containing global paths used across the project
- Creates schemas_and_manifests/zip_manifest.json (mapping for dataset downloading + unzipping)
- Creates schemas_and_manifests/tag_schema.json (error-tag definitions)

All files are created only if missing, so the script is safe to re-run.

Typical usage:

    python bootstrap.py

This script is automatically invoked by run_pipeline.py and normally should not
need to be run manually.
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent


def mkdir(p: Path):
    p.mkdir(parents = True, exist_ok = True)


def write_if_missing(path: Path, content: str, encoding = "utf-8"):
    mkdir(path.parent)

    if not path.exists():
        path.write_text(content, encoding = encoding)
        print(f"Created: {path}")
    else:
        print(f"Exists: {path}")


def main():
    # -----------------------
    # Create the core directories
    # -----------------------

    data = PROJECT_ROOT / "data"
    logs = PROJECT_ROOT / "logs"
    outputs = PROJECT_ROOT / "outputs"
    schemas = PROJECT_ROOT / "schemas_and_manifests"
    utils = PROJECT_ROOT / "utils"
    zips = PROJECT_ROOT / "zips"

    for d in [data, logs, outputs, schemas, utils, zips]:
        mkdir(d)

    # -----------------------
    # utils/config.py
    # -----------------------

    config_py = utils / "config.py"

    CONFIG = f"""\
    from pathlib import Path

    PROJECT_ROOT = Path(r"{PROJECT_ROOT}")

    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    TAGGED_DIR = DATA_DIR / "tagged"
    CLEANED_DIR = DATA_DIR / "cleaned"

    LOGS_DIR = PROJECT_ROOT / "logs"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    SCHEMAS_DIR = PROJECT_ROOT / "schemas_and_manifests"
    """

    write_if_missing(config_py, CONFIG)

    # -----------------------
    # zip_manifest.json
    # -----------------------

    zip_manifest = {
        "encadenada": {
            "url": "https://github.com/patymurrieta/New-Spain-Fleets/raw/main/Corpus_HTR_Encadenada_m2t4.zip",
            "unzip_to": str(PROJECT_ROOT / "data/raw/encadenada"),
        },
        "italica_cursiva": {
            "url": "https://github.com/patymurrieta/New-Spain-Fleets/raw/main/Corpus_HTR_Italica_cursiva_m3t1.zip",
            "unzip_to": str(PROJECT_ROOT / "data/raw/italica_cursiva"),
        },
        "procesal": {
            "url": "https://github.com/patymurrieta/New-Spain-Fleets/raw/main/Corpus_HTR_Procesal_m3t4.zip",
            "unzip_to": str(PROJECT_ROOT / "data/raw/procesal"),
        },
        "redonda": {
            "url": "https://github.com/patymurrieta/New-Spain-Fleets/raw/main/Corpus_HTR_Redonda_m1t3.zip",
            "unzip_to": str(PROJECT_ROOT / "data/raw/redonda"),
        },
        "ground_truths": {
            "url": "https://github.com/patymurrieta/New-Spain-Fleets/raw/main/Corpus_GT.zip",
            "unzip_to": str(PROJECT_ROOT / "data/raw/ground_truths"),
        },
    }

    write_if_missing(
        schemas / "zip_manifest.json",
        json.dumps(zip_manifest, indent = 2),
    )

    # -----------------------
    # tag_schema.json
    # -----------------------

    tag_schema = {
        "S1": {
            "L": "Leading whitespace",
            "T": "Trailing whitespace",
            "W": "Multiple / unexpected internal whitespace",
            "SP": "Space before punctuation",
            "C": "Suspicious / invisible Unicode character",
            "Z": "Zero-width Unicode character",
            "P": "Repeated or excessive punctuation",
            "G": "Non-Latin or unexpected glyph",
            "M": "Malformed or stray character",
            "MP": "Mixed punctuation inside word",
        },
        "S2": {
            "X": "Character misread (substitution)",
            "I": "Extra character in HTR (insertion)",
            "D": "Missing character in HTR (deletion)",
        },
        "S3": {
            "Q": "QU not followed by E or I",
            "WK": "Use of W or K",
            "DC": "Unexpected double consonants (excl. cc, ll, nn, rr)",
            "E": "Word ends in rare final consonant (C, F, K, M, P)",
            "T": "Triple letter repetition",
        },
    }

    write_if_missing(
        schemas / "tag_schema.json", 
        json.dumps(tag_schema, indent = 2, ensure_ascii = False),
    )


if __name__ == "__main__":
    main()
