"""
delete.py

Developer utility for resetting during testing.

This removes ONLY runtime-generated folders:

- data/
- logs/
- outputs/
- zips/

Source code (utils/, pipeline/, *.py) is never touched.

Intended for local experimentation and debugging.
"""

import shutil
from pathlib import Path

from utils.config import DATA_DIR, LOGS_DIR, OUTPUTS_DIR, ZIPS_DIR


def reset_project_data():
    """
    Delete all generated runtime folders so the pipeline can be rerun from scratch.

    This preserves all source code.
    """

# TODO FLAG: COULD IMPROVE THIS TO READ FROM SINGLE DIR SOURCE IN CONFIG.PY
    for d in [DATA_DIR, LOGS_DIR, OUTPUTS_DIR, ZIPS_DIR]:
        if d.exists():
            print(f"Removing {d}")
            shutil.rmtree(d)

    print("Project reset complete.")
