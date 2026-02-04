"""
run_pipeline.py

Main entrypoint for the HTR cleaning pipeline.

This script runs the full end-to-end workflow:

1. Pair HTR files with ground truths and create train/test splits
2. Run Step 1 preprocessing and initial tagging
3. (Future) Run Step 2 character-level correction
4. (Future) Run Step 3 heuristic post-processing

Users will (in the main) only need to execute:

    python run_pipeline.py

"""

from pipeline.run_split import run_split
from pipeline.run_step1 import run_step1


def main():
    print("Starting HTR cleaning pipeline")

    run_split()
    run_step1()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
