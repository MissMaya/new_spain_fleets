"""
run_pipeline.py

Main entrypoint for the HTR cleaning pipeline.

This script runs the full end-to-end workflow:

1. Pair HTR files with ground truths and create train/test splits
2. Run Step 1 preprocessing and surface tagging
3. Run Step 2 character-level alignment and confusion analysis
4. Run Step 3 heuristic linguistic tagging
5. Run posthoc overlap analysis

Non-technical users only need to execute:

    python run_pipeline.py

All pipeline stages are deterministic and write explicit outputs to disk.
"""

from pipeline.run_split import run_split
from pipeline.run_step1 import run_step1
from pipeline.run_step2 import run_step2
from pipeline.run_step3 import run_step3
from utils.posthoc_analysis import run_posthoc_analysis


def main():
    print("Starting HTR cleaning pipeline")

    run_split()
    run_step1()
    run_step2()
    run_step3()
    run_posthoc_analysis()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
    