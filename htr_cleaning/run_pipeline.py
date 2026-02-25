"""
run_pipeline.py

Main entrypoint for the HTR cleaning pipeline.

This script runs the full end-to-end workflow:

1. Pair HTR files with ground truth (GT) files and create train/test split
2. Run Step 1 preprocessing and basic tagging
3. Run Step 2 character-level alignment and confusion matrix analysis
4. Run Step 3 heuristic linguistic tagging
5. Run posthoc overlap analysis

To run the entire pipeline:

    python run_pipeline.py

Every stage of the pipeline should produce reproducible results. Outputs are saved as structured files.
"""
import time
from pipeline.run_split import run_split
from pipeline.run_step1 import run_step1
from pipeline.run_step2 import run_step2
from pipeline.run_step3 import run_step3
from utils.posthoc_analysis import run_posthoc_analysis
from pipeline.assign_issue_ids import assign_issue_ids_all_logs
from pipeline.build_review_pool import build_review_pool


def main():
    print("Starting HTR cleaning pipeline")

    run_split()
    run_step1()
    run_step2()
    run_step3()
    run_posthoc_analysis()
    assign_issue_ids_all_logs()
    build_review_pool()

    print("Pipeline complete.")


if __name__ == "__main__":
    start_time = time.perf_counter()

    print("\nPipeline starting...\n")

    main()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\n--- Pipeline Runtime ---")
    print(f"Total runtime: {minutes} min {seconds:.2f} secs")
    