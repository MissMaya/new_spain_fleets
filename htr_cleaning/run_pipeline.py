"""
run_pipeline.py

Main entrypoint for the HTR cleaning pipeline.

Stages:
1. Pair HTR-GT and create split
2. Step 1 tagging
3. Step 2 alignment
4. Step 3 heuristics
5. Build ordered index of issues across the corpus
6. Generate distribution of step 2 tags across the corpus
7. Posthoc overlap analysis
8. Assign deterministic issue IDs
9. Build review pool
10. Rank and sample for review
11. Allocate to reviewers

This pipeline produces reproducible outputs.
"""

import time

from pipeline.run_split import run_split
from pipeline.run_step1 import run_step1
from pipeline.run_step2 import run_step2
from pipeline.run_step3 import run_step3
from utils.posthoc_analysis import run_posthoc_analysis

from pipeline.assign_issue_ids import assign_issue_ids_all_logs
from pipeline.build_review_pool import build_review_pool
from pipeline.rank_and_sample_reviews import rank_and_sample
from pipeline.allocate_reviews import allocate_reviews
from utils.build_issue_index import build_issue_index
from utils.build_alignment_diagnostics import build_alignment_diagnostics
from pipeline.build_corpus_report import build_corpus_report

# -------------------------------------------------
# FLAGS
# -------------------------------------------------

RUN_CORE_PIPELINE = True
RUN_REVIEW_SAMPLING = True
RUN_REVIEW_ALLOCATION = True


def main():
    print("Starting HTR cleaning pipeline")

    if RUN_CORE_PIPELINE:
        run_split()

        run_step1()
        run_step2()
        run_step3()

        build_issue_index()
        build_alignment_diagnostics()

        run_posthoc_analysis()
        assign_issue_ids_all_logs()
        build_review_pool()
        build_corpus_report()

    if RUN_REVIEW_SAMPLING:
        rank_and_sample()

    if RUN_REVIEW_ALLOCATION:
        allocate_reviews()

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