"""
run_pipeline.py

Main entrypoint for the HTR cleaning pipeline.

Stages:
1. Pair HTR-GT and create split
2. Run line-count diagnostics on paired HTR and GT files
3. Step 1 tagging
4. Step 2 alignment
5. Step 3 heuristics
6. Build ordered index of issues across the corpus
7. Generate distribution of step 2 tags across the corpus
8. Posthoc overlap analysis
9. Assign deterministic issue IDs
10. Build corpus diagnostics report
11. Run sample selection for human review

This pipeline produces reproducible outputs.
"""

import time

from pipeline.run_split import run_split
from pipeline.run_line_count_diagnostics import run_line_count_diagnostics
from pipeline.run_step1 import run_step1
from pipeline.run_step2 import run_step2
from pipeline.run_step3 import run_step3
from utils.posthoc_analysis import run_posthoc_analysis

from pipeline.assign_issue_ids import assign_issue_ids_all_logs
from utils.build_issue_index import build_issue_index
from utils.build_alignment_diagnostics import build_alignment_diagnostics
from pipeline.build_corpus_report import build_report
from pipeline.build_human_review_sampling import run_human_review_sampling
from pipeline.build_document_subsets import build_document_subsets

def main():
    print("Starting HTR cleaning pipeline")

    run_split()
    run_line_count_diagnostics()

    run_step1()
    run_step2()
    run_step3()

    build_issue_index()
    build_alignment_diagnostics()

    run_posthoc_analysis()
    assign_issue_ids_all_logs()
    build_report()
    run_human_review_sampling()
    build_document_subsets()

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