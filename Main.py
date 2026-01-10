"""
main.py

Main entry point for the Loan Approval Analysis and Classification project.
This script orchestrates data preprocessing, visualization, and modeling.
"""

import os
from src.data_processing import preprocess_data
from src.visualization import run_eda, run_phase3_household_analysis


def main() -> None:
    """
    Run the full data preprocessing and Phase 1/Phase 3 EDA pipeline.
    """
    raw_data_path = "data/raw/raw.csv"
    processed_data_path = "data/processed/loan_data_processed.csv"
    figures_dir = "reports/figures"
    phase3_dir = os.path.join(figures_dir)

    try:
        # 1 Preprocess data and get DataFrame
        df = preprocess_data(
            input_path=raw_data_path,
            output_path=processed_data_path
        )
        print("\nData preprocessing completed successfully.")

        # 2 Run Phase 1 EDA (existing)
        run_eda(df, figures_dir)

        # 3 Run Phase 3 — Household Structure and Support (coapplicant analysis)
        run_phase3_household_analysis(df, phase3_dir)

    except Exception as exc:
        print("An error occurred during execution.")
        raise exc


if __name__ == "__main__":
    main()