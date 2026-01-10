"""
main.py

Main entry point for the Loan Approval Analysis and Classification project.
This script orchestrates data preprocessing, visualization, and modeling.
"""

from src.data_processing import preprocess_data


def main() -> None:
    """
    Run the full data preprocessing and Phase 1 EDA pipeline.
    """
    raw_data_path = "data/raw/raw.csv"
    processed_data_path = "data/processed/loan_data_processed.csv"
    figures_dir = "reports/figures"

    try:
        # 1 Preprocess data and get DataFrame
        df = preprocess_data(
            input_path=raw_data_path,
            output_path=processed_data_path
        )
        print("\nData preprocessing completed successfully.")

        # 2️ Run Phase 1 EDA
        from src.visualization import run_eda
        run_eda(df, figures_dir)

    except Exception as exc:
        print("An error occurred during execution.")
        raise exc


if __name__ == "_main_":
    main()