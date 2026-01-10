"""
main.py

Main entry point for the Loan Approval Analysis and Classification project.
This script orchestrates data preprocessing, visualization, and modeling.
"""

from src.data_processing import preprocess_data

def main() -> None:
    """
    Run the full data preprocessing pipeline.

    Returns:
        None
    """
    raw_data_path = "data/raw/raw.csv"
    processed_data_path = "data/processed/loan_data_processed.csv"

    try:
        preprocess_data(
            input_path=raw_data_path,
            output_path=processed_data_path
        )
        print("\nData preprocessing completed successfully.")
    except Exception as exc:
        print("An error occurred during preprocessing.")
        raise exc

if __name__ == "__main__":
    main()
