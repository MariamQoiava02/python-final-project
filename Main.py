"""
Main entry point for the Loan Approval Analysis and Classification project.
"""

import os
from src.data_processing import preprocess_data
from src.visualization import run_basic_data, run_financial_driver_data, run_support_structure_data, run_loan_term_data, \
    demographic_patterns_data, run_outliers_data, run_heatmap_data
from src.models import train_logistic_regression, train_decision_tree, train_random_forest


def main() -> None:
    raw_data_path = "data/raw/raw.csv"
    processed_data_path = "data/processed/loan_data_processed.csv"
    figures_dir = "reports/figures"


    try:
        # 1: Preprocess data and get DataFrame
        df = preprocess_data(
            input_path=raw_data_path,
            output_path=processed_data_path
        )
        print("\nData preprocessing completed successfully.")

        # 2: Run Basic Data Analysis
        run_basic_data(df, figures_dir)

        # 3: Run financial Driver Analysis
        run_financial_driver_data(df, figures_dir)

        # 4: Run Support Structure Analysis
        run_support_structure_data(df, figures_dir)

        # 5: Run Loan Term and Risk Analysis
        run_loan_term_data(df, figures_dir)

        # 6: Demographic Pattern Analysis
        demographic_patterns_data(df, figures_dir)

        # 7: Outliers Analysis
        run_outliers_data(df, figures_dir)

        # 8: Correlation Heatmap Analysis
        run_heatmap_data(df, figures_dir)


            # ML Models

        # Logistic Regression Model
        train_logistic_regression()

        # Decision Tree Model
        train_decision_tree()

        # Random Forest Model
        train_random_forest()


    except Exception as exc:
        print("An error occurred during execution.")
        raise exc


if __name__ == "__main__":
    main()