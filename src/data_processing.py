          # Data Processing

# This file contains functions and actions for loading, cleaning, and processing the loan approval dataset.
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw loan dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError as exc:
        raise FileNotFoundError("Dataset file not found.") from exc


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.

    Parameters:
        df (pd.DataFrame): Dataset to inspect.

    """
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate strategies.

    - Numerical columns: filled with median
    - Categorical columns: filled with mode

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with corrected data types.
    """
    df = df.copy()

    # Convert Dependents from '3+' to 3
    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # Convert Credit_History to integer
    df["Credit_History"] = df["Credit_History"].astype(int)

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and cap outliers using the IQR method.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with outliers capped.
    """
    df = df.copy()

    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing data.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with new features.
    """
    df = df.copy()

    # Total income feature
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    return df

def drop_low_importance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that do not contribute meaningful information
    to analysis or modeling.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with low-importance columns removed.
    """
    df = df.copy()

    # Loan_ID is a unique identifier and has no predictive value
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to improve readability and logical structure.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with reordered columns.
    """
    df = df.copy()

    preferred_order = [
        "Loan_Status",
        "ApplicantIncome",
        "CoapplicantIncome",
        "TotalIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ]

    # Keep only columns that exist in the dataset
    existing_columns = [col for col in preferred_order if col in df.columns]

    # Add any remaining columns at the end
    remaining_columns = [col for col in df.columns if col not in existing_columns]

    return df[existing_columns + remaining_columns]

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Load data
    - Inspect data
    - Handle missing values
    - Convert data types
    - Handle outliers
    - Create features
    - Save cleaned dataset

    Parameters:
        input_path (str): Path to raw dataset.
        output_path (str): Path to save processed dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = load_data(input_path)
    inspect_data(df)

    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = handle_outliers(df)
    df = create_features(df)
    df = drop_low_importance_columns(df)
    df = reorder_columns(df)

    df.to_csv(output_path, index=False)
    print("\nProcessed data saved to:", output_path)

    return df
