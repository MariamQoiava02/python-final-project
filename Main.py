# Loan Approval Analysis and Classification
# Basic setup and data loading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

print("Libraries imported successfully")

# Load the dataset
df = pd.read_csv("Data/raw/raw.csv")

# Basic dataset inspection
print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())



# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------

print("\nLoan Status Distribution:")
print(df["Loan_Status"].value_counts())

print("\nLoan Status Percentage:")
print(df["Loan_Status"].value_counts(normalize=True) * 100)
