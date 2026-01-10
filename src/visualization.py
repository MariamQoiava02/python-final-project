"""
visualization.py

Basic Exploratory Data Analysis (EDA) for the
Loan Approval Analysis and Classification project.

This module focuses on understanding data distributions.
All visualizations follow a consistent green color scheme.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Global style settings ----------
plt.style.use("seaborn-v0_8-whitegrid")

BAR_COLOR = "#9BCF9B"      # light green
HIST_COLOR = "#4CAF50"     # solid green
GRID_ALPHA = 0.3


def plot_loan_status_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar chart showing the distribution of loan approval outcomes.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    ax = df["Loan_Status"].value_counts().plot(
        kind="bar",
        color=BAR_COLOR,
        edgecolor="black"
    )

    # Change x-axis labels from Y/N to Yes/No
    ax.set_xticklabels(["Yes", "No"], rotation=0)

    plt.title("Loan Approval Distribution", fontsize=14, pad=12)
    plt.xlabel("Loan Status", fontsize=11)
    plt.ylabel("Number of Applications", fontsize=11)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loan_status_distribution.png"))
    plt.close()


def plot_total_income_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Histogram showing the distribution of total household income.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(
        df["TotalIncome"],
        bins=30,
        color=HIST_COLOR,
        edgecolor="black"
    )

    plt.title("Distribution of Total Household Income", fontsize=14, pad=12)
    plt.xlabel("Total Income", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_income_distribution.png"))
    plt.close()


def plot_loan_amount_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Histogram showing the distribution of loan amount.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(
        df["LoanAmount"],
        bins=30,
        color=HIST_COLOR,
        edgecolor="black"
    )

    plt.title("Distribution of Loan Amount", fontsize=14, pad=12)
    plt.xlabel("Loan Amount", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loan_amount_distribution.png"))
    plt.close()


def run_eda(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run basic EDA focused on distributions only.
    """
    print("Running basic EDA (distributions only)...")

    plot_loan_status_distribution(df, output_dir)
    plot_total_income_distribution(df, output_dir)
    plot_loan_amount_distribution(df, output_dir)

    print("EDA completed. Figures saved to:", output_dir)