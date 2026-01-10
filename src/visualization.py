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
import seaborn as sns



# ---------- Global style settings ----------
plt.style.use("seaborn-v0_8-whitegrid")

BAR_COLOR = "#9BCF9B"      # light green
HIST_COLOR = "#4CAF50"     # solid green
ACCENT_COLOR = "#7F7F7F"
NEG_COLOR = "#E57373"
GRID_ALPHA = 0.3

# for phase 3
def _is_approved(series: pd.Series) -> pd.Series:
    """
    Robust boolean Series of approval based on Loan_Status values.
    Accepts numeric 1/0, boolean, or common strings ('Y'/'N', 'Yes'/'No').
    """
    s = series.copy()
    # Numeric -> treat 1 as approved
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int) == 1
    # Boolean type
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    # Otherwise treat as string-like
    s = s.astype(str).str.strip().str.upper()
    return s.str.startswith("Y") | s.str.startswith("T")  # Y / TRUE
# for phase 3

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

# ----------------- PHASE 3: Household Structure and Support (simplified) ----------------- #
# - Approval rate bar chart (percentage) for With vs Without coapplicant (kept)
# - Grouped bar chart showing counts of Yes and No for With vs Without coapplicant (new)
# No density/KDE or boxplots included per your request.


def plot_coapplicant_approval_rates(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot approval rate (%) for applicants WITH and WITHOUT a coapplicant.
    Definition: has_coapplicant = CoapplicantIncome > 0.
    Saves: approval_rate_by_coapplicant.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "CoapplicantIncome" not in df.columns or "Loan_Status" not in df.columns:
        raise KeyError("CoapplicantIncome and/or Loan_Status column not found in DataFrame.")

    # Define groups
    has_coapp = df["CoapplicantIncome"].fillna(0) > 0
    labels = has_coapp.map({True: "With Coapplicant", False: "Without Coapplicant"})

    approved_mask = _is_approved(df["Loan_Status"])

    totals = labels.groupby(labels).size().reindex(["With Coapplicant", "Without Coapplicant"]).fillna(0).astype(int)
    approved = labels.groupby(labels).apply(lambda g: int(approved_mask[g.index].sum())).reindex(["With Coapplicant", "Without Coapplicant"]).fillna(0).astype(int)

    approval_rate = (approved / totals.replace(0, 1)) * 100  # avoid division by zero

    # Plot
    plt.figure(figsize=(7, 5))
    bars = plt.bar(approval_rate.index, approval_rate.values, color=[BAR_COLOR, ACCENT_COLOR], edgecolor="black", width=0.6)
    plt.ylim(0, 100)
    plt.ylabel("Approval Rate (%)", fontsize=11)
    plt.title("Loan Approval Rate: With vs Without Coapplicant", fontsize=14, pad=12)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # Annotate with percentages and sample sizes
    for i, label in enumerate(approval_rate.index):
        pct = approval_rate.values[i]
        n = totals[label]
        plt.text(i, pct + 2, f"{pct:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approval_rate_by_coapplicant.png"))
    plt.close()


def plot_coapplicant_status_counts_grouped(df: pd.DataFrame, output_dir: str) -> None:
    """
    Grouped bar chart showing counts of approvals (Yes) and rejections (No)
    for With Coapplicant and Without Coapplicant groups.

    Saves: coapplicant_yes_no_counts_grouped.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "CoapplicantIncome" not in df.columns or "Loan_Status" not in df.columns:
        raise KeyError("CoapplicantIncome and/or Loan_Status column not found in DataFrame.")

    # Prepare data
    has_coapp = df["CoapplicantIncome"].fillna(0) > 0
    group = has_coapp.map({True: "With Coapplicant", False: "Without Coapplicant"})
    approved = _is_approved(df["Loan_Status"])

    summary = pd.DataFrame({
        "group": group,
        "approved": approved
    })

    counts = summary.groupby(["group", "approved"]).size().unstack(fill_value=0)
    # Ensure both boolean columns exist (True/False)
    for col in [False, True]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.reindex(index=["With Coapplicant", "Without Coapplicant"]).fillna(0).astype(int)

    # Plot grouped bars: for each group, two bars (No, Yes)
    labels = counts.index.tolist()
    yes_counts = counts[True].values
    no_counts = counts[False].values

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([xi - width/2 for xi in x], yes_counts, width=width, label="Yes (Approved)", color=BAR_COLOR, edgecolor="black")
    plt.bar([xi + width/2 for xi in x], no_counts, width=width, label="No (Rejected)", color=NEG_COLOR, edgecolor="black")

    plt.xticks(x, labels)
    plt.ylabel("Number of Applications", fontsize=11)
    plt.title("Approval (Yes) vs Rejection (No): With vs Without Coapplicant", fontsize=14, pad=12)
    plt.legend()
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # Annotate bars with counts
    for i in x:
        plt.text(i - width/2, yes_counts[i] + max(1, int(0.01 * max(yes_counts.max(), no_counts.max()))), str(yes_counts[i]), ha="center", va="bottom", fontsize=9)
        plt.text(i + width/2, no_counts[i] + max(1, int(0.01 * max(yes_counts.max(), no_counts.max()))), str(no_counts[i]), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coapplicant_yes_no_counts_grouped.png"))
    plt.close()


def run_phase3_household_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """
    Runner for PHASE 3 — Household Structure and Support analysis.

    Produces:
      - approval_rate_by_coapplicant.png
      - coapplicant_yes_no_counts_grouped.png
    """
    print("Running Phase 3 — Household Structure and Support (coapplicant analysis)...")

    try:
        plot_coapplicant_approval_rates(df, output_dir)
        plot_coapplicant_status_counts_grouped(df, output_dir)
        print("Phase 3 visualizations saved to:", output_dir)
    except KeyError as exc:
        print("Required column missing for Phase 3 analysis:", exc)
        raise

# -----------------------end phase 3----------------------------------------