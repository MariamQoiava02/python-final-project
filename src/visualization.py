"""
Visualization
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


# Basic data analysis

# Bar chart showing how many loan applications were approved or rejected
def plot_loan_status_distribution(df: pd.DataFrame, output_dir: str) -> None:

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




# Histogram showing how total household income is distributed across all loan applicants
def plot_total_income_distribution(df: pd.DataFrame, output_dir: str) -> None:

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



# Histogram showing how requested loan amounts are distributed across all loan applications
def plot_loan_amount_distribution(df: pd.DataFrame, output_dir: str) -> None:

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



def run_basic_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run basic data analysis focused on distributions only.
    """

    print("Running basic data analysis...")

    basic_data_dir = os.path.join(output_dir, "Basic Data Analysis")

    plot_loan_status_distribution(df, basic_data_dir)
    plot_total_income_distribution(df, basic_data_dir)
    plot_loan_amount_distribution(df, basic_data_dir)

    print("Basic Data Analysis completed. Figures saved to:", basic_data_dir)



# Financial driver analysis

# Scatter plot showing the relationship between total household income and the requested loan amount for each application
def plot_income_vs_loan_amount(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["TotalIncome"],
        df["LoanAmount"],
        alpha=0.6,
        color=HIST_COLOR
    )

    plt.title("Household Income vs Loan Amount", fontsize=14, pad=12)
    plt.xlabel("Total Household Income")
    plt.ylabel("Loan Amount")
    plt.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "income_vs_loan_amount.png"))
    plt.close()



# Grouping applicants by income level and plotting the loan approval rate for each income group using a line chart
def plot_approval_rate_by_income_group(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df["Approved"] = (df["Loan_Status"] == "Y").astype(int)
    df["Income_Group"] = pd.qcut(df["TotalIncome"], q=5)

    approval_rate = (
        df.groupby("Income_Group", observed=True)["Approved"]
        .mean()
    )

    income_labels = [
        f"{int(interval.left)}–{int(interval.right)}"
        for interval in approval_rate.index
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        income_labels,
        approval_rate.values,
        marker="o",
        linewidth=2,
        color=HIST_COLOR
    )

    plt.title(
        "Loan Approval Rate Across Household Income Ranges",
        fontsize=14,
        pad=12
    )
    plt.xlabel("Total Household Income Range")
    plt.ylabel("Approval Rate")
    plt.ylim(0, 1)
    plt.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "approval_rate_by_income_range.png")
    )
    plt.close()



# 100% Stacked bar chart showing the percentage of approved and rejected loan applications across different loan amount ranges
def plot_loan_amount_distribution_by_status_percentage( df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()

    # Create loan amount groups
    df["LoanAmount_Group"] = pd.qcut(df["LoanAmount"], q=5)

    # Count approvals and rejections per group
    counts = (
        df.groupby(
            ["LoanAmount_Group", "Loan_Status"],
            observed=True
        )
        .size()
        .unstack(fill_value=0)
    )

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    labels = [
        f"{int(interval.left)}–{int(interval.right)}"
        for interval in percentages.index
    ]

    # Plot
    plt.figure(figsize=(9, 5))
    plt.bar(
        labels,
        percentages["Y"],
        label="Approved",
        color="#4CAF50",
    )
    plt.bar(
        labels,
        percentages["N"],
        bottom=percentages["Y"],
        label="Rejected",
        color="#E57373",
    )

    plt.title(
        "Loan Approval vs Rejection Rate by Loan Amount Range",
        fontsize=14,
        pad=12,
    )
    plt.xlabel("Loan Amount Range")
    plt.ylabel("Percentage of Applications")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "loan_amount_status_percentage.png")
    )
    plt.close()


def run_financial_driver_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run financial driver analysis focused on distributions only.
    """

    print("Running financial driver analysis...")

    financial_driver_data_dir = os.path.join(output_dir, "Financial Data Analysis")

    plot_income_vs_loan_amount(df, financial_driver_data_dir)
    plot_approval_rate_by_income_group(df, financial_driver_data_dir)
    plot_loan_amount_distribution_by_status_percentage(df, financial_driver_data_dir)

    print("Financial Driver Data Analysis completed. Figures saved to:", financial_driver_data_dir)


# ----------------- PHASE 3: Household Structure and Support (simplified) ----------------- #
# - Approval rate bar chart (percentage) for With vs Without coapplicant (kept)
# - Grouped bar chart showing counts of Yes and No for With vs Without coapplicant (new)
# No density/KDE or boxplots included per your request.

# Approval rate bar chart (percentage) for With vs Without coapplicant (kept)
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


# ----------------- PHASE 4: Loan Risk Structure ----------------- #
# - Loan term distribution (counts)
# - Approval rate by loan term (percentage)
# Both functions assume clean data (no NA handling) and rely on _is_approved.

def plot_loan_term_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar chart: distribution of Loan_Amount_Term values.
    X-axis: Loan Term
    Y-axis: Number of applications
    Saves: loan_term_distribution.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "Loan_Amount_Term" not in df.columns:
        raise KeyError("Loan_Amount_Term column not found in DataFrame.")

    terms = df["Loan_Amount_Term"]
    counts = terms.value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar([str(t) for t in counts.index], counts.values, color=BAR_COLOR, edgecolor="black")
    plt.title("Loan Term Distribution", fontsize=14, pad=12)
    plt.xlabel("Loan Term", fontsize=11)
    plt.ylabel("Number of Applications", fontsize=11)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # annotate counts above bars
    max_count = counts.values.max() if len(counts) > 0 else 0
    offset = max(1, int(0.01 * max_count))
    for i, v in enumerate(counts.values):
        plt.text(i, v + offset, str(int(v)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loan_term_distribution.png"))
    plt.close()

def plot_approval_rate_by_loan_term(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar chart: approval rate (%) for each Loan_Amount_Term value.
    Assumes Loan_Amount_Term and Loan_Status are clean (no NaNs).
    Saves: approval_rate_by_loan_term.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "Loan_Amount_Term" not in df.columns or "Loan_Status" not in df.columns:
        raise KeyError("Loan_Amount_Term and/or Loan_Status column not found in DataFrame.")

    approved = _is_approved(df["Loan_Status"])

    table = df[["Loan_Amount_Term"]].copy()
    table["approved"] = approved

    # Group by term and compute counts and approval rate
    agg = table.groupby("Loan_Amount_Term").agg(
        total=("approved", "size"),
        approved_count=("approved", "sum")
    )
    agg["approval_rate"] = agg["approved_count"] / agg["total"] * 100

    # Sort terms numerically
    agg = agg.sort_index()

    # Plot
    plt.figure(figsize=(10, 5))
    labels = [str(t) for t in agg.index]
    values = agg["approval_rate"].values
    plt.bar(labels, values, color=BAR_COLOR, edgecolor="black")
    plt.title("Approval Rate by Loan Term", fontsize=14, pad=12)
    plt.xlabel("Loan Term", fontsize=11)
    plt.ylabel("Approval Rate (%)", fontsize=11)
    plt.ylim(0, 110)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # Annotate each bar with percent and sample size
    for i, (term, row) in enumerate(agg.iterrows()):
        pct = row["approval_rate"]
        n = int(row["total"])
        plt.text(i, pct + 1.5, f"{pct:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approval_rate_by_loan_term.png"))
    plt.close()

def run_phase4_loan_term(df: pd.DataFrame, output_dir: str) -> None:
    """
    Runner for Phase 4 — Loan Term Distribution and Approval-by-Term charts.
    Produces:
      - loan_term_distribution.png
      - approval_rate_by_loan_term.png
    """
    print("Running Phase 4 — Loan Term Distribution and Approval-by-Term...")
    plot_loan_term_distribution(df, output_dir)
    plot_approval_rate_by_loan_term(df, output_dir)
    print("Phase 4 figures saved to:", output_dir)

# -----------------------end phase 4--------------------------------------


# ----------------- PHASE 5: Demographic and Structural Patterns ----------------- #
# - Approval rates by Education
# - Approval rates by Property_Area
# Both charts annotate sample size and median TotalIncome per group to give a simple
# relative comparison to a financial variable.

def plot_approval_rate_by_education(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar chart: approval rate (%) by Education.
    Annotates each bar with percent, sample size (n), and median TotalIncome.
    Saves: approval_rate_by_education.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "Education" not in df.columns or "Loan_Status" not in df.columns or "TotalIncome" not in df.columns:
        raise KeyError("Education, Loan_Status and/or TotalIncome column not found in DataFrame.")

    approved = _is_approved(df["Loan_Status"])

    table = df[["Education", "TotalIncome"]].copy()
    table["approved"] = approved

    # Group and aggregate
    agg = table.groupby("Education").agg(
        total=("approved", "size"),
        approved_count=("approved", "sum"),
        median_income=("TotalIncome", "median"),
    )
    agg["approval_rate"] = agg["approved_count"] / agg["total"] * 100

    # Sort labels alphabetically (keeps code simple)
    agg = agg.sort_index()

    # Plot
    plt.figure(figsize=(9, 5))
    labels = [str(x) for x in agg.index]
    values = agg["approval_rate"].values
    plt.bar(labels, values, color=BAR_COLOR, edgecolor="black")
    plt.title("Approval Rate by Education", fontsize=14, pad=12)
    plt.xlabel("Education", fontsize=11)
    plt.ylabel("Approval Rate (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # Annotate percent, n, and median income
    for i, (label, row) in enumerate(agg.iterrows()):
        pct = row["approval_rate"]
        n = int(row["total"])
        med = int(row["median_income"])
        plt.text(i, pct + 1.5, f"{pct:.1f}%\n(n={n})\nmedian income: {med:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approval_rate_by_education.png"))
    plt.close()


def plot_approval_rate_by_property_area(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar chart: approval rate (%) by Property_Area.
    Annotates each bar with percent, sample size (n), and median TotalIncome.
    Saves: approval_rate_by_property_area.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "Property_Area" not in df.columns or "Loan_Status" not in df.columns or "TotalIncome" not in df.columns:
        raise KeyError("Property_Area, Loan_Status and/or TotalIncome column not found in DataFrame.")

    approved = _is_approved(df["Loan_Status"])

    table = df[["Property_Area", "TotalIncome"]].copy()
    table["approved"] = approved

    # Group and aggregate
    agg = table.groupby("Property_Area").agg(
        total=("approved", "size"),
        approved_count=("approved", "sum"),
        median_income=("TotalIncome", "median"),
    )
    agg["approval_rate"] = agg["approved_count"] / agg["total"] * 100

    # Sort labels alphabetically
    agg = agg.sort_index()

    # Plot
    plt.figure(figsize=(9, 5))
    labels = [str(x) for x in agg.index]
    values = agg["approval_rate"].values
    plt.bar(labels, values, color=BAR_COLOR, edgecolor="black")
    plt.title("Approval Rate by Property Area", fontsize=14, pad=12)
    plt.xlabel("Property Area", fontsize=11)
    plt.ylabel("Approval Rate (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # Annotate percent, n, and median income
    for i, (label, row) in enumerate(agg.iterrows()):
        pct = row["approval_rate"]
        n = int(row["total"])
        med = int(row["median_income"])
        plt.text(i, pct + 1.5, f"{pct:.1f}%\n(n={n})\nmedian income: {med:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approval_rate_by_property_area.png"))
    plt.close()


def run_phase5_demographics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Runner for Phase 5 — Demographic and Structural Patterns.
    Produces:
      - approval_rate_by_education.png
      - approval_rate_by_property_area.png
    """
    print("Running Phase 5 — Demographic and Structural Patterns (education, property area)...")
    plot_approval_rate_by_education(df, output_dir)
    plot_approval_rate_by_property_area(df, output_dir)
    print("Phase 5 figures saved to:", output_dir)




# ----------------- PHASE 5: Demographic and Structural Patterns ----------------- #
# - Approval rates by Education
# - Approval rates by Property_Area
# Both charts annotate sample size and median TotalIncome per group to give a simple
# relative comparison to a financial variable.

def plot_approval_rate_by_education(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar chart: approval rate (%) by Education.
    Annotates each bar with percent, sample size , and median TotalIncome.
    Saves: approval_rate_by_education.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "Education" not in df.columns or "Loan_Status" not in df.columns or "TotalIncome" not in df.columns:
        raise KeyError("Education, Loan_Status and/or TotalIncome column not found in DataFrame.")

    approved = _is_approved(df["Loan_Status"])

    table = df[["Education", "TotalIncome"]].copy()
    table["approved"] = approved

    # Group and aggregate
    agg = table.groupby("Education").agg(
        total=("approved", "size"),
        approved_count=("approved", "sum"),
        median_income=("TotalIncome", "median"),
    )
    agg["approval_rate"] = agg["approved_count"] / agg["total"] * 100

    # Sort labels alphabetically (keeps code simple)
    agg = agg.sort_index()

    # Plot
    plt.figure(figsize=(9, 5))
    labels = [str(x) for x in agg.index]
    values = agg["approval_rate"].values
    plt.bar(labels, values, color=BAR_COLOR, edgecolor="black")
    plt.title("Approval Rate by Education", fontsize=14, pad=12)
    plt.xlabel("Education", fontsize=11)
    plt.ylabel("Approval Rate (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # Annotate percent, n, and median income
    for i, (label, row) in enumerate(agg.iterrows()):
        pct = row["approval_rate"]
        n = int(row["total"])
        med = int(row["median_income"])
        plt.text(i, pct + 1.5, f"{pct:.1f}%\n(n={n})\nmedian income: {med:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approval_rate_by_education.png"))
    plt.close()


def plot_approval_rate_by_property_area(df: pd.DataFrame, output_dir: str) -> None:
    """
    Bar chart: approval rate (%) by Property_Area.
    Annotates each bar with percent, sample size , and median TotalIncome.
    Saves: approval_rate_by_property_area.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if "Property_Area" not in df.columns or "Loan_Status" not in df.columns or "TotalIncome" not in df.columns:
        raise KeyError("Property_Area, Loan_Status and/or TotalIncome column not found in DataFrame.")

    approved = _is_approved(df["Loan_Status"])

    table = df[["Property_Area", "TotalIncome"]].copy()
    table["approved"] = approved

    # Group and aggregate
    agg = table.groupby("Property_Area").agg(
        total=("approved", "size"),
        approved_count=("approved", "sum"),
        median_income=("TotalIncome", "median"),
    )
    agg["approval_rate"] = agg["approved_count"] / agg["total"] * 100

    # Sort labels alphabetically
    agg = agg.sort_index()

    # Plot
    plt.figure(figsize=(9, 5))
    labels = [str(x) for x in agg.index]
    values = agg["approval_rate"].values
    plt.bar(labels, values, color=BAR_COLOR, edgecolor="black")
    plt.title("Approval Rate by Property Area", fontsize=14, pad=12)
    plt.xlabel("Property Area", fontsize=11)
    plt.ylabel("Approval Rate (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=GRID_ALPHA)

    # Annotate percent, n, and median income
    for i, (label, row) in enumerate(agg.iterrows()):
        pct = row["approval_rate"]
        n = int(row["total"])
        med = int(row["median_income"])
        plt.text(i, pct + 1.5, f"{pct:.1f}%\n(n={n})\nmedian income: {med:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approval_rate_by_property_area.png"))
    plt.close()


def run_phase5_demographics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Runner for Phase 5 — Demographic and Structural Patterns.
    Produces:
      - approval_rate_by_education.png
      - approval_rate_by_property_area.png
    """
    print("Running Phase 5 — Demographic and Structural Patterns (education, property area)...")
    plot_approval_rate_by_education(df, output_dir)
    plot_approval_rate_by_property_area(df, output_dir)
    print("Phase 5 figures saved to:", output_dir)