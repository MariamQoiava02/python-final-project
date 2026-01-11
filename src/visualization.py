"""
Visualization
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Styles
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




# Support Structure Analysis

# Grouped bar chart showing counts of Yes and No for With vs Without coapplicant
def plot_coapplicant_status_counts_grouped(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    if "CoapplicantIncome" not in df.columns or "Loan_Status" not in df.columns:
        raise KeyError("CoapplicantIncome and/or Loan_Status column not found in DataFrame.")

    # Prepare data
    has_coapp = df["CoapplicantIncome"].fillna(0) > 0
    group = has_coapp.map({True: "With Coapplicant", False: "Without Coapplicant"})
    approved = df["Loan_Status"]=='Y'

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

def run_support_structure_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run support structure analysis
    """

    print("Running support structure analysis...")

    support_structure_data_dir = os.path.join(output_dir, "Support Structure Analysis")

    plot_coapplicant_status_counts_grouped(df, support_structure_data_dir)


    print("Support Structure Analysis completed. Figures saved to:", support_structure_data_dir)



# Loan Term and Risk Structure


# Bar chart showing the distribution of loan terms
def plot_loan_term_distribution(df: pd.DataFrame, output_dir: str) -> None:

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



# bar chart showing the approval rate percentage for each term
def plot_approval_rate_by_loan_term(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    if "Loan_Amount_Term" not in df.columns or "Loan_Status" not in df.columns:
        raise KeyError("Loan_Amount_Term and/or Loan_Status column not found in DataFrame.")

    approved = df["Loan_Status"]=='Y'

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



def run_loan_term_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run Loan Term and Risk Analysis
    """

    print("Running loan term and risk analysis...")

    loan_term_data_dir = os.path.join(output_dir, "Loan term and Risk Analysis")

    plot_loan_term_distribution(df, loan_term_data_dir)
    plot_approval_rate_by_loan_term(df, loan_term_data_dir)


    print("Loan Term and Risk Analysis completed. Figures saved to:", loan_term_data_dir)






# Demographic Pattern Analysis


# Bar chart showing the approval rates by Education
def plot_approval_rate_by_education(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    if "Education" not in df.columns or "Loan_Status" not in df.columns or "TotalIncome" not in df.columns:
        raise KeyError("Education, Loan_Status and/or TotalIncome column not found in DataFrame.")

    approved = df["Loan_Status"]=='Y'

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



# Bar chart showing the approval rate based on Property
def plot_approval_rate_by_property_area(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    if "Property_Area" not in df.columns or "Loan_Status" not in df.columns or "TotalIncome" not in df.columns:
        raise KeyError("Property_Area, Loan_Status and/or TotalIncome column not found in DataFrame.")

    approved = df["Loan_Status"]=='Y'

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



def plot_approved_loans_by_gender(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Pie chart showing gender distribution among approved loans
    using a green color palette.
    """
    os.makedirs(output_dir, exist_ok=True)

    approved_df = df[df["Loan_Status"] == "Y"]
    counts = approved_df["Gender"].value_counts()

    # Green color palette
    colors = ["#2E7D32", "#66BB6A"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )

    plt.title(
        "Approved Loans by Gender",
        fontsize=14,
        pad=14,
        fontweight="bold",
    )
    plt.axis("equal")  # ensures perfect circle

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "approved_loans_by_gender.png")
    )
    plt.close()



def plot_approved_loans_by_marital_status(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Pie chart showing marital status distribution among approved loans
    using a green color palette.
    """
    os.makedirs(output_dir, exist_ok=True)

    approved_df = df[df["Loan_Status"] == "Y"]
    counts = approved_df["Married"].value_counts()

    # Green color palette
    colors = ["#1B5E20", "#81C784"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )

    plt.title(
        "Approved Loans by Marital Status",
        fontsize=14,
        pad=14,
        fontweight="bold",
    )
    plt.axis("equal")  # ensures perfect circle

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "approved_loans_by_marital_status.png")
    )
    plt.close()




def plot_approved_loans_by_self_employment(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Pie chart showing self-employment distribution among approved loans
    using a green color palette.
    """
    os.makedirs(output_dir, exist_ok=True)

    approved_df = df[df["Loan_Status"] == "Y"]
    counts = approved_df["Self_Employed"].value_counts()

    # Green color palette
    colors = ["#2E7D32", "#A5D6A7"]

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )

    plt.title(
        "Approved Loans by Self-Employment Status",
        fontsize=14,
        pad=14,
        fontweight="bold",
    )
    plt.axis("equal")  # ensures perfect circle

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "approved_loans_by_self_employment.png")
    )
    plt.close()




def demographic_patterns_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run Demographic Pattern Analysis
    """
    print("Running demographic pattern analysis...")

    demographic_data_dir = os.path.join(output_dir, "Demographic Pattern Analysis")

    plot_approval_rate_by_education(df, demographic_data_dir)
    plot_approval_rate_by_property_area(df, demographic_data_dir)
    plot_approved_loans_by_gender(df, demographic_data_dir)
    plot_approved_loans_by_marital_status(df, demographic_data_dir)
    plot_approved_loans_by_self_employment(df, demographic_data_dir)


    print("Demographic Pattern Analysis completed. Figures saved to:", demographic_data_dir)



# Outliers Analysis

# Returns lower and upper bounds using the IQR rule
def _iqr_bounds(series: pd.Series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


# Combined boxplot for the provided columns
def plot_boxplot_for_columns(df: pd.DataFrame, columns: list, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # Ensure columns exist
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"{col} column not found in DataFrame.")

    values = [df[col].values for col in columns]

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(values, labels=columns, patch_artist=True, vert=False)
    # style boxes
    for box in bp["boxes"]:
        box.set_facecolor(BAR_COLOR)
        box.set_edgecolor("black")

    plt.title("Boxplots — Income and Loan Amount (outliers shown as fliers)", fontsize=14, pad=12)
    plt.xlabel("Amount", fontsize=11)
    plt.grid(axis="x", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outliers_boxplots.png"))
    plt.close()


# Creating one simple boxplot per variable and save separately for inspection.
def plot_individual_boxplots(df: pd.DataFrame, columns: list, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for col in columns:
        plt.figure(figsize=(8, 2.5))
        plt.boxplot(df[col].values, vert=False, patch_artist=True)
        plt.title(f"Boxplot — {col}", fontsize=12)
        plt.xlabel(col, fontsize=10)
        plt.grid(axis="x", alpha=GRID_ALPHA)
        plt.tight_layout()
        filename = f"outlier_box_{col}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()



# Computing IQR bounds and counting of points below/above bounds for each column.
def save_outlier_summary(df: pd.DataFrame, columns: list, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for col in columns:
        series = df[col]
        lower, upper = _iqr_bounds(series)
        below = int((series < lower).sum())
        above = int((series > upper).sum())
        total = int(series.size)
        rows.append({
            "variable": col,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "count_below": below,
            "count_above": above,
            "total_count": total,
            "pct_below": below / total * 100.0,
            "pct_above": above / total * 100.0,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "outliers_summary.csv"), index=False)



def run_outliers_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run Outliers Analysis
    Produces:
      - outliers_boxplots.png (combined)
      - outlier_box_<column>.png (individual)
      - outliers_summary.csv (IQR bounds and counts)
    """
    print("Running Outliers Analysis...")

    cols = ["ApplicantIncome", "CoapplicantIncome", "TotalIncome", "LoanAmount"]

    outliers_dir = os.path.join(output_dir, "Outliers Analysis")

    plot_boxplot_for_columns(df, cols, outliers_dir)
    plot_individual_boxplots(df, cols, outliers_dir)
    save_outlier_summary(df, cols, outliers_dir)

    print("Outliers Analysis completed. Figures saved to:", outliers_dir)




def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Correlation heatmap of numerical features and loan approval.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()

    # Encode Loan_Status for correlation
    df["Loan_Status_Encoded"] = (df["Loan_Status"] == "Y").astype(int)

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        cmap="Greens",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"}
    )

    plt.title(
        "Correlation Heatmap of Loan Features",
        fontsize=14,
        pad=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "correlation_heatmap.png")
    )
    plt.close()


def run_heatmap_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Run Correlation Heatmap Analysis
    """
    print("Running Correlation Heatmap Analysis...")

    heatmap_dir = os.path.join(output_dir, "Correlation Heatmap Analysis")

    plot_correlation_heatmap(df, heatmap_dir)

    print("Correlation Heatmap Analysis completed. Figures saved to:", heatmap_dir)

