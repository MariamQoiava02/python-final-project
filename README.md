# Loan Approval Analysis and Classification

## Project Overview
This project analyzes a loan approval dataset to identify key factors influencing loan approval decisions and to build a machine learning model that predicts whether a loan application will be approved or not. The project follows a structured data science workflow, including data cleaning, exploratory data analysis, and classification modeling.

---

## Dataset Description
The dataset contains demographic, financial, and loan-related information about applicants. It includes both numerical and categorical variables, as well as a binary target variable indicating loan approval status.

The raw dataset is stored in the `data/raw/` directory and is not modified directly. All transformations are performed programmatically to ensure reproducibility.

---

## Data Quality Report

### Initial Data Inspection
Before cleaning, the dataset was inspected using Pandas to understand its structure and quality.

Key observations:
- Missing values were present in several columns, including `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Gender`, and `Self_Employed`.
- Some columns had inconsistent data types, such as `Dependents` containing values like `"3+"`.
- Income-related variables and loan amounts showed skewed distributions with potential outliers.
- The dataset included an identifier column (`Loan_ID`) that did not provide analytical or predictive value.

---

### Data Cleaning and Preprocessing Pipeline
A fully reproducible data preprocessing pipeline was implemented using Pandas. All steps are executed programmatically and can be rerun from the raw dataset without manual intervention.

The pipeline includes the following steps:
1. Loading the raw dataset  
2. Inspecting data structure and missing values  
3. Handling missing values  
4. Converting data types  
5. Removing low-importance columns  
6. Detecting and handling outliers  
7. Creating derived features  
8. Reordering columns for clarity  
9. Saving the processed dataset  

All preprocessing logic is implemented in `src/data_processing.py` and executed via `main.py`.

---

### Handling Missing Values
Different strategies were applied depending on the data type:

- **Numerical variables** (e.g., `LoanAmount`, `Loan_Amount_Term`)  
  Missing values were filled using the **median**, which is more robust to skewed distributions and outliers.

- **Categorical variables** (e.g., `Gender`, `Married`, `Self_Employed`)  
  Missing values were filled using the **mode**, preserving the most common category.

This approach avoids unnecessary data loss while maintaining dataset integrity.

---

### Data Type Conversions
Several columns required conversion to appropriate data types:
- `Dependents`: The value `"3+"` was converted to the integer `3`, and the column was cast to integer type.
- `Credit_History`: Converted from floating-point to integer type, as it represents a binary indicator.

These conversions ensure consistency and correct interpretation during analysis and modeling.

---

### Outlier Detection and Treatment
Outliers were identified in the following numerical columns:
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`

The **Interquartile Range (IQR)** method was used to detect outliers. Instead of removing rows, extreme values were capped at the calculated lower and upper bounds. This approach preserves the dataset size while reducing the influence of extreme values on the model.

---

### Removal of Low-Importance Columns
The column `Loan_ID` was removed during preprocessing.

**Justification:**  
`Loan_ID` is a unique identifier and does not contain information relevant to predicting loan approval outcomes. Including such identifiers in machine learning models can introduce noise without improving predictive performance.

---

### Feature Engineering
A derived feature was created to improve the dataset:

- **TotalIncome** = `ApplicantIncome` + `CoapplicantIncome`

This feature represents combined household income and provides a more realistic measure of an applicant’s financial capacity.

---

### Column Reordering
Columns were reordered to improve readability and logical structure:
- The target variable (`Loan_Status`) was placed first
- Financial variables were grouped together
- Demographic and categorical variables were placed afterward

This step improves clarity without modifying the underlying data.

---

### Final Dataset Quality
After preprocessing:
- All missing values were handled
- Data types were consistent
- Outliers were addressed
- Irrelevant columns were removed
- A clean, structured dataset was produced

The processed dataset is stored in the `data/processed/` directory and is suitable for exploratory data analysis and machine learning classification.

---

## Reproducibility
All data preprocessing steps can be reproduced by running `main.py`. The pipeline starts from the raw dataset and generates the cleaned dataset automatically.

---

## How to Run the Project
1. Install dependencies:
2. Run the main script:
3. Processed data and generated outputs will be saved to their respective directories.


## Exploratory Data Analysis (EDA)

As an initial step, we performed a basic exploratory data analysis to understand the structure and distribution of the dataset before modeling.

The analysis focuses on:
- The distribution of loan approval outcomes
- The distribution of total household income
- The distribution of loan amounts

These visualizations help identify class balance, variable scale, and overall data characteristics, ensuring that subsequent modeling steps are interpreted in the correct context.

All figures generated during EDA are saved in the `reports/figures/` directory.



Key Insights from Exploratory Data Analysis
1. Loan Approval Distribution

Approved loans significantly outnumber rejected ones, indicating a class imbalance.

This imbalance must be considered during model evaluation to avoid misleading accuracy results.

Modeling implication:
Use metrics such as precision, recall, or ROC-AUC rather than accuracy alone.

2. Household Income Effects

Total household income is right-skewed, with most applicants concentrated in mid-income ranges.

Approval rates increase slightly with income but do not increase consistently across all ranges.

Higher income improves approval chances but does not guarantee approval.

Modeling implication:
Income is an important predictor, but its effect is non-linear and should be combined with other variables.

3. Loan Amount as a Risk Indicator

Loan amounts show a concentrated core with a long tail of large loans.

Larger loan amounts are associated with higher rejection rates, even after normalizing to percentages.

The effect of loan size on rejection is stronger than the effect of income on approval.

Modeling implication:
Loan amount is a strong risk signal and should be prioritized as a core feature.

4. Relationship Between Income and Loan Amount

There is a clear positive relationship between household income and requested loan amount.

However, some applicants request large loans despite only moderate income levels, increasing rejection risk.

Modeling implication:
Interactions between income and loan amount (e.g., loan-to-income ratio) are likely informative.

5. Loan Term Patterns

Most applications use standard long-term loans (e.g., 360 months).

Approval rates vary by loan term, but rare terms show unstable patterns due to small sample sizes.

Modeling implication:
Loan term is useful but should be handled carefully to avoid noise from infrequent categories.

6. Demographic and Structural Factors

Graduates show higher approval rates than non-graduates, though the difference is moderate.

Semiurban properties have the highest approval rates, followed by urban and rural.

Applications with a coapplicant show stronger approval outcomes.

Modeling implication:
Demographic and structural variables add value but act as secondary drivers compared to financial features.

7. Outliers and Edge Cases

Extreme income and loan values are present but appear to be valid observations, not errors.

These cases likely represent high-income or high-risk applicants.

Modeling implication:
Capping extreme values is preferable to removing them, preserving information while reducing distortion.

Summary for Modeling

Primary drivers: Credit history, loan amount, household income

Secondary drivers: Education, property area, loan term, coapplicant presence

The EDA indicates that loan approval decisions are primarily driven by financial risk factors, while demographic variables provide supporting context. These insights guide feature selection, engineering, and model choice in later phases.



