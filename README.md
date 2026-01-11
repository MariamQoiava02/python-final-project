# Loan Approval Analysis and Classification

## Team Members
- Tamar Sanaia
- Mariam Koiava
- Saba Chichua
## Project Overview
This project analyzes loan application data to understand the factors that influence loan approval decisions.  
The project combines data preprocessing, exploratory data analysis (EDA), and machine learning classification models to predict whether a loan application will be approved or rejected.

## Problem Statement
Financial institutions receive many loan applications and must decide whether to approve or reject them.  
These decisions depend on factors such as income, loan amount, loan term, and applicant characteristics.  
The goal of this project is to analyze these factors and build machine learning models that can predict loan approval outcomes.

## Project Objectives
- Clean and preprocess raw loan application data
- Explore key patterns and relationships in the data
- Identify financial and demographic factors related to loan approval
- Build classification models to predict loan approval
- Compare model performance and summarize results
## Dataset Description
### Data Source
The dataset used in this project was taken from Kaggle.
It is a publicly available dataset that contains historical loan application records used for loan approval analysis and prediction tasks.
### Dataset Overview
The dataset includes information about applicants such as:
- Applicant income
- Coapplicant income
- Loan amount
- Loan term
- Education level
- Property area
- Marital status
- Employment status
- Loan approval status
### Target Variable
- **Loan_Status**  
  - `Y` → Loan approved  
  - `N` → Loan rejected  

## Project Structure
### Folder Structure

```commandline
project-root

data
  raw
    raw.csv
  processed
    loan_data_processed.csv

src
  data_processing.py
  visualization.py
  models.py

reports
  figures
    Basic Data Analysis
    Financial Data Analysis
    Support Structure Analysis
    Loan Term and Risk Analysis
    Demographic Pattern Analysis
    Outliers Analysis
    Correlation Heatmap Analysis

  Results
    Logistic Regression
    Decision Tree
    Random Forest

main.py
README.md
requirements.txt

```

### Key Files Description
- **`main.py`**  
  Main entry point of the project.  
  Runs data preprocessing, all exploratory data analysis steps, and machine learning models in sequence.

- **`src/data_processing.py`**  
  Handles data cleaning, missing values, feature creation, and saving the processed dataset.

- **`src/visualization.py`**  
  Contains all visualization functions used for exploratory data analysis, including distributions, financial drivers, demographic patterns, outliers, and correlation heatmaps.

- **`src/models.py`**  
  Contains machine learning training functions:
  - Logistic Regression  
  - Decision Tree Classifier  
  - Random Forest Classifier  
  Saves trained models, metrics, and evaluation outputs.

- **`data/raw/`**  
  Stores the original Kaggle dataset.

- **`data/processed/`**  
  Stores the cleaned and processed dataset used for modeling.

- **`reports/figures/`**  
  Contains all generated plots from the exploratory data analysis phase.

- **`reports/Results/`**  
  Contains machine learning outputs such as metrics, confusion matrices, and predictions.

## Data Preprocessing

Data preprocessing is performed before any analysis or modeling to ensure the dataset is clean, consistent, and suitable for machine learning.

### Data Cleaning
- Unnecessary columns such as ID fields are removed.
- Column names are standardized for easier handling.
- Data types are checked to ensure numerical and categorical variables are correctly identified.

### Feature Engineering
- A new feature `TotalIncome` is created by combining applicant income and coapplicant income.
- This helps represent total household financial capacity instead of treating incomes separately.

### Handling Missing Values
- Missing values in numerical columns are handled using simple statistical methods such as median values.
- Missing values in categorical columns are filled using the most frequent category.
- This approach prevents data loss while keeping the dataset consistent.

### Encoding and Scaling
- The target variable `Loan_Status` is encoded into binary values:
  - Approved (`Y`) → 1  
  - Rejected (`N`) → 0
- Categorical features are converted into numerical format using one-hot encoding.
- Numerical features are scaled when required (for Logistic Regression) to improve model performance.

## Machine Learning Models

The goal of the machine learning part of this project is to predict whether a loan application will be approved or rejected based on applicant and loan characteristics.

### Problem Type: Classification
This is a *binary classification problem* because the target variable Loan_Status has two possible outcomes:
- Approved (Y → 1)
- Rejected (N → 0)

Supervised learning models are used, as the correct outcome is known for each application.

### Logistic Regression Model
Logistic Regression is used as a baseline classification model.
- It models the probability of loan approval using a linear relationship between features and the target.
- Categorical variables are one-hot encoded.
- Numerical features are scaled to improve model performance.
- The model outputs class predictions and probabilities.

This model is easy to interpret and helps understand which factors influence approval decisions.

The Logistic Regression model was used as a baseline classification model to predict whether a loan application is approved or rejected.

Overall, the model achieved an accuracy of about 85%, which means it correctly classified most loan applications. The recall for approved loans (class 1) is very high (around 99%), showing that the model is very good at identifying applications that should be approved. This is important in a loan context, where missing eligible applicants can be costly.

However, the model performs less strongly on rejected loans (class 0). While precision for rejections is high, recall is lower, meaning some rejected cases are incorrectly predicted as approved. This imbalance is visible in the confusion matrix, where false positives (rejected loans predicted as approved) are higher than false negatives.

The ROC-AUC score (~0.86) indicates good overall discrimination ability between approved and rejected loans. This suggests the model captures meaningful patterns in the data, especially related to credit history and income-related features.

In summary, Logistic Regression provides a strong and interpretable baseline model. It is particularly effective at predicting approvals, but its performance on rejections could be improved using more complex models such as Decision Trees or Random Forests.

### Decision Tree Classifier
The Decision Tree model learns decision rules by splitting the data based on feature values.
- It can capture non-linear relationships between features.
- Feature scaling is not required.
- The model is easy to visualize and interpret.

This model helps understand how different conditions lead to approval or rejection.

The Decision Tree classifier achieved an overall accuracy of about 72%, which is slightly lower than the Logistic Regression model. The confusion matrix shows that the model correctly identified many approved loans, but it also made more mistakes compared to Logistic Regression, especially by misclassifying some approved applications as rejected.

The precision for approved loans is relatively high (~83%), meaning that when the model predicts an approval, it is often correct. However, the recall is lower (~75%), which indicates that the model fails to capture a noticeable portion of actual approved cases. This suggests that the tree is more conservative and misses some positive cases.

Overall, the Decision Tree is easy to interpret and can capture non-linear relationships in the data, but in this case it does not outperform Logistic Regression. It provides useful insights into decision rules, but its predictive performance is weaker and more sensitive to data variations.


### Random Forest Classifier (Extension)
The Random Forest model is an ensemble method built from multiple decision trees.
- It reduces overfitting compared to a single decision tree.
- It provides more stable and accurate predictions.
- Feature scaling is not required.

This model is included as an extension to compare performance with simpler models.

The Random Forest model was used as an extension to improve prediction performance compared to simpler models. It works by combining many decision trees and averaging their predictions, which helps reduce overfitting and improve generalization.

Based on the confusion matrix, the model correctly classified most loan approvals and rejections. It achieved an accuracy of about 85%, meaning it made correct predictions for the majority of applications. The model shows high recall for approved loans, indicating that it is very effective at identifying applicants who should receive a loan. Compared to the Decision Tree, the Random Forest reduced misclassifications and provided more stable results.

Overall, the Random Forest model performed slightly better than the Decision Tree and similarly to Logistic Regression, making it a strong and reliable classifier for this dataset.

## Model Evaluation

Model evaluation is used to assess how well each machine learning model predicts whether a loan application will be approved or rejected. All models are trained on the same data and evaluated on the same test set to ensure a fair and consistent comparison.

### Evaluation Metrics
The following evaluation metrics are used:
- *Accuracy*: Measures the overall proportion of correct predictions.
- *Precision*: Shows how many predicted approvals were actually approved.
- *Recall*: Measures how many actual approved loans were correctly identified.
- *F1-score*: Balances precision and recall into a single metric.

These metrics provide a complete view of model performance, especially for imbalanced datasets.

### Confusion Matrix Analysis
A confusion matrix is generated for each model to visualize prediction results.
- True Positives: Correctly predicted approved loans
- True Negatives: Correctly predicted rejected loans
- False Positives: Loans predicted as approved but actually rejected
- False Negatives: Loans predicted as rejected but actually approved

Confusion matrix heatmaps are saved for each model to help interpret classification errors.

### Model Performance Comparison
The performance of Logistic Regression, Decision Tree, and Random Forest models is compared using the same metrics.
- Logistic Regression serves as a baseline model.
- Decision Tree captures non-linear patterns in the data.
- Random Forest generally provides improved performance due to ensemble learning.

Comparing these models helps identify the most reliable approach for predicting loan approval.

## Results Summary

The results of this project show that loan approval decisions are strongly influenced by a combination of financial, support-related, and demographic factors.

From the exploratory data analysis, household income emerges as one of the most important drivers of loan approval. Applicants with higher total income generally request larger loan amounts and have higher approval rates. However, extremely high loan amounts and very long loan terms are associated with lower approval probabilities, indicating increased risk from the lender’s perspective.

Support structure also plays a role in approval outcomes. Applications with a coapplicant show a higher number of approved loans compared to those without a coapplicant, suggesting that shared financial responsibility increases approval likelihood.

Demographic patterns reveal additional differences. Graduated applicants tend to have higher approval rates than non-graduates, and applicants from semiurban areas show higher approval rates compared to rural and urban areas. Gender, marital status, and self-employment status affect the distribution of approved loans, although these factors appear less influential than income and credit-related variables.

In the modeling phase, all three machine learning models successfully learned meaningful patterns from the data. Logistic Regression provided a strong and interpretable baseline, the Decision Tree captured non-linear relationships, and the Random Forest classifier achieved the best overall performance with the most balanced predictions and lowest classification errors. 

Overall, the project demonstrates that combining exploratory analysis with machine learning models can provide reliable insights and accurate predictions for loan approval decisions.

---

## How to Run the Project

This section explains how to set up and execute the full project pipeline.

### Usage Example

After running the project, you can find:
- Cleaned data in `data/processed/loan_data_processed.csv`
- All visualizations in `reports/figures/`
- Model results in `reports/Results/`

Example use case:
- A bank analyst can use the trained models to estimate whether a new loan application is likely to be approved.
- The exploratory analysis can help understand which applicant characteristics are most important for approval decisions.

---

## Key Findings and Insights

The analysis reveals several important patterns that explain how loan approval decisions are made.

Total household income is one of the strongest predictors of loan approval. Applicants with higher combined income not only request larger loan amounts but also show consistently higher approval rates, indicating that income level is a key indicator of repayment capacity.

The presence of a coapplicant positively affects approval outcomes. Applications that include a coapplicant have a higher number of approved loans, suggesting that shared financial responsibility reduces perceived risk for lenders.

Loan characteristics also influence approval decisions. While moderate loan amounts and standard loan terms are more likely to be approved, very large loan amounts and longer repayment terms show lower approval rates. This reflects higher risk associated with long-term exposure and large credit amounts.

Demographic factors contribute additional differences. Applicants with higher education levels tend to have better approval outcomes, and property area plays a role, with semiurban applicants showing higher approval rates compared to rural and urban areas. These patterns may reflect differences in income stability and economic conditions.

From a modeling perspective, the Random Forest classifier delivers the strongest performance among all tested models. Its superior results indicate that ensemble methods are better at capturing complex interactions between income, loan characteristics, and demographic variables compared to simpler models like Logistic Regression and single Decision Trees.

Overall, the findings are consistent with real-world lending logic, where financial strength, shared responsibility, and risk-related loan features collectively shape approval decisions.

---

## Limitations and Future Improvements

This project has some limitations:
- The dataset size is relatively small, which may limit model generalization.
- Only basic feature engineering is applied.
- Hyperparameter tuning is minimal.

Possible future improvements include:
- Performing cross-validation and hyperparameter tuning
- Adding more advanced models such as Gradient Boosting or XGBoost
- Including additional financial and credit history features
- Testing the model on external datasets

---

## Conclusion

This project demonstrates a complete data science workflow from raw data to machine learning predictions.

Through exploratory data analysis, key factors affecting loan approval were identified. Machine learning models were then trained to predict approval outcomes, with Random Forest achieving the best performance.

Overall, the project shows how data analysis and machine learning can support decision-making in financial services.

