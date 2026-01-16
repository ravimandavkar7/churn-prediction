# Business Problem & Objective
Customer churn is a major challenge for telecom companies, directly impacting revenue and long-term growth. The objective of this project is to identify customers who are likely to churn and provide actionable insights that help the business design effective retention strategies and reduce customer loss.

# Technical Summary
Built and optimized a telecom customer churn prediction model using Logistic Regression and Random Forest (scikit-learn). Performed exploratory data analysis, feature engineering, SMOTE oversampling, and class-weight balancing to address class imbalance. Improved churn recall from 0.50 to 0.75 through probability threshold tuning (0.35), achieving 75% recall and 0.78 ROC-AUC on the test dataset.

# Key Business Insights
Customers on month-to-month contracts show significantly higher churn rates.
High monthly charges are strongly correlated with churn probability.
Long-tenure customers demonstrate stronger loyalty and are less likely to churn.

# Business Recommendations
Introduce incentives to encourage customers to switch to long-term contracts.
Provide targeted retention offers to high-risk customers identified by the model.
Strengthen onboarding and engagement programs during the first 90 days to improve early customer retention.
