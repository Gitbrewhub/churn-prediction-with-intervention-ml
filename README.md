# churn-prediction-with-intervention-ml
Machine learning pipeline for customer churn prediction with risk-based intervention strategies.

##Project Overview
Customer churn is a major challenge in the telecom industry, where retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to predict customer churn using machine learning and to design intervention logic that recommends retention actions based on churn risk.
Unlike traditional churn prediction projects that stop at classification, this project focuses on bridging machine learning predictions with business decision-making.

##Problem Statement
The objective of this project is to:
Predict whether a customer is likely to churn
Estimate the probability of churn
Segment customers into risk categories
Recommend targeted retention actions using intervention logic

##Dataset Description
Domain: Telecom
Data Type: Tabular customer activity data
Target Variable: churn
1 → Customer churned
0 → Customer retained
The dataset is provided in two predefined splits:
churn-80: Used for training, preprocessing, and model development
churn-20: Used only for final evaluation on unseen data
This split helps prevent data leakage and simulates real-world machine learning workflows.
( Due to privacy concerns, feature names are anonymized, which reflects real enterprise telecom datasets).

##Technologies Used
Python
Pandas, NumPy
Scikit-learn
Jupyter Notebook


##Machine Learning Pipeline
An end-to-end machine learning pipeline was implemented using scikit-learn to ensure consistency and prevent data leakage.
Pipeline Components:
Numerical feature scaling using StandardScaler
Categorical feature encoding using OneHotEncoder
Logistic Regression classifier
Probability-based predictions for business decision-making

##Model Used
Logistic Regression (Baseline Model)
Chosen for interpretability and probability estimation
Suitable for binary classification problems like churn prediction
Model Performance:
ROC-AUC Score: ~ 0.81
This indicates strong discriminatory power between churned and retained customers.

##Evaluation Metrics
The model was evaluated on the unseen churn-20 dataset using:
Precision
Recall
F1-score
ROC-AUC
In churn prediction problems, recall and ROC-AUC are prioritized to ensure churned customers are identified effectively.

##Intervention Logic (Key Feature)
| Risk Level | Churn Probability | Recommended Action           |
| ---------- | ----------------- | ---------------------------- |
| High       | ≥ 0.75            | Retention offer + agent call |
| Medium     | 0.50 – 0.75       | Personalized email/SMS       |
| Low        | < 0.50            | No action                    |

##Future Work
Cost–benefit simulation for intervention strategies
Threshold optimization for churn risk segmentation
Comparison with advanced models like Random Forest or XGBoost
Deployment as an API or dashboard  


How the run the Project?
Clone the repo = git clone Telecom_churn.ipynb
set the environment = pip install -r requirements.txt

Download the dataset:
Due to licensing constraints, the dataset is not included in this repository.
Download the Orange Telecom Customer Churn Dataset from:
Kaggle: <[https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets]>
After downloading, place the following files in the data/ directory:
data/
├── churn-80.csv
└── churn-20.csv

##Run the notebook
Open Jupyter Notebook and run:  notebooks/01_churn_pipeline_dev.ipynb


 
