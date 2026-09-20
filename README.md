# aws-demo-1
Final Project of Model Deployment | Credit Score Prediction

# Credit Score Prediction

An end-to-end machine learning project for predicting customer credit scores based on financial and credit-related information. This project covers data preprocessing, model development and evaluation, as well as cloud deployment using AWS and Streamlit.

## Background

Credit scores are commonly used to represent an individual's creditworthiness and can support financial institutions in assessing potential credit risk. However, determining credit scores involves considering multiple financial factors, such as income, outstanding debt, credit history, payment behavior, and credit utilization.

This project explores the use of machine learning to classify customers into three credit score categories: **Poor, Standard, and Good**. The project was developed as an end-to-end machine learning workflow, starting from raw data preparation and model experimentation to deployment as an interactive web application.

## Objectives

The main objectives of this project are:

- Prepare and transform financial data into a suitable format for machine learning.
- Identify relevant financial and credit-related features for credit score classification.
- Develop and compare several machine learning classification models.
- Handle class imbalance to improve model performance across credit score categories.
- Optimize model performance through hyperparameter tuning.
- Evaluate the models using appropriate multiclass classification metrics.
- Deploy the trained model as a cloud-based prediction service.
- Develop an interactive web application that allows users to obtain credit score predictions.

## Dataset

The dataset contains customer financial and credit-related information used to predict credit score categories.

The target variable is **Credit_Score**, consisting of three classes:

- **Poor**
- **Standard**
- **Good**

Some of the information used for prediction includes customer income, credit history, outstanding debt, credit utilization, number of loans, delayed payments, and other financial characteristics.

The dataset contains approximately **25,000 records** before preprocessing.

> The raw dataset is not included in this repository if redistribution is restricted by the original data source.

## Project Workflow

The overall machine learning workflow consists of:

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Model Development
7. Hyperparameter Tuning
8. Model Evaluation
9. Model Packaging
10. Cloud Deployment
11. Streamlit Application

## Data Preprocessing

Several preprocessing steps were performed before model training, including:

- Removing unnecessary identifier columns.
- Handling missing and invalid values.
- Detecting and handling unrealistic numerical values and outliers.
- Transforming skewed numerical features.
- Encoding categorical variables.
- Scaling numerical features.
- Splitting the dataset into training and testing sets while maintaining the target class distribution.

## Feature Engineering

Additional features were created to better represent customers' financial conditions and credit risk.

Examples include:

- **Debt-to-Income Ratio**
- **EMI Burden**
- **Financial Exposure**
- **Risk Score**

These features combine existing financial information into variables that may provide additional information for credit score classification.

## Model Development

Three machine learning algorithms were explored and compared:

- Logistic Regression
- Random Forest
- XGBoost

The models were trained using a stratified approach to maintain the distribution of the three credit score categories.

Class imbalance was also considered during model development using techniques such as class weighting.

Hyperparameter tuning was performed using **Optuna** to search for better model configurations.

## Model Evaluation

Model performance was evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

The comparison between models was used to determine the most suitable model for the final prediction system.


## Results

Two approaches were evaluated for handling class imbalance: **SMOTE** and **Class Weight**. Hyperparameter optimization using **Optuna** was applied to Logistic Regression, Random Forest, and XGBoost.

Model performance was evaluated using Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and ROC-AUC.

### Model Comparison

| Model | Method | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | SMOTE + Optuna | 66.04% | 64.51% | 69.79% | 65.46% | 0.8111 |
| Random Forest | SMOTE + Optuna | 74.14% | 71.73% | 74.67% | 72.87% | 0.8765 |
| XGBoost | SMOTE + Optuna | 74.28% | 73.02% | 71.63% | 72.27% | 0.8757 |
| Logistic Regression | Class Weight + Optuna | 65.94% | 64.35% | 69.57% | 65.36% | 0.8132 |
| **Random Forest** | **Class Weight + Optuna** | **75.10%** | **73.52%** | 72.99% | 73.24% | **0.8812** |
| **XGBoost** | **Class Weight + Optuna** | 74.64% | 72.61% | **74.31%** | **73.37%** | 0.8792 |

### Best Performing Models

The experiments showed that **Class Weight + Optuna** provided strong overall performance for both Random Forest and XGBoost.

**Random Forest with Class Weight + Optuna** achieved the highest overall accuracy and ROC-AUC:

- **Accuracy:** 75.10%
- **Macro Precision:** 73.52%
- **Macro Recall:** 72.99%
- **Macro F1-Score:** 73.24%
- **ROC-AUC:** 0.8812

Meanwhile, **XGBoost with Class Weight + Optuna** achieved the highest Macro F1-Score and Macro Recall:

- **Accuracy:** 74.64%
- **Macro Precision:** 72.61%
- **Macro Recall:** 74.31%
- **Macro F1-Score:** 73.37%
- **ROC-AUC:** 0.8792

This shows that model selection depends on the evaluation objective. Random Forest provided the strongest overall accuracy and class-separation performance, while XGBoost provided a slightly better balance between recall and F1-score across the three credit score categories.

### Class-Level Performance

For the XGBoost model with Class Weight + Optuna, the classification results were:

| Credit Score | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Poor | 0.74 | 0.77 | 0.75 |
| Standard | 0.79 | 0.75 | 0.77 |
| Good | 0.65 | 0.72 | 0.68 |

The model performed strongest on the **Standard** and **Poor** categories. The **Good** category remained the most challenging class, although class weighting helped the model achieve a recall of 72%.

### Confusion Matrix

The confusion matrix was used to analyze prediction errors across the three credit score categories.

For the XGBoost model with Class Weight + Optuna:

| Actual / Predicted | Poor | Standard | Good |
|---|---:|---:|---:|
| Poor | 1,106 | 283 | 51 |
| Standard | 376 | 1,975 | 300 |
| Good | 15 | 243 | 651 |

The model correctly classified **1,106 Poor**, **1,975 Standard**, and **651 Good** observations from the test set.

### Key Findings

- Random Forest with Class Weight + Optuna achieved the **highest Accuracy (75.10%) and ROC-AUC (0.8812)**.
- XGBoost with Class Weight + Optuna achieved the **highest Macro F1-Score (73.37%) and Macro Recall (74.31%)**.
- Class Weight generally produced competitive results compared with SMOTE while avoiding the need to generate synthetic training samples.
- Logistic Regression showed lower performance than the tree-based models, indicating that the relationships between financial features and credit score categories may not be fully captured by a linear decision boundary.
- Random Forest and XGBoost showed similar overall performance, with different advantages depending on the evaluation metric.

## Final Model Selection

Although Random Forest achieved the highest Accuracy (75.10%) and ROC-AUC (0.8812), XGBoost with Class Weight + Optuna achieved the highest Macro Recall (74.31%) and Macro F1-Score (73.37%).

XGBoost was selected for deployment because the project prioritized balanced predictive performance across all three credit score categories rather than accuracy alone.

The final model was then packaged and integrated into the cloud deployment pipeline using AWS SageMaker and EC2.

## Deployment

After model development and evaluation, the selected model was packaged as a reusable model artifact and prepared for cloud deployment.

The deployment architecture uses several AWS services:

- **Amazon S3** for storing the model artifact.
- **Amazon SageMaker** for serving the machine learning model through a real-time inference endpoint.
- **Amazon EC2** for hosting the Streamlit web application.

The Streamlit application collects financial information from users, sends the required data for prediction, and displays the predicted credit score category.

### Deployment Architecture

```text
                    ┌─────────────────────┐
                    │        User         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit App     │
                    │      AWS EC2        │
                    └──────────┬──────────┘
                               │
                         Prediction Request
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Amazon SageMaker    │
                    │ Inference Endpoint  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Trained Model     │
                    │   Model Artifact    │
                    └─────────────────────┘
                               ▲
                               │
                    ┌─────────────────────┐
                    │      Amazon S3      │
                    │   Model Storage     │
                    └─────────────────────┘

Documentation : https://drive.google.com/drive/folders/1EdFbWz2zYBAbWpGpBoyB6NfdKXqA7iUr?usp=sharing
AWS Public Link : https://aws-demo-1-uubwmepozsrbfpza5iga7z.streamlit.app/


