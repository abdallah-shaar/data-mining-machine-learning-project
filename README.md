# Data Mining & Machine Learning Project

This project presents a data mining and machine learning analysis using Python, Jupyter Notebook, and CSV datasets.

## Project Description

The project includes two main parts:

1. Heart Disease Classification  
   A machine learning classification task using the heart disease dataset. The project applies Decision Tree and K-Nearest Neighbors (KNN) models to classify the target variable and compare model performance.

2. Online Retail Association Rules Mining  
   A data mining task using online retail transaction data to discover purchasing patterns and relationships between products using association rules.

## Project Files

This repository includes:

- `heart.csv`  
  Heart disease dataset used for classification.

- `Online Retail.csv`  
  Sample online retail transaction dataset used for association rules mining.

- `output.ipynb`  
  Jupyter Notebook output showing the analysis process, model results, charts, and evaluation outputs.

- `data_mining_project.py`  
  Python source code for the full data mining and machine learning project.

## Tools and Libraries Used

- Python
- Jupyter Notebook
- Microsoft Excel
- NumPy
- Matplotlib
- Scikit-learn
- CSV data processing

## Methods Used

### 1. Classification

The heart disease dataset was analyzed using machine learning classification models:

- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- KNN with feature scaling

The models were evaluated using:

- Accuracy
- Macro Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### 2. Association Rules Mining

The online retail dataset was used to discover product relationships based on customer transactions.

The analysis included:

- Grouping products by invoice number
- Creating transaction baskets
- Calculating item support
- Finding frequent item pairs
- Generating association rules
- Ranking rules using support, confidence, and lift

## Project Process

The project followed these main steps:

1. Load and prepare the datasets
2. Split the heart disease dataset into training and testing sets
3. Apply Decision Tree and KNN classification models
4. Evaluate and compare model performance
5. Load online retail transaction data
6. Clean and filter transaction records
7. Apply association rules mining
8. Analyze results using support, confidence, and lift
9. Present the outputs using tables, charts, and evaluation metrics

## Dataset Note

The full Online Retail dataset was reduced to a sample of 1200 rows for GitHub upload due to file size limitations.

The sample dataset is used for academic and portfolio purposes, while the project code and Jupyter Notebook output demonstrate the full analysis process.

## MIS Relevance

This project reflects the role of Management Information Systems in using data mining and machine learning techniques to support decision-making.

The heart disease classification part demonstrates how data can be used to build predictive models, while the online retail analysis shows how transaction data can help discover customer purchasing patterns. These insights can support business decisions, product recommendations, and data-driven strategies.

## Author

Abdallah Shaar
