# Bank Customer Churn Prediction

Welcome to the Bank Customer Churn Prediction project! This repository explores the use of Python-based data science and machine learning techniques to predict customer attrition in the banking industry. The goal is to build a robust machine-learning model and leverage advanced visualization techniques to enhance customer retention strategies.

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Data](#data)
3. [Features](#features)
4. [Visualization](#visualization)
5. [Data Preprocessing](#data-preprocessing)
6. [Machine Learning Models](#machine-learning-models)
    1. [Model Training and Evaluation](#model-training-and-evaluation)
    2. [Hyperparameter Tuning](#hyperparameter-tuning)
    3. [Model Comparison](#model-comparison)
7. [Results and Conclusion](#results-and-conclusion)
8. [Saving the Model](#saving-the-model)
9. [Loading the Model](#loading-the-model)

## Problem Definition

The primary objective of this project is to predict customer churn in the banking industry. By analyzing various factors present in the dataset, we aim to accurately estimate the likelihood of customers leaving the bank. This project intends to uncover significant factors influencing customer churn decisions using advanced data science techniques.

## Data

The dataset for this project is obtained from Kaggle: [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset). It contains features such as credit score, country, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary, and the churn label.

## Features

The dataset includes the following features:

- credit_score
- country
- gender
- age
- tenure
- balance
- products_number
- credit_card
- active_member
- estimated_salary
- churn

## Visualization

Exploratory data analysis is a crucial step to understand the data and identifying patterns. Various visualizations are created to reveal relationships between different features and the target variable (churn). Some of the visualizations include histograms, count plots, heat maps, and more.

## Data Preprocessing

The data preprocessing phase involves handling missing values, encoding categorical features, and splitting the dataset into training and testing sets. Label encoding is applied to categorical columns like country and gender to convert them into numerical format.

## Machine Learning Models

Several machine learning models are utilized to predict customer churn, including:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### Model Training and Evaluation

The models are trained on the training dataset and evaluated using the test dataset. Accuracy is used as the evaluation metric. The accuracy scores for each model are compared, and their computational efficiency is considered.

### Hyperparameter Tuning

Hyperparameter tuning is performed to optimize model performance. RandomizedSearchCV is used to search for the best combination of hyperparameters for each model. The resulting accuracy scores and best parameters are reported.

### Model Comparison

A bar plot is generated to visually compare the accuracy of different models. This helps in selecting the most accurate model.

## Results and Conclusion

After thorough analysis and hyperparameter tuning, the results for each model are presented:

1. Random Forest:
   - Best Accuracy: 0.8615
   - Best Parameters: {'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 20}
   - Time Taken: 57.6084 seconds

2. Gradient Boosting:
   - Best Accuracy: 0.8650
   - Best Parameters: {'learning_rate': 0.01, 'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 19, 'min_samples_split': 12, 'n_estimators': 558}
   - Time Taken for Randomized Search: 2117.9838 seconds

3. XG Boost:
   - Best Accuracy: 0.8625
   - Best Parameters: {'subsample': 1.0, 'reg_lambda': 1, 'reg_alpha': 0.001, 'n_estimators': 400, 'min_child_weight': 8, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 0.3, 'colsample_bytree': 0.7}
   - Time Taken for Tuning: 821.8873 seconds

4. Light GBM:
   - Best Accuracy: 0.8631
   - Best Parameters: {"subsample": 0.8, "num_leaves": 31 "min_child_samples": 30, "max_depth": 15, "learning_rate": 0.05, "colsample_bytree": 1.0}
   - Time Taken for Tuning: 9.4179 seconds

The project concludes that Gradient Boosting exhibits the highest accuracy, demonstrating its potential for accurate customer churn prediction. The insights gained from this project can aid the banking industry in understanding customer behavior and making informed decisions.

## Saving the Model

The best-performing model (LightGBM) is saved using the `pickle` library for future use.

## Loading the Model

The saved model can be loaded and used to predict churn for new data points.

Feel free to explore the code and experiment with your own data to gain insights into customer churn prediction in the banking industry.
