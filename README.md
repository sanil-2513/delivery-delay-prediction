# Delivery Delay Prediction

## Project Overview

This project is an MLOps-based machine learning system for predicting whether a delivery will be delayed.

The project uses a delivery dataset containing 25,000 records and applies data cleaning, exploratory data analysis, visualization, machine learning, model evaluation, model saving, and prediction testing.

## Problem Statement

Delivery delays can affect customer satisfaction and logistics operations. The goal of this project is to build a machine learning model that predicts whether a delivery is likely to be delayed using information available before the delivery is completed.

## Dataset

- Number of records: 25,000
- Number of features: 15
- Target variable: `delayed`
- Target classes:
  - `no`: 73.324%
  - `yes`: 26.676%

### Main Features

- Delivery partner
- Package type
- Vehicle type
- Delivery mode
- Region
- Weather condition
- Distance
- Package weight
- Expected delivery time
- Delivery rating
- Delivery cost

## Data Cleaning

The dataset was processed using:

- Duplicate removal
- Missing value removal
- Data type correction
- Basic statistical analysis

The delivery time columns were corrected from incorrectly formatted timestamp values back to numerical hours.

## Exploratory Data Analysis

Three visualizations were created:

1. Delivery Delay Distribution
2. Delivery Delays by Weather Condition
3. Delivery Distance vs Delay

These visualizations help identify patterns related to delivery delays.

## Machine Learning

A Random Forest Classifier was used for prediction.

### Preprocessing

- Categorical features were encoded using One-Hot Encoding.
- Numerical features were passed directly to the model.
- An 80/20 stratified train-test split was used.

### Target Leakage Handling

Initially, the model achieved 100% accuracy because information that would only be known after delivery was available as input.

To prevent target leakage, the following columns were removed:

- `delivery_time_hours`
- `delivery_status`

The final model therefore uses information available before/during delivery prediction.

## Model Performance

Final model accuracy:

**97.5%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| no | 0.99 | 0.98 | 0.98 |
| yes | 0.94 | 0.97 | 0.95 |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 3587 | 79 |
| Actual Yes | 46 | 1288 |

## Model Saving

The trained model is saved as:

```text
models/delivery_delay_model.pkl