# Delivery Delay Prediction

## Project Overview

This project is an MLOps-based machine learning system for predicting whether a delivery is likely to be delayed.

The project uses a delivery dataset containing 25,000 records and performs data cleaning, exploratory data analysis, visualization, machine learning, model evaluation, model saving, and prediction testing.

The trained model can be used to predict delivery delays for new delivery information entered by the user.

---

## Problem Statement

Delivery delays can affect customer satisfaction and logistics operations.

The goal of this project is to build a machine learning model that predicts whether a delivery is likely to be delayed using information available before the delivery is completed.

---

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

---

## Data Cleaning

The dataset was processed using:

- Duplicate removal
- Missing value removal
- Data type correction
- Basic statistical analysis

The delivery time columns were corrected from incorrectly formatted timestamp values back to numerical hours.

---

## Exploratory Data Analysis

Three visualizations were created:

1. Delivery Delay Distribution
2. Delivery Delays by Weather Condition
3. Delivery Distance vs Delay

These visualizations help identify patterns related to delivery delays.

---

## Machine Learning

A **Random Forest Classifier** was used for delivery delay prediction.

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

---

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

---

## Model Saving

The trained model is saved as:

`models/delivery_delay_model.pkl`

The model is loaded during prediction using Joblib.

---

## Prediction System

The project includes an interactive prediction system in:

`src/test.py`

The prediction system provides four options:

### 1. Normal Delivery

Provides a realistic sample delivery and demonstrates an **ON TIME** prediction.

### 2. Risky Delivery

Uses a real delivery record from the dataset that is verified by the trained model to produce a **DELAYED** prediction.

The prediction is made by the trained model and is not hard-coded.

### 3. Custom Delivery

Allows the user to enter their own delivery details.

Default sample values are provided, so the user can press **ENTER** to use them or enter different values.

### 4. Exit

Closes the prediction system.

---

## Prediction Output

The system displays:

- Delivery details
- Predicted status
- Delay probability
- Prediction confidence

Example:

**Status:** DELAYED

**Delay Probability:** 96.00%

**Prediction Confidence:** 96.00%

The probability and confidence values are generated from the trained Random Forest model using `predict_proba()`.

---

## Project Structure

```text
delivery-delay-prediction/
│
├── data/
│   └── delivery_data.csv
│
├── models/
│   └── delivery_delay_model.pkl
│
├── src/
│   ├── train.py
│   └── test.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Requirements

The project uses:

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

Install the required packages using:

`pip install -r requirements.txt`

---

## How to Run

### 1. Train the Model

From the project root:

`python src/train.py`

This performs data processing, visualization, model training, evaluation, and saves the trained model.

### 2. Run the Prediction System

`python src/test.py`

Choose:

1. Normal Delivery
2. Risky Delivery
3. Custom Delivery
4. Exit

---

## MLOps Workflow

The project follows a basic MLOps workflow:

```text
Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Preprocessing
↓
Model Training
↓
Model Evaluation
↓
Model Saving
↓
Prediction Testing
```

The project is maintained using Git and GitHub for version control.

---

## Conclusion

The final Random Forest model achieved **97.5% accuracy** on the test dataset.

The system predicts whether a delivery is likely to be delayed and provides the estimated delay probability and prediction confidence.

The interactive prediction interface allows the model to be demonstrated using normal, risky, and custom delivery scenarios.