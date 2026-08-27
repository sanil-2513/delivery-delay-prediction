# Delivery Delay Prediction

## Topic

**Machine Learning Based Delivery Delay Prediction**

## Dataset

* **Total Rows:** 25,000
* **Total Columns:** 15
* **Target Variable:** `delayed`

### Columns

1. `delivery_id`
2. `delivery_partner`
3. `package_type`
4. `vehicle_type`
5. `delivery_mode`
6. `region`
7. `weather_condition`
8. `distance_km`
9. `package_weight_kg`
10. `delivery_time_hours`
11. `expected_time_hours`
12. `delayed`
13. `delivery_status`
14. `delivery_rating`
15. `delivery_cost`

### Target Distribution

* `no` → 18,331 records (73.324%)
* `yes` → 6,669 records (26.676%)

## Model Used

**Random Forest Classifier**

### Preprocessing

* Categorical features → One-Hot Encoding
* Numerical features → passed to the model
* Train-test split → 80/20 with stratification
* Target leakage features removed:

  * `delivery_time_hours`
  * `delivery_status`

## Model Accuracy

**97.5%**

### Model Output

The model predicts:

* **ON TIME**
* **DELAYED**

It also provides:

* Delay Probability
* Prediction Confidence
* Risk Level

### Risk Levels

* Below 40% → **LOW RISK**
* 40%–69.99% → **MEDIUM RISK**
* 70% and above → **HIGH RISK**

## Changes Added

* Updated `train.py` for model training and preprocessing.
* Added train-test splitting and model evaluation.
* Added target leakage handling.
* Added model saving using Joblib.
* Added `models/` folder.
* Added trained model file:
  `models/delivery_delay_model.pkl`
* Updated `test.py` to load the saved model.
* Added interactive prediction menu.
* Added **Normal Delivery** option.
* Added **Risky Delivery** option using a model-verified dataset record.
* Added **Custom Delivery** option for user-provided inputs.
* Added delay probability and prediction confidence.
* Added Low/Medium/High risk classification.
* Added input validation for custom delivery values.

## Prediction System

Run:

```bash
python src/test.py
```

Options:

```text
1. Normal Delivery
2. Risky Delivery
3. Custom Delivery
4. Exit
```

### Example Results

**Normal Delivery**

```text
Status               : ON TIME
Delay Probability    : 14.00%
Prediction Confidence: 86.00%
Risk Level           : LOW RISK
```

**Risky Delivery**

```text
Status               : DELAYED
Delay Probability    : 96.00%
Prediction Confidence: 96.00%
Risk Level           : HIGH RISK
```

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

## Technologies

* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn
