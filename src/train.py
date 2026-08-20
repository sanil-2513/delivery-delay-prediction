import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("data/delivery_data.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# -----------------------------
# 2. Data Cleaning
# -----------------------------

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print("\nShape after cleaning:", df.shape)


# -----------------------------
# 3. Target Distribution
# -----------------------------

print("\nTarget Distribution:")
print(df["delayed"].value_counts())

print("\nTarget Percentage:")
print(df["delayed"].value_counts(normalize=True) * 100)


# -----------------------------
# 4. Basic Statistics
# -----------------------------

print("\nNumerical Statistics:")
print(df.describe())


# -----------------------------
# 5. Visualization
# -----------------------------

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="delayed")
plt.title("Delivery Delay Distribution")
plt.xlabel("Delayed")
plt.ylabel("Number of Deliveries")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="weather_condition", hue="delayed")
plt.title("Delivery Delays by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Deliveries")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="delayed", y="distance_km")
plt.title("Delivery Distance vs Delay")
plt.xlabel("Delayed")
plt.ylabel("Distance (km)")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Machine Learning
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Features and target
X = df.drop(columns=["delayed", "delivery_time_hours", "delivery_status"])
y = df["delayed"]

# Categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object", "str"]).columns
numerical_columns = X.select_dtypes(exclude=["object", "str"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(pipeline, "models/delivery_delay_model.pkl")

print("\nModel saved successfully!")