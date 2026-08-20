import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)


# ============================================
# LOGISTIC REGRESSION
# Titanic Dataset
# ============================================

# ============================================
# 1. LOAD DATA
# ============================================


df = pd.read_csv("logistic-regression/train.csv")


print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# ============================================
# 2. SELECT USEFUL FEATURES
# ============================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]

y = df["Survived"]

# ============================================
# 3. HANDLE MISSING VALUES
# ============================================

# Age
X["Age"] = X["Age"].fillna(X["Age"].median())

# Embarked
X["Embarked"] = X["Embarked"].fillna(
    X["Embarked"].mode()[0]
)

# ============================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================

X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked"],
    drop_first=True
)

print("\nAfter encoding:")
print(X.head())

print("\nColumns after encoding:")
print(X.columns)

# ============================================
# 5. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================
# 6. FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================
# 7. CREATE LOGISTIC REGRESSION MODEL
# ============================================

model = LogisticRegression(
    max_iter=1000
)

# ============================================
# 8. TRAIN MODEL
# ============================================

model.fit(X_train, y_train)

# ============================================
# 9. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

print("\nFirst 10 predictions:")

print(y_pred[:10])

# ============================================
# 10. PREDICT PROBABILITIES
# ============================================

y_probability = model.predict_proba(X_test)

print("\nFirst 10 probabilities:")

print(y_probability[:10])

# ============================================
# 11. EVALUATION
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================
# 12. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================
# 13. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)

# ============================================
# 14. VISUALIZE CONFUSION MATRIX
# ============================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.yticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.show()

# ============================================
# 15. FEATURE COEFFICIENTS
# ============================================

print("\nModel Coefficients:")

for feature, coefficient in zip(
    X.columns,
    model.coef_[0]
):

    print(
        feature,
        ":",
        coefficient
    )

# ============================================
# 16. PREDICT A NEW PASSENGER
# ============================================

new_passenger = pd.DataFrame({
    "Pclass": [3],
    "Sex": ["male"],
    "Age": [25],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [10],
    "Embarked": ["S"]
})

# Encode the same way
new_passenger = pd.get_dummies(
    new_passenger,
    columns=["Sex", "Embarked"],
    drop_first=True
)

# Make sure columns match training data
new_passenger = new_passenger.reindex(
    columns=X.columns,
    fill_value=0
)

# Scale using the EXISTING scaler
new_passenger_scaled = scaler.transform(
    new_passenger
)

# Prediction
prediction = model.predict(
    new_passenger_scaled
)

# Probability
probability = model.predict_proba(
    new_passenger_scaled
)

print("\n========== NEW PASSENGER ==========")

if prediction[0] == 1:
    print("Prediction: Survived")
else:
    print("Prediction: Did NOT survive")

print(
    "Probability of survival:",
    probability[0][1]
)