import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree

) 
from sklearn.metrics import (
    accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)


# ========================== 1. Load Dataset ==========================#

data = load_breast_cancer(
    as_frame=True
)

X = data.data
y = data.target

print("Features:")
print(X.head())


print("\n Dataset shape:")
print(y.head())

print ("shape:")
print(X.shape)

print ("\n classes:")
print(data.target_names)



#========================== 2.  Train-Test Split ==========================#

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
    

)

print("\n Training sample:", len(X_train))
print(" Testing sample:", len(X_test))


#============= 3. Creating Decision Tree Classifier==================#


model = DecisionTreeClassifier(
    max_depth=4,
    criterion="gini",
    random_state= 42
)


# =========== 4.  Train =============


model.fit(X_train, y_train) 



#============= 5.  Predict =============

y_pred = model.predict(X_test)


# ========= 6. Evaluate =============

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n Model Performance:")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


#========= 7. Classification Report =============

print("\n Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))


# ======== 8. Confusion Matrix =============

cm = confusion_matrix(y_test, y_pred)   
print("\n Confusion Matrix:")
print(cm)


#========== 9.  feature Importance =============


importances = pd.DataFrame({
    "Feature": X.columns,   
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n Feature Importance:")
print(importances.head(10))



#========= 10. Visualize the Decision Tree =============

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree classifications")
plt.savefig("tree_classify.png")


# ============== 11.  predict for a new patient =============``


new_patient = pd.DataFrame({
    "mean radius": [14.0],
    "mean texture": [20.0],
    "mean perimeter": [90.0],
    "mean area": [600.0],
    "mean smoothness": [0.10],
    "mean compactness": [0.08],
    "mean concavity": [0.05],
    "mean concave points": [0.03],
    "mean symmetry": [0.18],
    "mean fractal dimension": [0.06],

    "radius error": [0.5],
    "texture error": [1.0],
    "perimeter error": [3.5],
    "area error": [40.0],
    "smoothness error": [0.007],
    "compactness error": [0.02],
    "concavity error": [0.02],
    "concave points error": [0.01],
    "symmetry error": [0.02],
    "fractal dimension error": [0.003],

    "worst radius": [16.0],
    "worst texture": [25.0],
    "worst perimeter": [105.0],
    "worst area": [800.0],
    "worst smoothness": [0.13],
    "worst compactness": [0.15],
    "worst concavity": [0.15],
    "worst concave points": [0.08],
    "worst symmetry": [0.25],
    "worst fractal dimension": [0.08]
})


# make prediction 

prediction = model.predict(new_patient)

print("\n Prediction for the new patient:")

print("Predicted class:", prediction[0])

print("prediction: ", data.target_names[prediction[0]])



#============= 12 . Predict probability =============

probabilities = model.predict_proba(new_patient)
print("\n Prediction Probabilities :")

print(
    "Malignant probability:",
    round(probabilities[0][0] * 100, 2), "%"

)

print(
    "Benign probability:",
    round(probabilities[0][1] * 100, 2), "%"
)