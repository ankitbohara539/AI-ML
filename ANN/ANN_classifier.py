import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



# ================= 1. LOAD DATASET ===============

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y =pd.Series(data.target)

print("First 5 rows of the dataset:")
print(X.head())

print("\n Dataset shape:")
print(X.shape)

print("\n Number of features: ")
print(X.shape[1])

print("\n Target Names:")
print(data.target_names)

print("\n Class Distribution:")
print(y.value_counts())



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



#============= 3. Feature Scaling ==========================#

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n Feature scaled successfully:")



#============= 4. Creating Neural Network ==========================#

model = MLPClassifier(
    hidden_layer_sizes=(32, 16, 8),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)


# =========== 5.  Display Network Architecture =============

print("\n Neural Network Architecture:")
print("\n Input Layer ")
print("Number of neurons: ", X_train.shape[1])
print("Represents the 30 input features of the dataset ")

print("\n Hidden Layer 1 ")
print("Number of neurons: 32")
print("Activation function: ReLU ")

print("\n Hidden Layer 2 ")
print("Number of neurons: 16")
print("Activation function: ReLU ")

print("\n Hidden Layer 3 ")
print("Number of neurons: 8")
print("Activation function: ReLU ")

print("\n Output Layer ")
print("Number of neurons: 1")
print("Activation function: Sigmoid") 

print("\n Network Structure:")
print(
    X_train.shape[1], "-> 32 -> 16 -> 8 -> 1"
)

print("\n Optimizer: Adam")
print("Maximum Iterations: 1000")


# =========== 6.  Train =============

print("\n Training the Neural Network...")
model.fit(X_train, y_train)
print ("\n Training completed successfully.")


#============= 7.  Training Losses After Each Iteration ==============

print("\n Training Losses After Each Iteration:")

for epoch, loss in enumerate(
    model.loss_curve_,
    start=1
):
    print("Iteration", epoch, "-> Loss:", loss)



# ============ 8. Make  Predictions =============

y_pred = model.predict(X_test)
print("\n First 10 Predictions:")
print(y_pred[:10])



# ============ 9. Predict Probabilities =============

y_probabilities = model.predict_proba(X_test)
print("\n First 10 Predicted Probabilities:")
print(y_probabilities[:10])



#========== = 10. Evaluate the Model =============

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)



# ============================================
# 11. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================
# 12. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)

# ============================================
# 13. VISUALIZE CONFUSION MATRIX
# ============================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    data.target_names
    
)

plt.yticks(
    [0, 1],
    data.target_names
    
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


# ============= 14. Loss Curve Visualization =============

plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("Loss Curve")

plt.title("Neural Network Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.grid()
plt.show()



# ================ 15.  predict for a new patient =============

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

new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
probabilities = model.predict_proba(new_patient_scaled)
print("\nNew Patient")
print("Prediction: ", data.target_names[prediction[0]])
print("Probabilities: ", probabilities[0])

