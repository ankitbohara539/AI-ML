# ============================================
# ARTIFICIAL NEURAL NETWORK
# Test a New Handwritten Digit
# PyTorch
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch

from sklearn.datasets import load_digits

# Import the NeuralNetwork class
from ANN_torch_train import NeuralNetwork

# ============================================
# 1. LOAD DIGITS DATASET
# ============================================

digits = load_digits()

X = digits.data
y = digits.target


# ============================================
# 2. LOAD SCALER
# ============================================

with open(
    "scaler.pkl",
    "rb"
) as file:

    scaler = pickle.load(file)

# ============================================
# 3. LOAD TRAINED MODEL
# ============================================

model = NeuralNetwork()

model.load_state_dict(
    torch.load(
        "model.pth",
        weights_only=True
    )
)

model.eval()

print(
    "\nTrained model loaded successfully."
)

# ============================================
# 4. SELECT A NEW IMAGE
# ============================================

image_index = 50

new_image = X[
    image_index
]

actual_digit = y[
    image_index
]


# ============================================
# 5. PREPROCESS IMAGE
# ============================================

# The model was trained using StandardScaler,
# so the new image must use the same scaler.

new_image = scaler.transform(
    new_image.reshape(1, -1)
)

new_image = torch.tensor(
    new_image,
    dtype=torch.float32
)


# ============================================
# 6. MAKE PREDICTION
# ============================================

with torch.no_grad():

    output = model(
        new_image
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    prediction = torch.argmax(
        output,
        dim=1
    )


predicted_digit = prediction.item()

probabilities = probabilities[
    0
].numpy()


# ============================================
# 7. DISPLAY RESULT
# ============================================

print(
    "\n========== NEW IMAGE =========="
)

print(
    "Actual digit:",
    actual_digit
)

print(
    "Predicted digit:",
    predicted_digit
)


# ============================================
# 8. DISPLAY PREDICTION PROBABILITIES
# ============================================

print(
    "\nPrediction probabilities:"
)

for digit, probability in enumerate(
    probabilities
):

    print(
        digit,
        ":",
        round(
            probability,
            4
        )
    )


# ============================================
# 9. DISPLAY IMAGE
# ============================================

plt.figure(figsize=(4, 4))

plt.imshow(digits.images[image_index],cmap="gray")

plt.title("Predicted: "+ str(predicted_digit))

plt.axis("off")
plt.show()