import numpy as np
import matplotlib.pyplot as plt
import copy 
import pickle

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)


# =============define neural network architecture

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.network(x)
    

# ===========main code to run once

if __name__ == "__main__":

    # ========== Load data========

    digits = load_digits()
    x = digits.data
    y = digits.target

    print("Dataset shape:")
    print(x.shape)

    print("\nNumber of images:")
    print(len(x))

    print("\nImage shape:")
    print(digits.images[0].shape)


print("\n Dataset shape:")
print(x.shape)

print("\n Number of images:")
print(len(x))

print("\n Image shape:")
print(digits.images[0].shape)

print("\n Number of classes:")
print(len(np.unique(y)))

print("\n Classes:")
print(np.unique(y))

print("\n Dictionary of key:")
print(digits.keys())

print("\n Feature names:")
print(digits.feature_names)

print("\n Target names:")
print(digits.target_names)

