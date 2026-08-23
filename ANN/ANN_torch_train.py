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

    # ========== 1. Load data========

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

    # ========== 2. Visualize a sample image =========
    plt.figure(figsize=(4, 4))

    plt.imshow(digits.images[0], cmap='gray')
    plt.title(
        "Digit: " + str(y[0])
    )

    plt.axis('off')
    plt.savefig("digit_image.png")


    # ========== 3. Train / Test  Split/ Validation =========

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
        )

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
        )

    print("\n Training samples:", len(x_train))
    print("\n Validation samples:", len(x_validation))
    print("\n Test samples:", len(x_test))

    # ========== 4. Feature Scaling =========


    Scaler = StandardScaler()
    X_train = Scaler.fit_transform(x_train)
    X_validation = Scaler.transform(x_validation)
    X_test = Scaler.transform(x_test)


    # ========== 5. Convert to PyTorch tensors =========

    X_train = torch.tensor(X_train, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)

    X_validation = torch.tensor(X_validation, dtype=torch.float32)

    y_validation = torch.tensor(y_validation, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_test = torch.tensor(y_test, dtype=torch.long)


    print("\n Tensors shape:", X_train.shape)
    print("Validation labels shape:", y_validation.shape)
    print("Test labels shape:", y_test.shape)


    # ========== 6. Create DataLoader =========

    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\n Training batches:")
    print("Batch size:", batch_size)
    print("Number of batches per epoch:", len(train_loader))

    # 7. Create Neural Network model

    model = NeuralNetwork()

    print("\n Neural Network architecture:")
    print(model)


    # ========== 8. Define loss function  =========

    loss_function = nn.CrossEntropyLoss()


    # ========== 9. Define optimizer =========

    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # ========== 10. Training loop =========

    epochs = 100
    loss_history = []
    validation_loss_history = []


    # ---------  Early stopping setting ---------

    patience = 10
    min_delta = 0.02
    counter = 0
    best_validation_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0


    print("\n Training the model...")
    print("Maximum epochs:", epochs)
    print("Batch size:", batch_size)
    print("Early stopping patience:", patience)


    for epoch in range (epochs):

        # Training

        model.train()
        total_training_losses = 0.0

        for X_batch, y_batch in train_loader:
                #--------Forward pass ---------
            outputs = model(X_batch)
                #--------Compute loss ---------
            loss = loss_function(outputs, y_batch)

            # === Clear Previous Gradients ===
            optimizer.zero_grad()

            # ====== Backpropagation ======
            loss.backward()

            #--------Update weights ---------
            optimizer.step()
    
            total_training_losses += (loss.item())

        # Average training loss for the batches
        training_loss = total_training_losses / len(train_loader)
        loss_history.append(training_loss)



        # Validation

        model.eval()
        with torch.no_grad():
             validation_outputs = model(X_validation)
             validation_loss = loss_function(validation_outputs, y_validation)

        validation_loss = validation_loss.item()
        validation_loss_history.append(validation_loss)



        # Early stopping check
        if validation_loss < best_validation_loss - min_delta:
             best_validation_loss = validation_loss
             best_epoch = epoch + 1
             epochs_without_improvement = 0


             # Save the best model
             best_model_state = copy.deepcopy(model.state_dict())

        else:
            epochs_without_improvement += 1


        # Displaay training progress

        if (epoch + 1) % 5 == 0 or epoch == 0:
             print(
                  "Epoch:", 
                   epoch + 1,
                   "/",
                   epochs,
                   "Training Loss:",
                   round(training_loss, 4),
                     "Validation Loss:",
                   round(validation_loss, 4)

             )


        # Check for early stopping

        if epochs_without_improvement >= patience:
            print(
                "Early stopping triggered:",
                epoch + 1,
                "Best epoch:",
                best_epoch,
                "Best validation loss:",
                round(best_validation_loss, 4)
            )
            break



    # 11. RESTORE the best model 

    model.load_state_dict(best_model_state)
    print("\n Best model restored from epoch:")

    # 12. save model

    torch.save(model.state_dict(),"model.pth")

    print("\nModel saved as: model.pkl")

    # =========13. save sclaer

    with open("scaler.pkl","wb") as file:
        pickle.dump(Scaler,file)
    print("Scaler saved as: sclaer.pkl")

    # ==14. make predictions

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs,dim=1)

    # conver to numpy
        
    y_pred = predictions.numpy()
    y_actual = y_test.numpy()

    print("\nfirst 20 predictions:")
    print(y_pred[:20])
    print("\nFirst 20 actual values:")
    print(y_actual[:20])


    # ==prediction probabilites

    with torch.no_grad():
        outputs = model(X_test)
        probabilites = torch.softmax(outputs, dim=1)

    print("\n probabilites for first image:")
    print(probabilites[0].numpy())

    # ====16. evaluation

    accuracy = accuracy_score(y_actual,y_pred)
    precision = precision_score(y_actual,y_pred,average="weighted")

    recall = recall_score(y_actual,y_pred,average="weighted")

    f1 = f1_score(y_actual,y_pred,average="weighted")

    cm = confusion_matrix(y_actual,y_pred)

    print("\n Model Evaluation Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)




    # ======= 17. Classification Report

    print("\nClassification Report:")

    print(
        classification_report(
            y_actual,
            y_pred
        )
    )


    # ======= 18. Confusion Matrix ============

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_actual,
            y_pred
        )
    )

    print("\nConfusion Matrix:")
    




    # ======= 19. Visualize Confusion Matrix ========

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)

    plt.title("ANN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(range(10), range(10))
    plt.yticks(range(10), range(10))

    for i in range(10):
        for j in range(10):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",  
            )

    plt.colorbar()
    plt.savefig("ann_confusion_matrix.png")



   # ==================== 20. VISUALIZE TRAINING AND VALIDATION LOSS =====================

    plt.figure(figsize=(8,5))
    plt.plot(loss_history,label="Training Loss")
    plt.plot(validation_loss_history,label="Validation Loss")

    plt.axvline(
        x=best_epoch,
        linestyle="--",
        label="Select Model",
    )

    plt.scatter(
        best_epoch,
        best_validation_loss,
        s=60,
        zorder=5,
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.grid()
    plt.savefig("training_validation_loss.png")

    









    






