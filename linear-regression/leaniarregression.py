import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)



#=================Loading the dataset=================#

df = pd.read_csv('regression/Student_Performance.csv')
print("Fist 5 rows of the dataset:")
print(df.head())


print("\n Dataset shape:")
print(df.shape)

print("\n Dataset columns: ")
print(df.columns)


print ("Dataset information:")
print(df.info())

print ("\n Missing values in the dataset:")
print(df.isnull().sum())


#====================== 1. Basic Preprocessing =====================#
# Converting categorical variables into numerical variables using mapping


df["Extracurricular Activities"] = (
    df["Extracurricular Activities"]
      .map({"Yes": 1,
             "No": 0
            })
)

print("\n After Encoding")
print (df.head())


#============= 2. DEFINING FEATURES AND TARGET VARIABLE =============#

x = df[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]

]

y =  df["Performance Index"]



#========================== 3. Train-Test Split ==========================#

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size=0.2,
    random_state=42

)

print("\n Training sample:", len(X_train))
print(" Testing sample:", len(X_test))



#========================= Model Training =========================#

# 4. Create Model
model = LinearRegression()


# 5. Train the model

model.fit(X_train, y_train)


# 6. Model Parameters

print("\n Intercept:")
print(model.intercept_)

print("\n Coefficients:")

for feature, coefficient in zip(x.columns, model.coef_):
    print(feature, ":", coefficient)


#============= 7. Make Predictions =============#

y_pred = model.predict(X_test)
print ("\n Actual vs Predicted:")
comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred})

print(comparison_df.head(10))


#========================= 8. Model Evaluation =========================#

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)


#========================= 9. Visualization =========================#
# Actual  vs Predicted graph

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted Performance Index")
plt.savefig("plot.png")


# Predict a new student

new_student = pd.DataFrame ({
    "Hours Studied": [12],
    "Previous Scores": [65],
    "Extracurricular Activities": [0],
    "Sleep Hours": [7],
    "Sample Question Papers Practiced": [8]
})

prediction = model.predict(new_student)
print("\n Predicted Performance Index for the new student:", prediction[0])
