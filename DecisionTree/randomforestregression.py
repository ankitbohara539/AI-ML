import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
) 
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)



df = pd.read_csv('dtr/house_price_regression_dataset.csv')
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




# ============ Feature and Target ============ #


X = df[
    [
        "Square_Footage",
        "Num_Bedrooms",
        "Num_Bathrooms",
        "Year_Built",
        "Lot_Size",
        "Garage_Size",
        "Neighborhood_Quality",
    ]
]

Y = df["House_Price"]



#========================== Train-Test Split ==========================#

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42

)

print("\n Training sample:", len(X_train))
print(" Testing sample:", len(X_test))




# ========== Create Random Forest =========

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)


# 5. Train the model

model.fit(X_train, y_train)


## 6. Make predictions on the test set

y_pred = model.predict(X_test)


#===================7. Model Evaluation =========================#


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Random Forest Results:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


# 8.  Actual vs Predicted

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\n Actual vs Predicted:")
print(comparison_df.head(10))   


# 9. Feature Importance

importances = pd.DataFrame({
    "Feature": X.columns,   
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n Feature Importance:")
print(importances)


# 10. Visualize the Decision Tree



# 11. Prediction house price for a new house

new_house = pd.DataFrame({
    "Square_Footage": [2000],
    "Num_Bedrooms": [3],
    "Num_Bathrooms": [3],
    "Year_Built": [2010],
    "Lot_Size": [5000],
    "Garage_Size": [2],
    "Neighborhood_Quality": [8]
})

predicted_price = model.predict(new_house)

print("\n Predicted House Price for the new house:")
print(predicted_price[0])