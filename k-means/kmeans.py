import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



#=================Loading the dataset=================#

df = pd.read_csv('k-means/Mall_Customers.csv')
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


# ===================== 2. select features =====================#

X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"]]

print("\n Selected features:")
print(X.head())

# ========================== 3. Feature Scaling ==========================#
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. Create a KMeans model 

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)


