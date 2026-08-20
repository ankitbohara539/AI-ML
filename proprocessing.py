import pandas as pd
from date_normalization import parse_date
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, chi2_contingency

print("loading dataset")
df = pd.read_csv("customer_raw.csv")

print(df.head(4))

df.info()

duplicates = df.duplicated()

if duplicates.any():
    print(f"duplicate rows: {duplicates.sum()}")

else:
    print("no duplicate rows found")

df = df.drop_duplicates()

print("\n\nRemoving duplicate rows\n")

duplicates = df.duplicated()

if duplicates.any():
    print(f"duplicate rows: {duplicates.sum()}")

else:
    print("No duplicate rows found.")

print("duplicate rows removed\n")

    #  check null entries

print("\nNumber of null entries\n")
print(df.isnull().sum())

null_counts = df.isnull().sum()
print(null_counts[null_counts >0 ])
rows_with_null = df.isnull().any(axis=1).sum()
print(f"\nRows with at least one missing value: { rows_with_null}/{len(df)} \n\n")

#    create a new dataset without null entries

print("creating new datasets by removing null entries")
df_no_null = df.dropna()
df_no_null.to_csv("customer_raw_no_null.csv")
print("saved to customer_raw_no_null.csv\n")


# split dataframe into categorical and numerical dataframes

numeric_cols = [
    "Age",
    "Annual_Income",
    "Purchase_Amount"
]

identifier_cols = [
    "Customer_ID",
    "Customer_Name",
    "Phone",
    "Email"
]

categorical_cols = [
    "Gender",
    "City",
    "Payment_Method",
    "Customer_Status",
    "Customer_Segment"
]

date_cols = [
    "Registration_Date"
]

numeric_df = df[numeric_cols].copy()
categorical_df = df[categorical_cols].copy()
identifier_df = df[identifier_cols].copy()
date_df = df[date_cols].copy()


    #   parse date columns

date_df["Registration_Date"] = date_df["Registration_Date"].apply(parse_date)
date_df.to_csv("date.csv")


# -----------format numeric data

numeric_df["Age"] = pd.to_numeric(numeric_df["Age"], errors="coerce").astype("Int64")

numeric_df["Annual_Income"] = (
    numeric_df["Annual_Income"]
    .astype(str)
    .str.replace(",","",regex=False)
    .str.replace("Rs.","",regex=False)
    .str.strip()
)
numeric_df["Annual_Income"] = pd.to_numeric(
    numeric_df["Annual_Income"],
    errors = "coerce"
).astype(float)

numeric_df["Purchase_Amount"] = (
    numeric_df["Purchase_Amount"]
    .astype(str)
    .str.replace(",","",regex=False)
    .str.replace("Rs.","",regex=False)
    .str.strip()
)

numeric_df["Purchase_Amount"] = pd.to_numeric(
    numeric_df["Purchase_Amount"],
    errors = "coerce"
).astype(float)

print(numeric_df[["Age","Annual_Income","Purchase_Amount"]].dtypes)

# --------------handling categorical columns


print("\n\nCleaning categorical entries\n")

print("\n\n")

for col in categorical_df.columns:
    print(f"\n---{col}---")
    print(categorical_df[col].value_counts(dropna=False))



# formatiting gender
    
categorical_df["Gender"] = categorical_df["Gender"].replace({
    "M": "Male",
    "m": "Male",
    "Man": "Male",
    "F": "Female",
    "f": "Female",
    "Woman": "Female"
  })

# --------------------------formatting all categorical column

for col in categorical_df.columns:
    categorical_df[col] = (
        categorical_df[col]
        .str.strip()
        .str.title()
    )

print("\n\\n")
for col in categorical_df.columns:
    print(f"\n----{col}----")
    print(categorical_df[col].value_counts(dropna=False))


    # format name and phone entries

print("\n\nFormatting name and phone entries\n")

identifier_df["Customer_Name"] = (
    identifier_df["Customer_Name"]
    .str.strip()
    .str.title()
)

identifier_df["Phone"] = (
    identifier_df["Phone"]
    .astype(str)
    .str.replace(r"\+977", "", regex=True)
    .str.replace(r"/D", "", regex=True)
)



# ------------------------check outliers

print("\n\nCHECKING boxplots of numeric columns to detect outliers\n")

fig, axs = plt.subplots(len(numeric_df.columns), 1, figsize=(7,3*len(numeric_df.columns)), dpi=95)
axs = np.atleast_1d(axs)
for i, col in enumerate(numeric_df.columns):
    axs[i].boxplot(numeric_df[col].dropna(),vert=False)
    axs[i].set_xlabel(col)

plt.tight_layout()
# plt.show()
plt.savefig("boxplot.png")


# remove outliers---------------------

print("\n\n Removing outliers using iqr method\n")

outlier_indices = set()
for col in numeric_cols:
    Q1 = numeric_df[col].quantile(0.25)
    Q3 = numeric_df[col].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5* IQR
    upper_bound = Q3  + 1.5 * IQR

    outliers = numeric_df[
        (numeric_df[col] < lower_bound)  |
        (numeric_df[col] > upper_bound)
    ]

    print(
        f"{col}: {len(outliers)} outliers"
        f"({lower_bound:.2f} to {upper_bound:.2f})"
    )
    outlier_indices.update(outliers.index)

print(f"\n Total rows to remove:  {len(outlier_indices)}")

# remove the same rows from every dataframe

numeric_df = numeric_df.drop(index=outlier_indices)
categorical_df = categorical_df.drop(index=outlier_indices)
identifier_df = identifier_df.drop(index=outlier_indices)
date_df = date_df.drop(index=outlier_indices)


# ----------------------data validation
print("\n\nData Validation")

invalid_indices = set()

# ---------- Age ----------
invalid_age = numeric_df[
    (numeric_df["Age"] < 0) |
    (numeric_df["Age"] > 120)
]

if len(invalid_age) > 0:
    print(f"Invalid Age: {len(invalid_age)} rows")
    print(invalid_age.index.tolist())
    invalid_indices.update(invalid_age.index)

# ---------- Annual Income ----------
invalid_income = numeric_df[
    numeric_df["Annual_Income"] < 0
]

if len(invalid_income) > 0:
    print(f"Invalid Annual Income: {len(invalid_income)} rows")
    print(invalid_income.index.tolist())
    invalid_indices.update(invalid_income.index)


# ---------- Purchase Amount ----------
invalid_purchase = numeric_df[
    numeric_df["Purchase_Amount"] <= 0
]

if len(invalid_purchase) > 0:
    print(f"Invalid Purchase Amount: {len(invalid_purchase)} rows")
    print(invalid_purchase.index.tolist())
    invalid_indices.update(invalid_purchase.index)



# -----------------checking correlation for feature selection

print("\n\n Checking correlation between features and target to determmine important features\n")

results  = []

# ---------target(encoding for ordinality)
    
target_encoded = categorical_df["Customer_Segment"].map({
    "New": 0,
    "Regular": 1,
    "Premium": 2
})

# ---------Numeric Features-------

for col in numeric_cols:
    valid = (
        numeric_df[col].notna() &
        target_encoded.notna()
        )
    r,p = pearsonr(
        numeric_df.loc[valid,col],
        target_encoded.loc[valid]
    )

    results.append({
        "Feature": col,
        "Type": "Numerical",
        "Test": "Pearson",
        "Score" : r,
        "p-value" : p
    })

    # categorical featuress

    categorical_features = [
        "Gender",
        "City",
        "Payment_Method",
        "Customer_Status"
    ]

    for col in categorical_features:
        table = pd.crosstab(
            categorical_df[col],
            categorical_df["Customer_Segment"]
        )

        chi2, p, dof, expected = chi2_contingency(table)

        results.append({
            "Feature": col,
            "Type": "Categorical",
            "Test": "Chi-square",
            "Score": chi2,
            "p-value": p
        })

# -------summary table----
        
feature_selection = pd.DataFrame(results)
feature_selection["Select"] = np.where(
    feature_selection["p-value"] < 0.05,
    "Yes",
    "No"
)

print(feature_selection)


# -------------------- Normalization -------------------

print("\n\nNormalizing numeric features\n")

scaler = MinMaxScaler()
numeric_df["Purchase_Amount_Scaled"] = scaler.fit_transform(
    numeric_df[["Purchase_Amount"]]
)

print(numeric_df[["Purchase_Amount", "Purchase_Amount_Scaled"]].head(10))

df_clean = pd.concat(
    [
        identifier_df,
        categorical_df,
        numeric_df,
        date_df
    ],
    axis=1
)


# Restore original colum order

df_clean = df_clean[
    [
        "Customer_ID",
        "Customer_Name",
        "Age",
        "Gender",
        "City",
        "Registration_Date",
        "Annual_Income",
        "Purchase_Amount",
        "Payment_Method",
        "Phone",
        "Email",
        "Customer_Status",
        "Customer_Segment"
    ]
]

# Reset_Row_Index

df_clean = df_clean.reset_index(drop = True)
df_clean.to_csv(
    "customer_clean.csv",
    index = False
)


print("\n CLEAN DATASET SAVED")
print(f"Rows: {len(df_clean)}")
print(f"Column: {len(df_clean.columns)}")
print("file: customer_clean.csv")


# ---------Smothing---------------

numeric_df["Income_Bin"] = pd.cut(
    numeric_df["Annual_Income"],
    bins=5
)

numeric_df["Annual_Income_Smoothed"] = (
    numeric_df
    .groupby("Income_Bin", observed=True) ["Annual_Income"]
    .transform("mean")
)

print(
    numeric_df[
        ["Annual_Income", "Income_Bin", "Annual_Income_Smoothed"]
    ].head(10)
)


# -----------Binnig--------

numeric_df["Age_Group"] = pd.cut(
    numeric_df["Age"],
    bins=[0, 18, 30, 45, 60, 120],
    labels=[
        "0-18",
        "19-30",
        "31-45",
        "46-60",
        "60+"
    ]
)



# ============= ORDINAL ENCODING =============

segment_mapping = {
    "New": 0,
    "Regular": 1,
    "Premium": 2
}

categorical_df["Customer_Segment_Encoded"] = (
    categorical_df["Customer_Segment"]
    .map(segment_mapping)
)


#=========== One hot Encoding ====
encoded_payment = pd.get_dummies(
    categorical_df["Payment_Method"],
    prefix="Payment"
)

#=======Combine Cleaned + Derived Data ===========

df_clean_derived = pd.concat(
    [
        identifier_df,
        numeric_df,
        categorical_df,
        date_df,
        encoded_payment
    ],
    axis = 1
)


df_clean_derived = df_clean_derived.reset_index(drop=True)

df_clean_derived.to_csv(
    "Customer_clean_derived.csv",
    index=False
)

print("\n Clean Derived Dataset Saved")
print(f"Rows: {len(df_clean_derived)}")
print(f"Columns: {len(df_clean_derived.columns)}")
print("File: customer_clean_derived.csv")