import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ===================================================
# TASK 1 : DATASET UNDERSTANDING
# ===================================================

print("=" * 60)
print("DATASET UNDERSTANDING")
print("=" * 60)

df = pd.read_csv("adult.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ===================================================
# TASK 2 : DATA CLEANING
# ===================================================

print("\n")
print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# replace ? with NaN

df.replace(" ?", np.nan, inplace=True)
df.replace("?", np.nan, inplace=True)

# remove missing values

df.dropna(inplace=True)

# remove duplicates

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)

# ===================================================
# TARGET COLUMN IDENTIFICATION
# ===================================================

target_col = df.columns[-1]

print("\nTarget Column:", target_col)

# ===================================================
# TASK 3 : FEATURE ENGINEERING
# ===================================================

print("\n")
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Label Encode Target

le = LabelEncoder()

df[target_col] = le.fit_transform(df[target_col])

# One Hot Encoding

df = pd.get_dummies(df, drop_first=True)

# Split Features and Target

X = df.drop(target_col, axis=1)
y = df[target_col]

# Feature Scaling

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("Feature Engineering Completed")

# ===================================================
# TRAIN TEST SPLIT
# ===================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ===================================================
# MODEL TRAINING
# ===================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": SVC(
        probability=True
    )
}

results = []

print("\n")
print("=" * 60)
print("MODEL BUILDING")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc_auc = roc_auc_score(y_test, predictions)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

    print(f"{name} Completed")

# ===================================================
# TASK 5 : PERFORMANCE EVALUATION
# ===================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(results_df)

# ===================================================
# BEST MODEL
# ===================================================

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest Model Based on Accuracy:")
print(best_model)

# ===================================================
# VISUALIZATION
# ===================================================

plt.figure(figsize=(10,5))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.title("Accuracy Comparison of Models")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()

# ===================================================
# SAVE RESULTS
# ===================================================

results_df.to_csv(
    "model_results.csv",
    index=False
)

print("\nResults saved to model_results.csv")