"""
ML Assignment 2 - Classification Models Training Script
Student: Ayush Sharma | BITS ID: 2025AD05035
Dataset: Breast Cancer Wisconsin (Diagnostic) - sklearn built-in
         569 samples, 30 features, binary classification
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")


# Load Dataset

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print("=" * 60)
print("DATASET: Breast Cancer Wisconsin (Diagnostic)")
print("=" * 60)
print(f"Samples   : {X.shape[0]}")
print(f"Features  : {X.shape[1]}")
print(f"Classes   : {list(data.target_names)} (0=malignant, 1=benign)")
print(f"Class dist: {dict(pd.Series(y).value_counts())}")

# Train / Test Split (80/20) and Preprocessing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")

# Save test data for Streamlit demo
test_df = pd.DataFrame(X_test, columns=data.feature_names)
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("test_data.csv saved.")


# Define Models

models_config = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
 
}


# Our next step!
# Train, Evaluate, Save

os.makedirs("model", exist_ok=True)
results = {}

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION")
print("=" * 60)

for name, clf in models_config.items():
    print(f"\n[{name}]")

    # Models that need scaled data
    needs_scaling = name in ["Logistic Regression", "kNN", "Naive Bayes Gaussian"]
    Xtr = X_train_sc if needs_scaling else X_train.values
    Xte = X_test_sc if needs_scaling else X_test.values


    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    y_prob = (
        clf.predict_proba(Xte)[:, 1]
        if hasattr(clf, "predict_proba")
        else y_pred.astype(float)
    )

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "needs_scaling": needs_scaling,
        "y_test": y_test.values.tolist(),
        "y_pred": y_pred.tolist(),
    }

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC Score : {auc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  MCC Score : {mcc:.4f}")
