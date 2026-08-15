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
