"""
ML Assignment 2 — Streamlit Web Application
Student: Ayush Sharma | BITS ID: 2025AD05035
Dataset: Breast Cancer Wisconsin (Diagnostic)
Models: Logistic Regression, Decision Tree, kNN, Naive Bayes Gaussian, Naive Bayes Multinomial, Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ML Assignment 2 — Cancer Classification",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Breast Cancer Classification — ML Assignment 2")
st.markdown(
    "**Student:** Ayush Sharma &nbsp;|&nbsp; "
    "**BITS ID:** 2025AD05035 &nbsp;|&nbsp; "
    "**M.Tech AIML — BITS Pilani WILP**"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Load Pre-trained Models
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree':       'model/decision_tree.pkl',
    'kNN':                 'model/knn.pkl',
    'Naive Bayes Gaussian':         'model/naive_bayes_gaussian.pkl',
    'Naive Bayes Multinomial':         'model/naive_bayes_multinomial.pkl',
    'Random Forest':       'model/random_forest.pkl',
}

@st.cache_resource
def load_models():
    loaded = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                loaded[name] = pickle.load(f)
    return loaded

models = load_models()

if not models:
    st.error("No models found. Please run `python ml_models.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Configuration")

# Model selection
selected_models = st.sidebar.multiselect(
    "Select Models to Evaluate",
    options=list(MODEL_FILES.keys()),
    default=list(MODEL_FILES.keys())
)

if not selected_models:
    st.sidebar.warning("Select at least one model.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Info")
st.sidebar.markdown(
    "**Breast Cancer Wisconsin (Diagnostic)**\n\n"
    "- 569 samples, 30 features\n"
    "- Binary: 0 = malignant, 1 = benign\n"
    "- Source: sklearn / UCI\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# Main — CSV Upload
# ─────────────────────────────────────────────────────────────────────────────

st.header("1. Upload Test Data (CSV)")
st.info(
    "Upload a CSV file with the same 30 features as the Breast Cancer dataset. "
    "If a `target` column is present (0=malignant, 1=benign), "
    "ground-truth metrics will be calculated. "
    "Otherwise, only predictions are shown."
)

uploaded = st.file_uploader("Upload test CSV", type=["csv"])

if uploaded is None:
    st.warning("No file uploaded. Using bundled test_data.csv for demo.")
    if os.path.exists("test_data.csv"):
        df = pd.read_csv("test_data.csv")
        st.success(f"Loaded test_data.csv — {len(df)} rows, {df.shape[1]} columns.")
    else:
        st.error("test_data.csv not found. Please run ml_models.py first.")
        st.stop()
else:
    df = pd.read_csv(uploaded)
    st.success(f"Uploaded file — {len(df)} rows, {df.shape[1]} columns.")

# Preview
with st.expander("Preview uploaded data (first 5 rows)"):
    st.dataframe(df.head(), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Prepare Features
# ─────────────────────────────────────────────────────────────────────────────

has_target = 'target' in df.columns

if has_target:
    y_true = df['target'].values
    X_test = df.drop(columns=['target'])
else:
    y_true = None
    X_test = df.copy()

# Load feature names from any model
reference_features = list(models[list(models.keys())[0]]['feature_names'])

# Check feature compatibility
missing = [c for c in reference_features if c not in X_test.columns]
if missing:
    st.error(f"Missing columns in uploaded data: {missing}")
    st.stop()

X_test = X_test[reference_features]

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Selected Models
# ─────────────────────────────────────────────────────────────────────────────

st.header("2. Evaluation Metrics")

results_rows = []
pred_dict = {}

for name in selected_models:
    if name not in models:
        continue
    m = models[name]
    clf = m['model']
    scaler = m.get('scaler')
    needs_sc = m.get('needs_scaling', False)

    X = scaler.transform(X_test) if (needs_sc and scaler is not None) else X_test.values

    y_pred = clf.predict(X)
    pred_dict[name] = y_pred

    if has_target:
        y_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else y_pred.astype(float)
        acc  = accuracy_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        mcc  = matthews_corrcoef(y_true, y_pred)
        results_rows.append({
            'Model':     name,
            'Accuracy':  round(acc,  4),
            'AUC':       round(auc,  4),
            'Precision': round(prec, 4),
            'Recall':    round(rec,  4),
            'F1':        round(f1,   4),
            'MCC':       round(mcc,  4)
        })

if has_target and results_rows:
    metrics_df = pd.DataFrame(results_rows).set_index('Model')
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color='#90EE90')
                        .highlight_min(axis=0, color='#FFB6C1'),
        use_container_width=True
    )
    st.caption("Green = highest value per metric | Red = lowest value per metric")

    # Winner
    best_model = metrics_df['Accuracy'].idxmax()
    st.success(f"Overall Winner (by Accuracy): **{best_model}**")

elif not has_target:
    st.info("No `target` column found — showing predictions only.")

# ─────────────────────────────────────────────────────────────────────────────
# Predictions Table
# ─────────────────────────────────────────────────────────────────────────────

st.header("3. Predictions")
pred_df = pd.DataFrame(pred_dict, index=X_test.index)
pred_df = pred_df.replace({0: 'Malignant', 1: 'Benign'})
if has_target:
    pred_df.insert(0, 'True Label', pd.Series(y_true, index=X_test.index).map({0: 'Malignant', 1: 'Benign'}))

with st.expander("Predictions (first 20 rows)", expanded=True):
    st.dataframe(pred_df.head(20), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────

if has_target and selected_models:
    st.header("4. Confusion Matrices")
    n_cols = min(3, len(selected_models))
    cols = st.columns(n_cols)

    for i, name in enumerate(selected_models):
        if name not in pred_dict:
            continue
        y_pred = pred_dict[name]
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'])
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        cols[i % n_cols].pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Classification Reports
# ─────────────────────────────────────────────────────────────────────────────

if has_target:
    st.header("5. Classification Reports")
    model_tab = st.selectbox("Select model for detailed report:", selected_models)

    if model_tab in pred_dict:
        y_pred = pred_dict[model_tab]
        report = classification_report(
            y_true, y_pred, target_names=['Malignant', 'Benign'])
        st.code(report, language='text')

# ─────────────────────────────────────────────────────────────────────────────
# Metrics Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

if has_target and results_rows:
    st.header("6. Visual Comparison")

    mdf = metrics_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    fig, ax = plt.subplots(figsize=(10, 5))
    mdf.T.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white')
    ax.set_title('Model Comparison — All Metrics', fontsize=13, fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "**M.Tech AIML — BITS Pilani WILP** | Machine Learning Assignment 2 | "
    "Ayush Sharma (2025AD05035) | Aug 2026"
)
