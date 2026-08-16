# breast-cancer-classifier-app
## ML Assignment 2 : 
A Streamlit-powered web app that classifies breast tumors as benign or malignant using multiple machine learning models trained on the Wisconsin Breast Cancer dataset.

**Student:** Ayush Sharma | **BITS ID:** 2025AD05035  
**Course:** Machine Learning | M.Tech AIML : BITS Pilani WILP  

## LIVE Link to deployed app on Streamlit Community Cloud
**LINK:** [https://2025ad05035-ayushsharma-breast-cancer-classifier-app-app-q4hnka.streamlit.app/](https://2025ad05035-ayushsharma-breast-cancer-classifier-app-app-q4hnka.streamlit.app/)

---

## a. Problem Statement

Design and implement a multi-model machine learning classification pipeline that:
1. Trains multiple classification algorithms on a real-world medical dataset
2. Evaluates and compares each model using six metrics
3. Deploys the trained models as an interactive Streamlit web application

---

## b. Dataset Description

**Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) / `sklearn.datasets.load_breast_cancer()`  
**Instances:** 569  
**Features:** 30 numerical features (real-valued, computed from digitized images of fine needle aspirate of breast mass)  
**Target:** Binary classification; 0 = Malignant, 1 = Benign  
**Class Distribution:** 212 malignant (37.3%), 357 benign (62.7%)

**Key Features Include:**
- Mean radius, texture, perimeter, area, smoothness
- Compactness, concavity, concave points, symmetry, fractal dimension
- (x3 variants: mean, standard error, worst - totaling 30 features)

**Train/Test Split:** 80/20 stratified (455 train, 114 test)  
**Preprocessing:** StandardScaler applied to LR, kNN, Naive Bayes; tree-based models use raw features

---

## c. GitHub Repository Link

> **[https://github.com/2025ad05035-ayushsharma/breast-cancer-classifier-app](https://github.com/2025ad05035-ayushsharma/breast-cancer-classifier-app)**



---

## e. Models Used and their Comparison Table

|  Model | Accuracy | AUC | Precision | Recall | F1 | MCC| 
|---|---|---|---|---|---|---|
|  Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623|  
|  Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174|  
|  kNN | 0.9737 | 0.9884 | 0.96 | 1 | 0.9796 | 0.9442|  
|  Naive Bayes Gaussian | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492|  
|  Naive Bayes Multinomial | 0.9298 | 0.9312 | 0.9211 | 0.9722 | 0.9459 | 0.8487|  
|  Random Forest | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054|  


### Observations on Model Performance

| ML Model Name | Observation |
|---------------|-------------|
| Logistic Regression | **Best overall performer**, achieves 98.25% accuracy and 0.9954 AUC. Excels because the cancer classification problem is largely linearly separable in the 30-dimensional feature space. Benefits strongly from feature scaling. |
| Decision Tree | Lowest accuracy (91.23%) and notably lower AUC (0.9157), indicating some overfitting even with max_depth=8. Tree-based models tend to create axis-aligned decision boundaries that can overfit in high-dimensional spaces. |
| kNN | Second best accuracy (97.37%) with perfect recall (1.0000), meaning it misses NO malignant cases, highly desirable in medical diagnosis. Sensitive to scaling; needs careful k selection. |
| Naive Bayes | Moderate performance (92.98%). Gaussian NB assumes feature independence, which is violated here (features are correlated). Despite this, it generalises well due to the probabilistic framework and high AUC (0.9868). |
| Random Forest | Strong ensemble performance (95.61% accuracy, 0.9937 AUC). Robust to outliers and overfitting through bagging. Slightly below LR and kNN on this dataset but would likely outperform on noisier, larger datasets. |


**Overall Winner: Logistic Regression**, best on Accuracy, AUC, Precision, F1, and MCC. For medical datasets with relatively clean, linearly separable features, Logistic Regression consistently delivers excellent results with high interpretability.

---

*BITS Pilani WILP | M.Tech AIML | Machine Learning Assignment 2 | Aug 2026*
