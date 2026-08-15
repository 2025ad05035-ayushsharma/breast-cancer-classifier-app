# breast-cancer-classifier-app
## ML Assignment 2 : 
A Streamlit-powered web app that classifies breast tumors as benign or malignant using multiple machine learning models trained on the Wisconsin Breast Cancer dataset.

**Student:** Ayush Sharma | **BITS ID:** 2025AD05035  
**Course:** Machine Learning | M.Tech AIML : BITS Pilani WILP  


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

