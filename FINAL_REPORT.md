# Medical Image Classification: Predictor & Validation Report

## 1. Executive Summary
This report details the indexing and validation of **5,232** medical images for biomarker detection. A 10-Fold Stratified Cross-Validation was performed to ensure predictor stability.

## 2. Dataset Distribution
### Visual: Class Distribution
![Class Distribution](data/visualizations/class_distribution.png)

## 3. Predictor Strength & Stability
Using a Logistic Regression baseline, the **Biomarker Score** shows strong diagnostic impact. The following metrics demonstrate the consistency of this predictor across the dataset.

### Visual: Stability & Strength
![Stability](data/visualizations/predictor_stability.png)
![Strength](data/visualizations/predictor_strength.png)

## 4. Model Performance Metrics
The diagnostic power was evaluated using Area Under the Curve (AUC) and a Confusion Matrix.

### Visual: ROC Curve & Confusion Matrix
![ROC Curve](data/visualizations/roc_curve.png)
![Confusion Matrix](data/visualizations/confusion_matrix.png)

## 5. Appendix: 10-Fold CV Raw Data
|   Fold |   Coefficient |   Accuracy |
|-------:|--------------:|-----------:|
|      1 |       17.5866 |   0.959924 |
|      2 |       17.6928 |   0.95229  |
|      3 |       17.6494 |   0.965583 |
|      4 |       17.4874 |   0.965583 |
|      5 |       17.5932 |   0.957935 |
|      6 |       17.6058 |   0.954111 |
|      7 |       17.6694 |   0.948375 |
|      8 |       17.5271 |   0.978967 |
|      9 |       17.5727 |   0.961759 |
|     10 |       17.6726 |   0.950287 |
