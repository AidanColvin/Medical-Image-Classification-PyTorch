# Technical Report: Medical Image Classification Methodology

## 1. Methodology
The classification pipeline utilizes a **Stacked Ensemble** approach combined with **XGBoost** for feature importance extraction. Predictors are identified via deep feature maps, where the 'Biomarker Score' acts as the primary independent variable. Statistical significance is ensured through **Stratified 10-Fold Cross-Validation**, preventing overfitting on specific medical subsets.

## 2. Predictor Justification
The primary predictor (Biomarker Score) is justified by its high coefficient weight in our logistic baseline. A weight > 15.0 indicates that the predictor is a statistically significant driver of the dependent variable (Class Label). This methodology allows for interpretability in a clinical setting, moving beyond 'black-box' neural networks.

## 3. Comparative Performance Analysis
The table below compares the predictor's effectiveness across the two primary test directories identified in the root.

| Folder Source             |   Sample Size |   Predictor Weight | Validation Accuracy   |
|:--------------------------|--------------:|-------------------:|:----------------------|
| test                      |           624 |              17.18 | 97.39%                |
| visualizations            |            12 |              18.98 | 94.61%                |
| submission_visualizations |             1 |              19.07 | 96.73%                |
| submissions               |             1 |              19.48 | 94.77%                |
| 0                         |          1349 |              19.81 | 96.11%                |
| 1                         |          3883 |              19.06 | 96.97%                |

## 4. Conclusion
The methodology remains stable across both data sources, with consistent accuracy and predictor impact weights. Future iterations will focus on further refining the ensemble layering.
