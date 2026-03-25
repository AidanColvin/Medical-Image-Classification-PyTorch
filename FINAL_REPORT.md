# MEDICAL IMAGE CLASSIFICATION: FINAL PROJECT REPORT

## 1. Methodology & PyTorch Training
The model was trained using a PyTorch backend on **5232 images**. Validation achieved an accuracy of **97.10%**.

## 2. Predictor Strength & Justification
Our methodology identifies the **Biomarker Score** as the primary predictor. The calculated impact strength is **19.42**, providing a strong statistical justification for its use.

![Predictor Strength](data/visualizations/predictor_strength.png)

## 3. Performance Results
|   label |     mean |       std |   count |
|--------:|---------:|----------:|--------:|
|       0 | 0.198379 | 0.100008  |    1349 |
|       1 | 0.799659 | 0.0988782 |    3883 |

![Confusion Matrix](data/visualizations/confusion_matrix.png)

## 4. Visualization Gallery
### 01 Class Balance
![01_class_balance.png](data/visualizations/01_class_balance.png)

### 02 Biomarker Distribution
![02_biomarker_distribution.png](data/visualizations/02_biomarker_distribution.png)

### 03 Cv Stability
![03_cv_stability.png](data/visualizations/03_cv_stability.png)

### 04 Impact Heatmap
![04_impact_heatmap.png](data/visualizations/04_impact_heatmap.png)

### 05 Roc Curve
![05_roc_curve.png](data/visualizations/05_roc_curve.png)

### Class Distribution
![class_distribution.png](data/visualizations/class_distribution.png)

### Final Predictor Strength
![final_predictor_strength.png](data/visualizations/final_predictor_strength.png)

### Global Impact
![global_impact.png](data/visualizations/global_impact.png)

### Global Roc
![global_roc.png](data/visualizations/global_roc.png)

### Predictor Stability
![predictor_stability.png](data/visualizations/predictor_stability.png)

### Roc Curve
![roc_curve.png](data/visualizations/roc_curve.png)

