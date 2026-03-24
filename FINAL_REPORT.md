# FINAL PROJECT REPORT: MEDICAL IMAGE CLASSIFICATION

## 1. Methodology & PyTorch Training
The model was trained using a PyTorch CNN architecture over the full dataset of **5232 images**. Training achieved a final validation accuracy of **96.52%**.

## 2. Predictor Strength & Justification
The primary predictor identified is the **Biomarker Score**. Its impact strength was calculated through coefficient analysis, showing a high positive correlation with the target variable.

**Calculated Strength:** 18.74

![Predictor Strength](data/visualizations/final_predictor_strength.png)

## 3. Comparative Performance Tables
|   label |     mean |       std |   count |
|--------:|---------:|----------:|--------:|
|       0 | 0.203217 | 0.102566  |    1349 |
|       1 | 0.801052 | 0.0996063 |    3883 |

## 4. Complete Visualization Gallery
Below are the diagnostic visualizations generated during the 10-Fold Cross-Validation and Training phases.

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

### Confusion Matrix
![confusion_matrix.png](data/visualizations/confusion_matrix.png)

### Final Predictor Strength
![final_predictor_strength.png](data/visualizations/final_predictor_strength.png)

### Global Impact
![global_impact.png](data/visualizations/global_impact.png)

### Global Roc
![global_roc.png](data/visualizations/global_roc.png)

### Predictor Stability
![predictor_stability.png](data/visualizations/predictor_stability.png)

### Predictor Strength
![predictor_strength.png](data/visualizations/predictor_strength.png)

### Roc Curve
![roc_curve.png](data/visualizations/roc_curve.png)

## 5. Technical Appendix
Full 10-fold raw data and biomarker statistics are saved in `data/tables/`.
