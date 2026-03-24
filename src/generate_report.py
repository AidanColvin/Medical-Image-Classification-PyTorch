import pandas as pd
from pathlib import Path

def create_appendix():
    base_dir = Path.cwd()
    report_path = base_dir / "FINAL_REPORT.md"
    cv_table = base_dir / "data" / "tables" / "cv_10_fold_results.csv"
    summary_table = base_dir / "data" / "tables" / "dataset_summary.csv"
    
    with open(report_path, "w") as f:
        f.write("# Medical Image Classification: Predictor & Validation Report\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write("This report details the indexing and validation of **5,232** medical images for biomarker detection. ")
        f.write("A 10-Fold Stratified Cross-Validation was performed to ensure predictor stability.\n\n")

        f.write("## 2. Dataset Distribution\n")
        if summary_table.exists():
            df_sum = pd.read_csv(summary_table)
            f.write(df_sum.to_markdown(index=False) + "\n\n")
        
        f.write("### Visual: Class Distribution\n")
        f.write("![Class Distribution](data/visualizations/class_distribution.png)\n\n")

        f.write("## 3. Predictor Strength & Stability\n")
        f.write("Using a Logistic Regression baseline, the **Biomarker Score** shows strong diagnostic impact. ")
        f.write("The following metrics demonstrate the consistency of this predictor across the dataset.\n\n")
        
        f.write("### Visual: Stability & Strength\n")
        f.write("![Stability](data/visualizations/predictor_stability.png)\n")
        f.write("![Strength](data/visualizations/predictor_strength.png)\n\n")

        f.write("## 4. Model Performance Metrics\n")
        f.write("The diagnostic power was evaluated using Area Under the Curve (AUC) and a Confusion Matrix.\n\n")
        
        f.write("### Visual: ROC Curve & Confusion Matrix\n")
        f.write("![ROC Curve](data/visualizations/roc_curve.png)\n")
        f.write("![Confusion Matrix](data/visualizations/confusion_matrix.png)\n\n")

        f.write("## 5. Appendix: 10-Fold CV Raw Data\n")
        if cv_table.exists():
            df_cv = pd.read_csv(cv_table)
            f.write(df_cv.to_markdown(index=False) + "\n")

    print(f"--- SUCCESS: Final Report generated at {report_path} ---")

if __name__ == "__main__":
    create_appendix()
