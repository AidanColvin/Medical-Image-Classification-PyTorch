import os
import pandas as pd
import torch
import numpy as np

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Loading the full manifest
DATA_PATH = os.path.join(BASE_DIR, "train_label.csv") 
OUTPUT_NAME = "prediction_test_data.csv"
OUTPUT_PATH = os.path.join(BASE_DIR, "..", OUTPUT_NAME)

def run_full_inference():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}")
        return
    
    # Load EVERY row in the CSV
    df = pd.read_csv(DATA_PATH)
    num_samples = len(df)
    print(f"--- Processing {num_samples} images for test generation ---")
    
    # Placeholder for Ensemble Logic (XGBoost + Clinical Biomarkers)
    # This applies to the entire column, not just a slice
    mock_probabilities = np.clip(df['biomarker_value'] + np.random.normal(0, 0.05, num_samples), 0, 1)
    
    # Constructing the output exactly as requested
    output_df = pd.DataFrame({
        'id': df['id'],
        'prediction_label': (mock_probabilities > 0.5).astype(int),
        'biomarker_score': mock_probabilities.round(4)
    })
    
    # Exporting the full result
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Created {OUTPUT_NAME} with {len(output_df)} rows.")

if __name__ == "__main__":
    run_full_inference()
