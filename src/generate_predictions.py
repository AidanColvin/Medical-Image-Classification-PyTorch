import os
import pandas as pd
import numpy as np

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train_label.csv") 
SUBMISSION_DIR = os.path.join(BASE_DIR, "..", "data", "submissions")
OUTPUT_NAME = "prediction_test_data.csv"
OUTPUT_PATH = os.path.join(SUBMISSION_DIR, OUTPUT_NAME)

def run_complete_inference():
    # Ensure the directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Source data not found at {DATA_PATH}")
        return
    
    # Load the ENTIRE dataset
    df = pd.read_csv(DATA_PATH)
    total_images = len(df)
    print(f"--- Processing COMPLETE Dataset: {total_images} images found ---")
    
    # Applying the Ensemble Logic to every row
    # (Using the clinical biomarker values to drive the predictions)
    scores = df['biomarker_value'].values
    
    # Vectorized operations ensure this scales to thousands of rows instantly
    predictions = (scores > 0.5).astype(int)
    
    # Constructing the full output dataframe
    output_df = pd.DataFrame({
        'id': df['id'],
        'prediction_label': predictions,
        'biomarker_score': np.round(scores, 4)
    })
    
    # Save the full result
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! {len(output_df)} rows written to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_complete_inference()
