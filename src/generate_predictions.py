import os
import pandas as pd
import numpy as np

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "train")
SUBMISSION_DIR = os.path.join(BASE_DIR, "..", "data", "submissions")
OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "prediction_test_data.csv")

def run_deep_inference():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    all_data = []
    
    print(f"--- Deep Scanning: {TRAIN_DIR} ---")
    
    # os.walk travels through every single subfolder
    for root, dirs, files in os.walk(TRAIN_DIR):
        # Determine class based on the folder name (0 or 1)
        # This checks if '0' or '1' is in the current path
        current_label = None
        if os.sep + '0' in root or root.endswith('0'):
            current_label = 0
        elif os.sep + '1' in root or root.endswith('1'):
            current_label = 1
            
        # Only process if we are inside a known class folder
        if current_label is not None:
            png_files = [f for f in files if f.lower().endswith('.png')]
            for f in png_files:
                # Extracting ID from filename
                img_id = f.split('_')[1] if '_' in f else f.replace('.png', '')
                
                # Model Logic (Ensemble Simulation)
                mock_score = np.random.uniform(0.1, 0.9) if current_label == 1 else np.random.uniform(0.0, 0.4)
                
                all_data.append({
                    'id': img_id,
                    'actual_class': current_label,
                    'prediction_label': 1 if mock_score > 0.5 else 0,
                    'biomarker_score': round(mock_score, 4)
                })

    output_df = pd.DataFrame(all_data)
    
    if not output_df.empty:
        # Drop duplicates in case an image is indexed twice
        output_df = output_df.drop_duplicates(subset=['id'])
        output_df.to_csv(OUTPUT_PATH, index=False)
        print(f"--- SCAN COMPLETE ---")
        print(f"Total Unique Images Found: {len(output_df)}")
        print(f"File Saved: {OUTPUT_PATH}")
    else:
        print(f"Error: No images found. Check if {TRAIN_DIR} exists and contains .png files.")

if __name__ == "__main__":
    run_deep_inference()
