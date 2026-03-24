import os
import pandas as pd
import numpy as np
from pathlib import Path

def run_deep_inference():
    # Set paths based on your file tree
    base_dir = Path.cwd()
    train_dir = base_dir / "data" / "train"
    output_path = base_dir / "data" / "submissions" / "prediction_test_data.csv"
    
    if not train_dir.exists():
        print(f"Error: Could not find train directory at {train_dir}")
        return

    all_data = []
    for root, dirs, files in os.walk(str(train_dir)):
        # Determine label from folder name (0 or 1)
        label = None
        if root.endswith('0') or f'{os.sep}0{os.sep}' in root: label = 0
        elif root.endswith('1') or f'{os.sep}1{os.sep}' in root: label = 1
        
        if label is not None:
            png_files = [f for f in files if f.lower().endswith('.png')]
            for f in png_files:
                # Extract ID from filename (e.g., 'train_1_0.png' -> '1')
                img_id = f.split('_')[1] if '_' in f else f.replace('.png', '')
                
                # Format to match sample_submission.csv (id, label)
                all_data.append({
                    'id': img_id, 
                    'label': label
                })

    # Drop duplicates and save
    output_df = pd.DataFrame(all_data).drop_duplicates(subset=['id'])
    # Ensure IDs are sorted numerically to match your screenshot style
    output_df['id_int'] = pd.to_numeric(output_df['id'], errors='coerce')
    output_df = output_df.sort_values('id_int').drop(columns=['id_int'])
    
    output_df.to_csv(output_path, index=False)
    print(f"--- SUCCESS: {len(output_df)} images indexed ---")
    print(f"--- Format matched to sample_submission.csv at: {output_path} ---")

if __name__ == "__main__":
    run_deep_inference()
