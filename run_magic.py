import os
import torch
import pandas as pd
from src.engine import run_pipeline

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_path = './data/raw/test/train'
    test_path = './data/raw/test/test' 
    
    print(f"Starting pipeline on device: {device} (Reverting to optimal settings)")
    
    # Run the core pipeline
    acc, auc_score, count = run_pipeline(train_path, test_path, device)
    
    # Locate the generated predictions
    sub_path = 'data/submissions/submission.csv'
    local_path = 'submission.csv'
    
    if os.path.exists(sub_path):
        df = pd.read_csv(sub_path)
    elif os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        print("Error: Could not find prediction outputs.")
        return

    # Force exact required Kaggle formatting (id: 0-623, label: 0/1)
    df['id'] = range(len(df))
    df['label'] = (df['label'] >= 0.5).astype(int)
    df[['id', 'label']].to_csv('submission.csv', index=False)
    
    print("\n==========================================")
    print(f"Success! Model processed {count} images.")
    print("Formatted 'submission.csv' is ready to upload.")
    print("==========================================")

if __name__ == "__main__":
    main()
