import torch
import pandas as pd
import os
import sys
from tqdm import tqdm
from src.engine import get_model  # Assumes your model loader is here

def main():
    """
    Orchestrates a robust inference pass.
    Saves 'submission_v13.csv' to root and 'data/submissions/'
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting v13 pipeline on: {device}")

    # 1. Setup paths
    root_file = "submission_v13.csv"
    data_dir = "data/submissions"
    data_file = os.path.join(data_dir, "submission_v13.csv")
    os.makedirs(data_dir, exist_ok=True)

    # 2. Load weights
    model = get_model()
    ckpt = 'data/models/best_model.pth'
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
    else:
        print(f"Error: {ckpt} not found. Ensure training is complete.")
        return

    # 3. Inference on test set (Exactly 624 images)
    test_dir = "./data/raw/test/test"
    images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Processing {len(images)} images to mirror template...")
    
    results = []
    with torch.no_grad():
        for i, img_name in enumerate(tqdm(images)):
            # Replace with your actual image loading & prediction:
            # pred = model(load_img(os.path.join(test_dir, img_name)))
            label = 1 # Placeholder for model prediction
            results.append({"id": i, "label": int(label)})

    # 4. Save formatted CSV
    df = pd.DataFrame(results)
    df[['id', 'label']].to_csv(root_file, index=False)
    df[['id', 'label']].to_csv(data_file, index=False)
    
    print(f"\n[LOCAL] Generated {root_file}")
    print(f"[LOCAL] Synchronized to {data_file}")

if __name__ == "__main__":
    main()
