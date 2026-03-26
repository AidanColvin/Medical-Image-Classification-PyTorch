import torch
import pandas as pd
import os
from src.utils import get_versioned_path
# Assuming your model and loader are defined in your src directory
# from src.model import MedicalModel 

def generate():
    # 1. Setup Environment
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Mocking labels for the structure - replace this with your actual model.predict() loop
    # This ensures the 2-column format: id, label
    test_ids = range(500) # Adjust to your actual test set size
    predictions = [1 if x % 3 == 0 else 0 for x in test_ids] # Replace with model(images)
    
    df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    # 3. Save using versioning logic
    output_path = get_versioned_path("submission.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved to: {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate()
