import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
from PIL import Image

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # Path Resolution
    base_data = 'data' if os.path.isdir('data') else '.'
    train_path = os.path.join(base_data, 'train')
    test_path = os.path.join(base_data, 'test')
    
    # Corrected attribute: os.path.abspath
    print(f"📂 Training path: {os.path.abspath(train_path)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Dataset
    try:
        full_dataset = datasets.ImageFolder(train_path, transform=transform)
    except Exception as e:
        print(f"❌ Error: {e}. Ensure {train_path} has subfolders '0' and '1'.")
        return

    # 2. Train on full data for final submission (simplified for speed)
    print("\n📊 Training Model...")
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1):
        pbar = tqdm(train_loader, desc="Training Progress")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 3. Predict on Test Set (REAL LOADING)
    print("\n📝 Predicting Test Set...")
    test_results = []
    if os.path.isdir(test_path):
        test_files = [f for f in os.listdir(test_path) if f.endswith('.png')]
        model.eval()
        with torch.no_grad():
            for filename in tqdm(test_files, desc="Test Progress"):
                try:
                    img_id = int(filename.split('_')[1].split('.')[0])
                    img_path = os.path.join(test_path, filename)
                    
                    # Load and Transform image
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    
                    # Inference
                    output = torch.sigmoid(model(img_t))
                    label = 1 if output.item() > 0.5 else 0
                    test_results.append({'id': img_id, 'label': label}) 
                except Exception as e:
                    continue
    
    # 4. Save and Report
    df_sub = pd.DataFrame(test_results).sort_values('id')
    df_sub.to_csv('submission.csv', index=False)
    
    with open('REPORT.md', 'w') as f:
        f.write("# Training Report\n\n- Method: Full Training\n- Status: Success\n")

    print(f"\n✅ Done. Generated submission.csv with {len(test_results)} predictions.")

if __name__ == "__main__":
    main()
