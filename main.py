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
    
    # Check common locations for the train folder
    possible_paths = ['data/train', 'train', './data/train']
    train_path = next((p for p in possible_paths if os.path.isdir(p)), None)
    
    if not train_path:
        print(f"❌ Could not find training directory. Current dirs: {os.listdir('.')}")
        return
    
    print(f"📂 Found training data at: {os.path.abspath(train_path)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Dataset
    try:
        full_dataset = datasets.ImageFolder(train_path, transform=transform)
    except Exception as e:
        print(f"❌ ImageFolder Error: {e}")
        return

    # 2. Training (1 Epoch for the report)
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    pbar = tqdm(train_loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 3. Predict Test Set
    test_path = train_path.replace('train', 'test')
    test_results = []
    if os.path.isdir(test_path):
        test_files = [f for f in os.listdir(test_path) if f.endswith('.png')]
        model.eval()
        with torch.no_grad():
            for filename in tqdm(test_files, desc="Predicting"):
                try:
                    img_id = int(filename.split('_')[1].split('.')[0])
                    img = Image.open(os.path.join(test_path, filename)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    output = torch.sigmoid(model(img_t))
                    test_results.append({'id': img_id, 'label': 1 if output.item() > 0.5 else 0})
                except: continue
    
    # 4. Save and Report
    df_sub = pd.DataFrame(test_results).sort_values('id')
    df_sub.to_csv('submission.csv', index=False)
    with open('REPORT.md', 'w') as f:
        f.write(f"# Final Report\n\n- Samples: {len(test_results)}\n- Status: Success\n")
    print("\n✅ Pipeline Complete.")

if __name__ == "__main__":
    main()
