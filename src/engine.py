import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from PIL import Image

def run_pipeline(train_path, test_path, device):
    os.makedirs('visuals', exist_ok=True)
    os.makedirs('data/submissions', exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Dataset
    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    # Filter for actual class folders
    valid_indices = [i for i, (path, label) in enumerate(full_dataset.imgs) 
                     if os.path.basename(os.path.dirname(path)) in ['0', '1']]
    dataset = Subset(full_dataset, valid_indices)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Model Setup
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop (5 Epochs)
    print(f"🚀 Training on {len(dataset)} images...")
    all_probs, all_labels = [], []
    for epoch in range(5):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/5")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics & Visuals
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Save Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='viridis')
    plt.title(f"Confusion Matrix (Acc: {acc:.2%})")
    plt.savefig('visuals/confusion_matrix.png')
    plt.close()

    # Save ROC Curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('visuals/roc_curve.png')
    plt.close()

    # 5. Inference
    results = []
    if os.path.isdir(test_path):
        test_files = [f for f in os.listdir(test_path) if f.endswith('.png')]
        model.eval()
        for f in tqdm(test_files, desc="Inference"):
            try:
                img_id = int(f.split('_')[1].split('.')[0])
                img = Image.open(os.path.join(test_path, f)).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.sigmoid(model(img_t)).item()
                results.append({'id': img_id, 'label': 1 if prob > 0.5 else 0})
            except: continue

    # Save CSVs
    df_sub = pd.DataFrame(results)
    if not df_sub.empty:
        df_sub = df_sub.sort_values('id')
    df_sub.to_csv('submission.csv', index=False)
    df_sub.to_csv('data/submissions/submission_archive.csv', index=False)
    
    return acc, roc_auc, len(results)
