import os
import shutil
import glob
import torch
import torch.nn as nn
import re
import seaborn as sns
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image

def organize_and_prep():
    print("\n--- [1/6] Cleaning Root & Organizing src/ ---")
    for d in ['data/visualizations', 'data/tables', 'submissions', 'src']:
        os.makedirs(d, exist_ok=True)
    
    # Move all scripts (except main.py) to src/
    all_scripts = glob.glob("*.py")
    for s in all_scripts:
        if s not in ['main.py', 'run_final_submission.py']:
            shutil.move(s, os.path.join('src', s))
    
    # Delete old messy reports
    for old in ['Report', 'FINAL_REPORT.md', 'PROJECT_REPORT.md', 'Visualizations']:
        if os.path.exists(old): os.remove(old)
    print("✓ Root cleaned. Scripts moved to src/.")

class TestDataset(Dataset):
    def __init__(self, directory, transform):
        self.filepaths = sorted(glob.glob(os.path.join(directory, '*.png')))
        self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img_id = int(re.findall(r'\d+', os.path.basename(path))[0])
        return self.transform(Image.open(path).convert('RGB')), img_id

def build_model(device):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*14*14, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device)

def run_5_fold_cv():
    print("--- [2/6] Executing 5-Fold Cross Validation ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_data = datasets.ImageFolder('data/train', transform=transform)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()
    model = build_model(device)
    
    metrics = {'loss': [], 'true': [], 'prob': [], 'pred': []}
    
    for fold, (t_idx, v_idx) in enumerate(kfold.split(train_data)):
        train_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(t_idx))
        val_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(v_idx))
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(1): # Reduced to 1 for speed; increase for better results
            model.train()
            loop = tqdm(train_loader, desc=f"Fold {fold+1}/5", leave=False)
            epoch_loss = 0
            for imgs, lbls in loop:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            metrics['loss'].append(epoch_loss / len(train_loader))
        
        model.eval()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                out = model(imgs.to(device))
                probs = torch.softmax(out, dim=1)[:, 1]
                _, preds = torch.max(out, 1)
                metrics['true'].extend(lbls.numpy())
                metrics['prob'].extend(probs.cpu().numpy())
                metrics['pred'].extend(preds.cpu().numpy())
    
    return model, device, transform, metrics

def save_outputs(model, metrics):
    print("--- [3/6] Generating New Visuals & Tables ---")
    # 1. Confusion Matrix (New Visual)
    cm = confusion_matrix(metrics['true'], metrics['pred'])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix: Validation Set')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('data/visualizations/confusion_matrix.png')
    plt.close()

    # 2. AUC Curve
    fpr, tpr, _ = roc_curve(metrics['true'], metrics['prob'])
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color='orange', label=f'AUC: {auc(fpr, tpr):.2f}')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.title('ROC Curve (5-Fold CV)')
    plt.legend()
    plt.savefig('data/visualizations/auc_curve.png')
    plt.close()

    # 3. Parameter Impact Table
    impact = [{'Layer': n, 'Value': p.abs().mean().item()} for n, p in model.named_parameters() if 'weight' in n]
    impact_df = pd.DataFrame(impact).sort_values('Value', ascending=False)
    impact_df.to_csv('data/tables/parameter_impact.csv', index=False)
    return impact_df

def generate_submission(model, device, transform):
    print("--- [4/6] Creating Formatting-Matched Submission ---")
    test_dir = 'data/test' if os.path.exists('data/test') else 'test'
    loader = DataLoader(TestDataset(test_dir, transform), batch_size=32, shuffle=False)
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc="Inference"):
            _, preds = torch.max(model(imgs.to(device)), 1)
            for img_id, p in zip(ids, preds):
                results.append({'id': img_id, 'label': p.item()})
    
    # Strictly 2 columns: id, label
    sub_df = pd.DataFrame(results).sort_values('id')
    sub_df.to_csv('submission.csv', index=False)
    print("✓ submission.csv saved to root.")

def write_final_report(impact_df):
    print("--- [5/6] Writing Updated PROJECT_REPORT.md ---")
    report = f"""# Medical Image Classification: Master Report
**Status:** All Visuals & Tables Regenerated (5-Fold CV)

## 1. Classification Performance
### Confusion Matrix
Determines how often the model correctly identifies Pneumonia vs. Normal cases.
![CM](data/visualizations/confusion_matrix.png)

### AUC/ROC
![AUC](data/visualizations/auc_curve.png)

## 2. Component Impact
The following table shows the mean absolute weights of each network layer.
{impact_df.to_markdown(index=False)}
"""
    with open('PROJECT_REPORT.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    organize_and_prep()
    model, device, transform, metrics = run_5_fold_cv()
    impact_df = save_outputs(model, metrics)
    generate_submission(model, device, transform)
    write_final_report(impact_df)
    # Move the run script to src at the very end
    shutil.move('run_final_submission.py', 'src/run_final_submission.py')
    print("\n🚀 DONE: Root is clean, submission is ready, and report is updated.")
