import os
import shutil
import glob
import torch
import torch.nn as nn
import re
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.model_selection import KFold
from tqdm import tqdm
from PIL import Image

def total_reset():
    print("\n--- [1/6] Nuclear Reset: Purging All Old Data & Visuals ---")
    # Force delete and recreate output directories to ensure NO old files remain
    for d in ['data/visualizations', 'data/tables', 'submissions', 'src']:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    # Move all .py files except main.py and this one to src/
    for f in glob.glob("*.py"):
        if f not in ['main.py', 'final_reconstruction.py']:
            shutil.move(f, os.path.join('src', f))
            
    # Remove all old report variants
    for r in ['Report', 'FINAL_REPORT.md', 'PROJECT_REPORT.md', 'Visualizations']:
        if os.path.exists(r): os.remove(r)
    print("✓ Workspace is now a blank slate.")

class TestDataset(Dataset):
    def __init__(self, directory, transform):
        self.filepaths = sorted(glob.glob(os.path.join(directory, '*.png')))
        self.transform = transform
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img_id = int(re.findall(r'\d+', os.path.basename(path))[0])
        return self.transform(Image.open(path).convert('RGB')), img_id

def get_net(device):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*14*14, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device)

def run_pipeline():
    print("--- [2/6] Running 5-Fold CV Training ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_set = datasets.ImageFolder('data/train', transform=transform)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = get_net(device)
    
    history = {'loss': [], 'y_true': [], 'y_prob': [], 'y_pred': []}
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(train_set)):
        t_loader = DataLoader(train_set, batch_size=32, sampler=SubsetRandomSampler(t_idx))
        v_loader = DataLoader(train_set, batch_size=32, sampler=SubsetRandomSampler(v_idx))
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        model.train()
        fold_loss = 0
        pbar = tqdm(t_loader, desc=f"Fold {fold+1}/5", leave=False)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            output = model(imgs)
            loss = torch.nn.CrossEntropyLoss()(output, lbls)
            loss.backward()
            opt.step()
            fold_loss += loss.item()
        history['loss'].append(fold_loss / len(t_loader))
        
        # Validation
        model.eval()
        with torch.no_grad():
            for imgs, lbls in v_loader:
                out = model(imgs.to(device))
                probs = torch.softmax(out, dim=1)[:, 1]
                preds = torch.argmax(out, dim=1)
                history['y_true'].extend(lbls.numpy())
                history['y_prob'].extend(probs.cpu().numpy())
                history['y_pred'].extend(preds.cpu().numpy())

    print("--- [3/6] Generating Advanced Visuals & CSV Tables ---")
    # 1. Confusion Matrix
    cm = confusion_matrix(history['y_true'], history['y_pred'])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.savefig('data/visualizations/confusion_matrix.png')
    pd.DataFrame(cm).to_csv('data/tables/confusion_matrix_data.csv')

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(history['y_true'], history['y_prob'])
    plt.figure(figsize=(7,5))
    plt.plot(rec, prec, color='purple', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('data/visualizations/precision_recall.png')
    pd.DataFrame({'Precision': prec[:-1], 'Recall': rec[:-1]}).to_csv('data/tables/precision_recall_data.csv')

    # 3. AUC Curve
    fpr, tpr, _ = roc_curve(history['y_true'], history['y_prob'])
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color='red', label=f'AUC: {auc(fpr, tpr):.2f}')
    plt.plot([0,1],[0,1], ls='--')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('data/visualizations/auc_curve.png')
    pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_csv('data/tables/roc_data.csv')

    # 4. Impact Table
    impact = [{'Layer': n, 'Weight_Impact': p.abs().mean().item()} for n, p in model.named_parameters() if 'weight' in n]
    impact_df = pd.DataFrame(impact).sort_values('Weight_Impact', ascending=False)
    impact_df.to_csv('data/tables/layer_impact.csv', index=False)

    print("--- [4/6] Generating Root Submission (id,label) ---")
    test_dir = 'data/test' if os.path.exists('data/test') else 'test'
    test_loader = DataLoader(TestDataset(test_dir, transform), batch_size=32, shuffle=False)
    model.eval()
    sub_results = []
    with torch.no_grad():
        for imgs, ids in tqdm(test_loader, desc="Final Inference"):
            preds = torch.argmax(model(imgs.to(device)), dim=1)
            for img_id, p in zip(ids, preds):
                sub_results.append({'id': img_id, 'label': p.item()})
    
    pd.DataFrame(sub_results).sort_values('id').to_csv('submission.csv', index=False)

    print("--- [5/6] Finalizing PROJECT_REPORT.md ---")
    report_text = f"""# Medical Image Analysis: Final Technical Report
**Generated:** {pd.Timestamp.now()}

## 1. Diagnostic Performance
The model was validated using a 5-Fold Cross-Validation strategy.

### Confusion Matrix (Normal vs Pneumonia)
![CM](data/visualizations/confusion_matrix.png)

### Precision-Recall Analysis
![PR](data/visualizations/precision_recall.png)

### ROC/AUC Performance
![AUC](data/visualizations/auc_curve.png)

## 2. Feature & Layer Importance
Determines which convolutional stages were most active during classification.
{impact_df.to_markdown(index=False)}

*Full data tables for all visualizations are stored in `data/tables/`.*
"""
    with open('PROJECT_REPORT.md', 'w') as f:
        f.write(report_text)

if __name__ == "__main__":
    total_reset()
    run_pipeline()
    # Move this runner to src at the very end
    shutil.move('final_reconstruction.py', 'src/final_reconstruction.py')
    print("\n🚀 SUCCESS: Submission generated in root. Report updated. Workspace cleaned.")
