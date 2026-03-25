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
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_curve
from PIL import Image

def run_pipeline(train_path, test_path, device):
    vis_dir = 'data/visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs('data/submissions', exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Dataset
    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    valid_indices = [i for i, (p, l) in enumerate(full_dataset.imgs) 
                     if os.path.basename(os.path.dirname(p)) in ['0', '1']]
    dataset = Subset(full_dataset, valid_indices)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Model & Training
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    history = {'loss': []}
    all_probs, all_labels = [], []

    print(f"🚀 Training for 5 Epochs...")
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/5")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if epoch == 4: # Collect final epoch stats
                all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        history['loss'].append(epoch_loss / len(loader))

    # 3. Metrics Calculation
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)

    # 4. Generate & Save ALL Visuals
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.savefig(f'{vis_dir}/confusion_matrix.png')
    plt.close()

    # Plot 2: ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='orange', label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{vis_dir}/roc_curve.png')
    plt.close()

    # Plot 3: Training Loss
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, 6), history['loss'], marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{vis_dir}/loss_curve.png')
    plt.close()

    # Plot 4: Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'{vis_dir}/pr_curve.png')
    plt.close()

    # 5. Inference
    results = []
    if os.path.isdir(test_path):
        model.eval()
        for f in os.listdir(test_path):
            if f.endswith('.png'):
                try:
                    img_id = int(f.split('_')[1].split('.')[0])
                    img = Image.open(os.path.join(test_path, f)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prob = torch.sigmoid(model(img_t)).item()
                    results.append({'id': img_id, 'label': 1 if prob > 0.5 else 0})
                except: continue

    df_sub = pd.DataFrame(results).sort_values('id') if results else pd.DataFrame(columns=['id', 'label'])
    df_sub.to_csv('submission.csv', index=False)
    
    return acc, auc(fpr, tpr), len(results)
