import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDatasetFixed(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["Label"])
        fname = row["Filename"]
        img_path = self.image_root / str(label) / fname
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, fname
