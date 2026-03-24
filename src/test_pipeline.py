import torch
import pandas as pd
import os
import pytest
from PIL import Image
from main import build_resnet50, CONFIG, ChestXrayDataset, get_transforms

def test_pytorch_version():
    """Ensure we are on a modern torch version."""
    assert torch.__version__.startswith("2") or torch.__version__.startswith("3")

def test_model_params():
    """Verify ResNet50 complexity."""
    model = build_resnet50()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 20_000_000

def test_csv_structure():
    """Check for required data columns."""
    df = pd.read_csv("train_label.csv")
    assert all(col in df.columns for col in ["Filename", "Label"])

def test_label_integrity():
    """Verify labels are strictly binary."""
    df = pd.read_csv("train_label.csv")
    assert df["Label"].isin([0, 1]).all()

def test_dataset_output():
    """Test the PyTorch Dataset return types."""
    df = pd.read_csv("train_label.csv")
    train_tf, _ = get_transforms()
    dataset = ChestXrayDataset(df, "train", transform=train_tf)
    img, label, fname = dataset[0]
    assert torch.is_tensor(img)
    assert isinstance(label, (int, torch.Tensor))

def test_image_resolution():
    """Ensure images in the folder match expected sizes."""
    df = pd.read_csv("train_label.csv")
    sample_img = os.path.join("train", df.iloc[0]["Filename"])
    with Image.open(sample_img) as img:
        assert img.size[0] >= 224 # Check width

def test_transform_normalization():
    """Verify tensor scaling."""
    _, val_tf = get_transforms()
    dummy_img = Image.new('RGB', (224, 224), color='white')
    tensor = val_tf(dummy_img)
    assert tensor.max() <= 3.0 and tensor.min() >= -3.0

def test_device_availability():
    """Check if the configured device is valid."""
    assert CONFIG["device"] in ["cuda", "cpu", "mps"]

def test_folder_presence():
    """Final check on directory alignment."""
    assert os.path.exists("train"), "Train folder missing!"
    assert os.path.exists("test"), "Test folder missing!"

def test_batch_forward_pass():
    """Test model logic with a mock batch."""
    model = build_resnet50()
    x = torch.randn(2, 3, 224, 224).to(CONFIG["device"])
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 1)

def test_csv_not_empty():
    """Ensure the manifest actually contains data."""
    df = pd.read_csv("train_label.csv")
    assert len(df) > 0
