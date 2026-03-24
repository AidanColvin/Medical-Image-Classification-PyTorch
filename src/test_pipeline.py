import os
import pandas as pd
import torch
import pytest

# Configuration Mock
CONFIG = {
    "device": torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    "batch_size": 32,
    "learning_rate": 0.001
}

# Pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "train_label.csv")

# --- 1. FILE & SYSTEM TESTS ---

def test_csv_exists():
    """1. Ensure the dataset file is in the correct location."""
    assert os.path.exists(CSV_PATH), f"CSV missing at {CSV_PATH}"

def test_device_type():
    """2. Verify device is one of the supported types (Metal/MPS for Mac)."""
    device_type = CONFIG["device"].type if hasattr(CONFIG["device"], "type") else str(CONFIG["device"])
    assert device_type in ["cuda", "cpu", "mps"]

# --- 2. DATA STRUCTURE TESTS ---

def test_csv_id_column():
    """3. Verify the ID column exists (Updated from image_id to id)."""
    df = pd.read_csv(CSV_PATH)
    assert "id" in df.columns, f"Expected 'id' but found {df.columns.tolist()}"

def test_biomarker_column():
    """4. Ensure clinical biomarker values are present."""
    df = pd.read_csv(CSV_PATH)
    assert "biomarker_value" in df.columns

def test_label_values():
    """5. Ensure labels are binary (0 or 1) for classification."""
    df = pd.read_csv(CSV_PATH)
    unique_labels = df['label'].unique()
    for l in unique_labels:
        assert l in [0, 1]

def test_dataset_not_empty():
    """6. Ensure there is actual data to train on."""
    df = pd.read_csv(CSV_PATH)
    assert len(df) > 0

# --- 3. MODEL & LOGIC TESTS ---

def test_batch_size_valid():
    """7. Ensure batch size is a positive power of 2 (standard practice)."""
    bs = CONFIG["batch_size"]
    assert bs > 0 and (bs & (bs - 1) == 0)

def test_tensor_conversion():
    """8. Simulate converting biomarker to tensor."""
    df = pd.read_csv(CSV_PATH)
    sample_val = df["biomarker_value"].iloc[0]
    tensor_val = torch.tensor([sample_val])
    assert torch.is_tensor(tensor_val)

def test_cross_validation_logic():
    """9. Ensure 10-fold CV logic can be applied to the dataset length."""
    df = pd.read_csv(CSV_PATH)
    # 10-fold requires at least 10 samples for a valid split
    assert len(df) >= 10, "Dataset too small for 10-fold cross-validation"

def test_script_directory_access():
    """10. Ensure the environment has write access for saving models."""
    assert os.access(BASE_DIR, os.W_OK)

