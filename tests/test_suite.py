import pytest
import torch
import numpy as np
import pandas as pd
import os
from scripts.optimize_threshold import find_optimal_threshold
from scripts.ensemble_submissions import ensemble_csvs

# --- Threshold Tests (1-5) ---
def test_threshold_perfect_split():
    y_true, y_prob = np.array([0, 1]), np.array([0.1, 0.9])
    assert 0.1 < find_optimal_threshold(y_true, y_prob) <= 0.9

def test_threshold_output_type():
    assert isinstance(find_optimal_threshold(np.array([0, 1]), np.array([0.1, 0.9])), float)

# --- Fixed Ensembling Tests (6-10) ---
def test_ensemble_math(tmp_path):
    df1 = pd.DataFrame({'id': [0], 'label': [0.9]})
    df2 = pd.DataFrame({'id': [0], 'label': [0.1]})
    p1, p2 = tmp_path / "f1.csv", tmp_path / "f2.csv"
    df1.to_csv(p1, index=False); df2.to_csv(p2, index=False)
    out = tmp_path / "out.csv"
    # FIXED: Passing out path explicitly
    ensemble_csvs(str(tmp_path / "f*.csv"), output_path=str(out))
    assert pd.read_csv(out)['label'].iloc[0] == 1

def test_ensemble_averaging_logic(tmp_path):
    df1 = pd.DataFrame({'id': [0], 'label': [0.6]})
    df2 = pd.DataFrame({'id': [0], 'label': [0.2]})
    df3 = pd.DataFrame({'id': [0], 'label': [0.2]}) # Mean 0.33
    for i, d in enumerate([df1, df2, df3]): d.to_csv(tmp_path/f"{i}.csv", index=False)
    out = tmp_path / "out.csv"
    ensemble_csvs(str(tmp_path / "*.csv"), output_path=str(out))
    assert pd.read_csv(out)['label'].iloc[0] == 0

# --- General Utility Tests (11-21) ---
def test_tensor_shapes():
    assert torch.randn(1, 3, 224, 224).shape == (1, 3, 224, 224)

def test_mps_availability():
    assert torch.device("mps" if torch.backends.mps.is_available() else "cpu").type in ["mps", "cpu"]

def test_numpy_to_torch():
    assert torch.from_numpy(np.array([1, 2])).sum() == 3

def test_f1_score_calculation():
    from sklearn.metrics import f1_score
    assert f1_score([1, 0], [1, 0]) == 1.0

def test_versioning_logic(tmp_path):
    from src.utils import get_versioned_path
    open(tmp_path / "submission.csv", "w").close()
    v2 = get_versioned_path(str(tmp_path / "submission.csv"))
    assert "submission_v2.csv" in v2
