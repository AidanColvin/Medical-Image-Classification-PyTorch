import torch

def test_pytorch_visibility():
    # Confirming the environment we just installed is working
    assert torch.__version__ is not None
