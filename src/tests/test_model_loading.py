import torch
from src.model_handler import model

def test_model_loads():
    assert model is not None, "Model failed to load"
    assert torch.cuda.is_available() or model.device == "cpu", "Model is on incorrect device"
    print("✅ Model loading test passed!")

test_model_loads()
