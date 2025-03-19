import torch
from src.model_handler import model
from src.utils.logging_utils import get_logger

# Set up the logger
logger = get_logger()

def test_model_loads():
    try:
        # Test if model is loaded correctly
        logger.info("Starting model load test...")

        assert model is not None, "Model failed to load"
        logger.info("Model loaded successfully.")

        # Test if the model is on the correct device
        assert torch.cuda.is_available() or model.device == "cpu", "Model is on incorrect device"
        logger.info(f"Model is on the correct device: {model.device}")

        print("✅ Model loading test passed!")
        logger.info("Model loading test passed!")

    except AssertionError as e:
        logger.error(f"Assertion failed: {e}")
        print(f"❌ Model loading test failed: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error occurred during model loading test: {e}")
        print(f"❌ Unexpected error occurred during model loading test: {e}")

if __name__ == "__main__":
    test_model_loads()
