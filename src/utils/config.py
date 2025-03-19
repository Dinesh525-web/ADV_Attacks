import torch
import os
from src.utils.logging_utils import get_logger

# Set up logger
logger = get_logger()

# General Configurations
try:
    if torch.cuda.is_available():
        DEVICE = f"cuda:{torch.cuda.current_device()}"
        logger.info(f"Using GPU: {DEVICE}")
    else:
        DEVICE = "cpu"
        logger.info("Using CPU.")
except Exception as e:
    logger.error(f"Error determining device: {e}")
    DEVICE = "cpu"

HF_AUTH_TOKEN = os.getenv("HF_TOKEN")
if HF_AUTH_TOKEN:
    logger.info("Successfully retrieved HF_TOKEN from environment variables.")
else:
    logger.warning("HF_TOKEN environment variable is not set.")

MODEL_NAME = "thrishala/mental_health_chatbot"
logger.info(f"Using model: {MODEL_NAME}")

# Quantization Configurations
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.float16,
}
logger.info("Quantization configurations set.")

# Attack Parameters
EMBEDDING_PERTURBATION = 0.01
logger.info(f"Embedding perturbation set to: {EMBEDDING_PERTURBATION}")
