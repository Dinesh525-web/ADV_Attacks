import torch
import os

# General Configurations
DEVICE = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
HF_AUTH_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "thrishala/mental_health_chatbot"

# Quantization Configurations
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
}

# Attack Parameters
EMBEDDING_PERTURBATION = 0.01