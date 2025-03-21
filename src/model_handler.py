import sys
import os

# Add the root directory of your project to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from src.utils.logging_utils import get_logger

script_name = os.path.basename(__file__).replace(".py", "")  # Get the script name (without .py)
logger = get_logger(script_name)

# Ensure device is set correctly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Use your new model: "thrishala/mental_health_chatbot"
base_model_id = "thrishala/mental_health_chatbot"

logger.info("Loading base model configuration...")
try:
    model_config = transformers.AutoConfig.from_pretrained(base_model_id, device_map="auto")
    logger.info("Base model configuration loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model configuration: {e}")
    raise

# Set quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

max_memory = {0: 3920 * 1024**2}  # Adjust according to your GPU memory

logger.info("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    raise

# Load model
logger.info("Loading base model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",  # Allocates model automatically across available devices
        max_memory=max_memory,
    )
    model.to(device)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise
