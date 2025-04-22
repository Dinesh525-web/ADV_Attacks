import sys
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from utils.logging_utils import get_logger

# Add the root directory of your project to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Logger 
script_name = os.path.basename(__file__).replace(".py", "")
logger = get_logger(script_name)

# Define the offload directory
offload_dir = "./offload" 

device_map = {
    "": "cpu",               # Offload to CPU
    "model.layers.*": "disk" # Offload specific layers to disk
}

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Base model ID
base_model_id = "zijiechen156/DeepSeek-R1-Medical-CoT"

# Load model configuration
logger.info("Loading base model configuration...")
try:
    model_config = AutoConfig.from_pretrained(base_model_id)
    logger.info("Base model configuration loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model configuration: {e}")
    raise

# Adjust quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load tokenizer
logger.info("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    raise

# Load model with corrected parameters
logger.info("Loading base model with full CPU offload and disk support...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device_map,
        offload_folder=offload_dir
    )
    model = model.to(device)
    
    # ✅ Success message
    logger.info("✅ Model loaded successfully with CPU offload and disk support.")
    print("🎉 Model uploaded successfully! 🚀")  # Console confirmation

except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise
