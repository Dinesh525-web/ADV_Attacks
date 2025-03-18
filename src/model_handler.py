import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
# Ensure device is set correctly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use your new model: "thrishala/mental_health_chatbot"
base_model_id = "thrishala/mental_health_chatbot"

print("Loading base model configuration...")
model_config = transformers.AutoConfig.from_pretrained(base_model_id,device_map="auto")

# Set quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

max_memory = {0: 3920 * 1024**2}  # Adjust according to your GPU memory

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",  # Allocates model automatically across available devices
    max_memory=max_memory,
)

model.to(device)
print("Model loaded successfully.")
