import transformers
from peft import PeftModel, PeftConfig
import torch
from torch import cuda, bfloat16
import os

# Load Hugging Face Token (Set this before running)
hf_auth = os.getenv("HF_TOKEN")  # Or manually set: "your-huggingface-access-token"

# Use your new model: "thrishala/mental_health_chatbot"
base_model_id = "thrishala/mental_health_chatbot"

# Ensure model runs on GPU
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

# Enable 4-bit Quantization (Optimized for 4GB GPU)
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16
)


# Load Base Model Configuration
print("🔹 Loading base model configuration...")
model_config = transformers.AutoConfig.from_pretrained(base_model_id, token=hf_auth)

# Load Pretrained Model (Optimized for Low VRAM)
print("🔹 Loading base model...")
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU Offloading
    token=hf_auth,
)


# Enable Model Evaluation Mode
model.eval()
print(f" Model loaded successfully on {device}!")

# Load Tokenizer
print("🔹 Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id, token=hf_auth)

# Optional: Further Optimize Inference Speed (if CUDA 11.8+ is available)
if device.startswith("cuda"):
    model = torch.compile(model)
    print("🚀 Model compiled for faster inference!")
