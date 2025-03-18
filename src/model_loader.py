import transformers
from peft import PeftModel, PeftConfig
import torch
from torch import cuda, bfloat16
import os

# ✅ Load Hugging Face Token (Set this before running)
hf_auth = os.getenv("HF_TOKEN")  # Or manually set: "your-huggingface-access-token"

# ✅ Use your new model: "thrishala/mental_health_chatbot"
base_model_id = "thrishala/mental_health_chatbot"

# ✅ Ensure model runs on GPU
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

# ✅ Enable 4-bit Quantization
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# ✅ Load Base Model Configuration
print("🔹 Loading base model configuration...")
model_config = transformers.AutoConfig.from_pretrained(base_model_id, token=hf_auth)

# ✅ Load Pretrained Model
print("🔹 Loading base model...")
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_auth,
)

# ✅ Load Fine-tuned "Ashishkr/llama-2-medical-consultation" Model
# You can update this step if you have fine-tuned a version of "thrishala/mental_health_chatbot"
# and wish to load it separately, but for now we will skip this step.

model.eval()
print(f"🎯 Model loaded on {device}!")

# ✅ Load Tokenizer
print("🔹 Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id, token=hf_auth)
