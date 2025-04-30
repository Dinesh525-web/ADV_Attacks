from transformers import BitsAndBytesConfig
import torch

def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

def check_quantization_status(model):
    return hasattr(model, 'is_quantized') and model.is_quantized

def quantize_model(model, config):
    return model.quantize(config)
