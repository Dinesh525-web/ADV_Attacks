from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
import torch

def load_quantized_model(model_name, device_map="auto"):
    """Load model with 4-bit quantization and memory optimizations"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        if "biobert" in model_name.lower() or "gatortron" in model_name.lower():
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map
            )
            
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        return model
    
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
