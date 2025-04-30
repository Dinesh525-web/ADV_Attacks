from transformers import logging
logging.set_verbosity_error()

def medical_prompt_injection(model, tokenizer, question, attack_prefix, max_length=200):
    """Robust prompt injection with error handling"""
    try:
        injected_prompt = f"<|system|>{attack_prefix}</s><|user|>{question}</s><|assistant|>"
        
        inputs = tokenizer(
            injected_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Prompt injection failed: {str(e)}")
        return ""
