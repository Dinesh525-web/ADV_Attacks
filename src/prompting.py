from model_handler import model, tokenizer
import torch

def generate_response(prompt, max_length=150):
    """Generates a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_prompt = "What are the symptoms of anxiety?"
    response = generate_response(sample_prompt)
    print("Response:", response)