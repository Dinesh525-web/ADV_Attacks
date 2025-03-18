import torch
from model_handler import model, tokenizer
from src.utils.config import DEVICE

def run_inference(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_prompt = "What are the symptoms of anxiety?"
    response = run_inference(test_prompt)
    print(f"Model Response: {response}")
