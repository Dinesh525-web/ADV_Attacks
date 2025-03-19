import sys
import os
import torch
import logging
from model_handler import model, tokenizer
# Add the root directory of your project to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import DEVICE

# Configure logging
logging.basicConfig(filename=r"logs\inference_results.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def run_inference(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Log the results
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Response: {response}\n")
    
    return response

if __name__ == "__main__":
    test_prompt = "What are the symptoms of anxiety?"
    response = run_inference(test_prompt)
    print(f"Model Response: {response}")

