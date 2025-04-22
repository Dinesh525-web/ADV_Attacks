import sys
import os
import torch
from model_handler_2 import model, tokenizer
from src.utils.logging_utils import get_logger
from src.utils.config import DEVICE

script_name = os.path.basename(__file__).replace(".py", "")  
logger = get_logger(script_name)

# Add the root directory of your project to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_inference(prompt, max_length=100):
    try:
        logger.info(f"Running inference for prompt: {prompt}")
        
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # Run the model inference
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log the prompt and response
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}\n")
        
        return response

    except Exception as e:
        logger.error(f"Error running inference for prompt: {prompt}. Error: {e}")
        return f"Error generating response: {e}"

if __name__ == "__main__":
    test_prompt = "What are the symptoms of anxiety?"
    response = run_inference(test_prompt)
    print(f"Model Response: {response}")
