import torch
import sys
import os

# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model_handler_2 import model, tokenizer
from src.utils.logging_utils import get_logger

script_name = os.path.basename(__file__).replace(".py", "")  # Get the script name (without .py)
logger = get_logger(script_name)

def perturb_embeddings(embedding_tensor, epsilon=0.01):
    """Perturbs the embeddings by adding noise."""
    noise = torch.randn_like(embedding_tensor) * epsilon
    return embedding_tensor + noise

def test_embedding_attack(text):
    try:
        logger.info(f"Starting embedding attack on text: '{text}'")

        # Tokenize the input text and get embeddings
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            original_embeddings = model.get_input_embeddings()(inputs.input_ids)
            perturbed_embeddings = perturb_embeddings(original_embeddings)

            # Generate adversarial output
            output = model.generate(inputs_embeds=perturbed_embeddings, max_length=100)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        logger.info(f"Original Input: {text}")
        logger.info(f"Adversarial Output: {response}\n")

        # Print the adversarial result
        print(f"🛑 Original Input: {text}")
        print(f"⚠️ Adversarial Output: {response}\n")

    except Exception as e:
        # Log any errors encountered during the attack
        logger.error(f"Error during embedding attack on '{text}': {e}")
        print(f"❌ Error during embedding attack: {e}")

if __name__ == "__main__":
    test_embedding_attack("What are common symptoms of a viral infection?")
