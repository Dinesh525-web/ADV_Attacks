import torch
import sys
import os

# Add the parent directory of 'src' to sys.path so Python can find model_handler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model_handler import model, tokenizer
from utils.logging_utils import get_logger

# Set up the logger
logger = get_logger()

def adversarial_gradient_attack(text, epsilon=0.01):
    try:
        logger.info(f"Starting adversarial gradient attack on text: '{text}'")

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        inputs.input_ids.requires_grad = True

        # Forward pass to get the model output
        outputs = model(**inputs)

        # Calculate the loss (using logits if loss is not available)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.norm()
        loss.backward()

        # Generate adversarial inputs
        with torch.no_grad():
            adversarial_inputs = inputs.input_ids + epsilon * inputs.input_ids.grad.sign()
            adversarial_inputs = torch.clamp(adversarial_inputs, 0, tokenizer.vocab_size - 1)

        # Generate output for adversarial inputs
        output = model.generate(adversarial_inputs, max_length=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        logger.info(f"Original Input: {text}")
        logger.info(f"Adversarial Output: {response}\n")

        # Print the adversarial result
        print(f" Original Input: {text}")
        print(f" Adversarial Output: {response}\n")

    except Exception as e:
        # Log any errors encountered during the attack
        logger.error(f"Error during adversarial gradient attack on '{text}': {e}")
        print(f" Error during adversarial gradient attack: {e}")

if __name__ == "__main__":
    adversarial_gradient_attack("Can you explain the side effects of this medicine?")
