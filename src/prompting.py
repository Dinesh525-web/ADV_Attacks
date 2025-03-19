from model_handler import model, tokenizer
import torch
from src.utils.logging_utils import get_logger

# Set up logger
logger = get_logger()

def generate_response(prompt, max_length=150):
    """Generates a response from the model given a prompt."""
    logger.info(f"Generating response for prompt: {prompt}")

    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response without tracking gradients
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and return the generated response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response for prompt: {prompt}. Error: {e}")
        return f"Error generating response: {e}"

if __name__ == "__main__":
    sample_prompt = "What are the symptoms of anxiety?"
    response = generate_response(sample_prompt)
    print("Response:", response)
