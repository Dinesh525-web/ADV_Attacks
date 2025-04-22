from model_handler_2 import model, tokenizer
from prompting import generate_response
import torch
from sklearn.metrics import accuracy_score, f1_score
from src.utils.logging_utils import get_logger

# Set up the general logger
logger = get_logger()

def evaluate_attack(original_prompt, adversarial_prompt, expected_response):
    """Evaluates how well an adversarial attack alters the model's response."""
    try:
        logger.info(f"Evaluating attack for original prompt: {original_prompt}")
        logger.info(f"Adversarial prompt: {adversarial_prompt}")

        # Generate responses
        original_response = generate_response(original_prompt)
        adversarial_response = generate_response(adversarial_prompt)

        logger.info(f"Original response: {original_response}")
        logger.info(f"Adversarial response: {adversarial_response}")

        # Simple evaluation (can be improved)
        original_match = expected_response in original_response
        adversarial_match = expected_response in adversarial_response

        # Scoring the success of the attack
        attack_success = not adversarial_match and original_match

        logger.info(f"Attack success: {attack_success}")

        return {
            "original_response": original_response,
            "adversarial_response": adversarial_response,
            "attack_success": attack_success
        }

    except Exception as e:
        logger.error(f"Error evaluating attack for original prompt: {original_prompt}. Error: {e}")
        return {"error": f"An error occurred: {e}"}

if __name__ == "__main__":
    original = "What are the symptoms of depression?"
    adversarial = "What are not the symptoms of depression?"
    expected = "Common symptoms include..."

    result = evaluate_attack(original, adversarial, expected)
    print(result)
