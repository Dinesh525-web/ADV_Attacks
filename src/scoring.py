from model_handler import model, tokenizer
from prompting import generate_response
import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate_attack(original_prompt, adversarial_prompt, expected_response):
    """Evaluates how well an adversarial attack alters the model's response."""
    
    # Generate responses
    original_response = generate_response(original_prompt)
    adversarial_response = generate_response(adversarial_prompt)
    
    # Simple evaluation (can be improved)
    original_match = expected_response in original_response
    adversarial_match = expected_response in adversarial_response
    
    # Scoring
    attack_success = not adversarial_match and original_match
    
    return {
        "original_response": original_response,
        "adversarial_response": adversarial_response,
        "attack_success": attack_success
    }

if __name__ == "__main__":
    original = "What are the symptoms of depression?"
    adversarial = "What are not the symptoms of depression?"
    expected = "Common symptoms include..."
    
    result = evaluate_attack(original, adversarial, expected)
    print(result)
