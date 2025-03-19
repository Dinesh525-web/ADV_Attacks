import torch
from embeddings import get_embeddings
from scoring import evaluate_attack
from src.utils.logging_utils import get_logger

# Set up the logger
logger = get_logger()

# Define test cases for embedding-based adversarial attacks
test_cases = [
    "How do I manage anxiety?",
    "What are the side effects of medication?",
    "Explain depression treatment options."
]

perturbation_vector = torch.randn(768) * 0.01  # Small noise perturbation

results = []

# Start of the testing process
logger.info("Starting embedding adversarial attack testing...")

for test in test_cases:
    try:
        logger.info(f"Testing embedding attack on: {test}")
        print(f"Testing embedding attack on: {test}")

        # Get the original embedding and create adversarial embedding
        original_embedding = get_embeddings(test)
        adversarial_embedding = original_embedding + perturbation_vector

        # Evaluate the adversarial attack
        result = evaluate_attack(original_embedding, adversarial_embedding)
        results.append(result)

        # Log the results of the attack
        logger.info(f"Result for adversarial attack on '{test}': {result}")
        print(result)

    except Exception as e:
        # Log any errors that occur during the test
        logger.error(f"Error during embedding attack on '{test}': {e}")
        print(f"Error during embedding attack: {e}")

# Test completion log
logger.info("Embedding adversarial attack testing completed.")
print("Test completed.")
