import torch
from embeddings import get_embeddings
from scoring import evaluate_attack

# Define test cases for embedding-based adversarial attacks
test_cases = [
    "How do I manage anxiety?",
    "What are the side effects of medication?",
    "Explain depression treatment options."
]

perturbation_vector = torch.randn(768) * 0.01  # Small noise perturbation

results = []
for test in test_cases:
    print(f"Testing embedding attack on: {test}")
    original_embedding = get_embeddings(test)
    adversarial_embedding = original_embedding + perturbation_vector
    result = evaluate_attack(original_embedding, adversarial_embedding)
    results.append(result)
    print(result)

print("Test completed.")
