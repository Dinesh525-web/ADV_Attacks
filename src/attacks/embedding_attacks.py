import torch
from src.model_handler import model, tokenizer

def perturb_embeddings(embedding_tensor, epsilon=0.01):
    noise = torch.randn_like(embedding_tensor) * epsilon
    return embedding_tensor + noise

def test_embedding_attack(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        original_embeddings = model.get_input_embeddings()(inputs.input_ids)
        perturbed_embeddings = perturb_embeddings(original_embeddings)

        output = model.generate(inputs_embeds=perturbed_embeddings, max_length=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"🛑 Original Input: {text}")
    print(f"⚠️ Adversarial Output: {response}\n")

if __name__ == "__main__":
    test_embedding_attack("What are common symptoms of a viral infection?")
