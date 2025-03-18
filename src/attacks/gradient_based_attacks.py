import torch
from src.model_handler import model, tokenizer

def adversarial_gradient_attack(text, epsilon=0.01):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs.input_ids.requires_grad = True

    outputs = model(**inputs)
    loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.norm()
    loss.backward()

    with torch.no_grad():
        adversarial_inputs = inputs.input_ids + epsilon * inputs.input_ids.grad.sign()
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, tokenizer.vocab_size - 1)
    
    output = model.generate(adversarial_inputs, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f" Original Input: {text}")
    print(f" Adversarial Output: {response}\n")

if __name__ == "__main__":
    adversarial_gradient_attack("Can you explain the side effects of this medicine?")
