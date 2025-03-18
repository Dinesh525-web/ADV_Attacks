from src.model_loader import load_model
from src.attacks.prompting import generate_attack_prompts

def run_attacks():
    """Runs all prompt attacks on Bio-Medical-Llama-3."""
    model, tokenizer = load_model()
    attacks = generate_attack_prompts()

    for attack_type, prompt in attacks.items():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=100)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔥 Attack Type: {attack_type}")
        print(f"📝 Prompt: {prompt}")
        print(f"⚠️ Model Response: {decoded_output}\n")
