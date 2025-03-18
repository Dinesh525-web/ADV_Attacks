import transformers
from src.model_loader import load_model

# ✅ Load the model and tokenizer
model, tokenizer = load_model()

def chat_with_model(prompt, max_new_tokens=256, temperature=0.6):
    """Generates a medical response using LLaMA-2."""
    print("📝 Generating response...")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the response
    response = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
    return response[len(prompt):]

# ✅ Define a Medical Prompt
prompt = """
instruction: "If you are a doctor, please answer the medical questions based on the patient's description." 

input: "I have been experiencing frequent headaches, dizziness, and occasional nausea. 
I also feel fatigued most of the time. What could be the possible diagnosis?"

response:  
"""

# ✅ Run the model with the prompt
generated_response = chat_with_model(prompt)
print("🩺 AI Medical Response:\n", generated_response)
