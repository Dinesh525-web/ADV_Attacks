import torch

def extract_embedding(tokenizer, model, text):
    """Extracts token embeddings from Bio-Medical-Llama-3."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract last hidden state embeddings
    embeddings = outputs.hidden_states[-1].squeeze(0)
    return embeddings
