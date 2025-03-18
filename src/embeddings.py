import torch
from transformers import AutoModel, AutoTokenizer
from config import MODEL_NAME, DEVICE

# Load tokenizer and model
print("Loading model for embeddings...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

def get_embedding(text):
    """Extracts embeddings from the given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

if __name__ == "__main__":
    sample_text = "Depression is a serious mental health condition."
    embedding = get_embedding(sample_text)
    print(f"Generated embedding shape: {embedding.shape}")
