import sys
import os

# Add the root directory of your project to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import AutoModel, AutoTokenizer
from src.utils.logging_utils import get_logger
from utils.config import MODEL_NAME, DEVICE

# Set up the logger
logger = get_logger()

# Load tokenizer and model
logger.info("Loading model for embeddings...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

def get_embedding(text):
    """Extracts embeddings from the given text."""
    try:
        logger.info(f"Extracting embedding for text: {text[:50]}...")  # Log a snippet of the text
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate the mean embedding across the tokens
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        # Log the shape of the generated embedding
        logger.info(f"Generated embedding shape: {embedding.shape}")
        return embedding
    
    except Exception as e:
        logger.error(f"Error extracting embedding for text: {text}. Error: {e}")
        return None

if __name__ == "__main__":
    sample_text = "Depression is a serious mental health condition."
    embedding = get_embedding(sample_text)
    
    
    if embedding is not None:
        print(f"Generated embedding shape: {embedding.shape}")
    else:
        print("Error occurred while generating embedding.")
