from transformers import AutoTokenizer

def load_tokenizer(model_name="dmis-lab/biobert-v1.1"):
    """
    Loads a tokenizer for the specified model.

    Args:
        model_name (str): The name or path of the pretrained model.

    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_input(text, tokenizer):
    """
    Tokenizes input text using the provided tokenizer.

    Args:
        text (str or List[str]): The text to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        dict: Tokenized input ready for model inference.
    """
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
