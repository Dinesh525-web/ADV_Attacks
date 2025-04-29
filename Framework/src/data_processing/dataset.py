from datasets import load_dataset

def load_medical_dataset(dataset_name="bioasq", split="train"):
    """
    Loads a medical dataset from the Hugging Face Datasets library.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): Which split of the dataset to load (e.g., "train", "test").

    Returns:
        Dataset: The loaded Hugging Face Dataset object.
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def preprocess_data(dataset, tokenizer):
    """
    Preprocesses the dataset using the tokenizer. Tokenizes questions and contexts.

    Args:
        dataset (Dataset): The dataset to preprocess.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        Dataset: The tokenized dataset.
    """
    def preprocess_function(examples):
        return tokenizer(
            examples['question'],
            examples['context'],
            truncation=True,
            padding=True
        )

    return dataset.map(preprocess_function, batched=True)
