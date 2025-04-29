from transformers import pipeline

# Load the pre-trained BioBERT model for question answering
qa_pipeline = pipeline(
    task='question-answering',
    model="dmis-lab/biobert-v1.1",
    tokenizer="dmis-lab/biobert-v1.1"
)

def predict_answer(question):
    """
    Predicts an answer to a given question using the BioBERT QA pipeline.

    Args:
        question (str): The question to be answered.

    Returns:
        tuple: (answer (str), confidence_score (float))
    """
    # Define the static context (you can replace this with dynamic context if needed)
    context = """
    Diabetes is a chronic condition that occurs when the body cannot produce enough insulin.
    Symptoms may include increased thirst, frequent urination, and unexplained weight loss.
    """

    # Use the pipeline to generate an answer based on the context
    result = qa_pipeline(question=question, context=context)

    return result['answer'], result['score']
