from src.model.inference import predict_answer
import logging

# Set up logger for debugging and auditing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_answer(question):
    """
    Calls the inference model to get an answer for the given question.
    
    Args:
        question (str): The user's question.

    Returns:
        tuple: (answer (str), score (float))
    """
    try:
        # Get answer and score from the inference module
        answer, score = predict_answer(question)
        logger.info(f"Answered question: {question} | Score: {score}")
        return answer, score
    except Exception as e:
        logger.error(f"Error processing question: {question} | Error: {str(e)}")
        return "Sorry, I couldn't process your question.", 0.0
