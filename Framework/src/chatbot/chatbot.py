from src.model.inference import predict_answer
 import logging
 # Set up logger for debugging and auditing
 logger = logging.getLogger(__name__)
 def get_answer(question):
    try:
        # Get answer from the inference module
        answer, score = predict_answer(question)
        logger.info(f"Answered question: {question}")
        return answer, score
    except Exception as e:
        logger.error(f"Error while processing question: {question}. Error: 
{str(e)}")
        return "Sorry, I couldn't process your question.", 0.0