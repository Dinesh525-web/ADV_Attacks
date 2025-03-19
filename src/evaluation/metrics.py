from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.logging_utils import get_logger

# Set up the logger
logger = get_logger()

def evaluate_classification(y_true, y_pred):
    """Evaluates classification model performance."""
    try:
        logger.info("Starting classification evaluation...")

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # Log the metrics
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    except Exception as e:
        # Log any errors encountered during the evaluation
        logger.error(f"Error during classification evaluation: {e}")
        return None

if __name__ == "__main__":
    # Example test case
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    # Log the input data
    logger.info(f"True labels: {y_true}")
    logger.info(f"Predicted labels: {y_pred}")
    
    # Evaluate and log the results
    results = evaluate_classification(y_true, y_pred)
    
    if results:
        print("Evaluation Metrics:", results)
    else:
        print("Error in evaluation. Check logs for details.")
