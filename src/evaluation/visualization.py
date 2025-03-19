import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils.logging_utils import get_logger

# Set up the logger
logger = get_logger()

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plots a confusion matrix for model evaluation."""
    try:
        logger.info("Starting to plot confusion matrix...")

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Log the true and predicted labels
        logger.info(f"True labels: {y_true}")
        logger.info(f"Predicted labels: {y_pred}")
        
        # Plotting the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        logger.info("Confusion matrix plotted successfully.")

    except Exception as e:
        # Log any errors encountered during the plotting process
        logger.error(f"Error plotting confusion matrix: {e}")

if __name__ == "__main__":
    # Example test case
    y_true = [0, 1, 1, 0, 2]
    y_pred = [0, 1, 0, 0, 2]
    labels = ["Negative", "Neutral", "Positive"]
    
    # Log the start of the process
    logger.info("Starting confusion matrix test...")

    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels)
    
    logger.info("Test completed.")
