import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plots a confusion matrix for model evaluation."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    # Example test case
    y_true = [0, 1, 1, 0, 2]
    y_pred = [0, 1, 0, 0, 2]
    labels = ["Negative", "Neutral", "Positive"]
    plot_confusion_matrix(y_true, y_pred, labels)
