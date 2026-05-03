"""
Task 1: Evaluation and Error Analysis
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_classification_metrics


def find_misclassified(texts, y_true, y_pred, n=5):
    """Find n misclassified examples with their true and predicted labels."""
    misclassified = []
    label_map = {0: "Negative", 1: "Positive"}
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            misclassified.append({
                "index": i,
                "text": texts[i],
                "true_label": label_map[int(y_true[i])],
                "predicted_label": label_map[int(y_pred[i])],
            })
        if len(misclassified) >= n:
            break
    return misclassified


def error_analysis(texts, y_true, y_pred, model_name):
    """Perform error analysis for a model."""
    metrics = compute_classification_metrics(y_true, y_pred)
    misclassified = find_misclassified(texts, y_true, y_pred, n=5)

    print(f"\n--- {model_name} Error Analysis ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"\nMisclassified Examples:")
    for i, ex in enumerate(misclassified):
        print(f"  [{i+1}] \"{ex['text'][:100]}...\"")
        print(f"       True: {ex['true_label']}, Predicted: {ex['predicted_label']}")

    return {
        "model": model_name,
        "metrics": {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
        "misclassified": misclassified,
    }
