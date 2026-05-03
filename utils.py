"""
CENG 467 - NLU&G Take-Home Midterm
Shared Utilities
Student ID: 300201071
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import FIGURES_DIR, RESULTS_DIR


def compute_classification_metrics(y_true, y_pred):
    """Compute Accuracy and Macro-F1 for classification tasks."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)
    return {"accuracy": acc, "macro_f1": f1, "report": report}


def save_results(results, filename):
    """Save results dictionary to JSON."""
    path = os.path.join(RESULTS_DIR, filename)
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to {path}")


def plot_training_curves(train_losses, val_losses, title, filename):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    if val_losses:
        plt.plot(val_losses, label="Val Loss", color="#FF5722", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")


def plot_comparison_bar(model_names, metric_values, metric_name, title, filename):
    """Create a bar chart comparing models on a given metric."""
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, metric_values, color=colors[: len(model_names)], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    plt.ylabel(metric_name)
    plt.title(title)
    plt.ylim(0, max(metric_values) * 1.15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
