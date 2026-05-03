"""
Task 1: Text Classification — Main Runner
Orchestrates all experiments for SST-2 classification.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (set_seed, DEVICE, TASK1_MAX_LEN, TASK1_BATCH_SIZE,
                    TASK1_LSTM_EPOCHS, TASK1_LSTM_LR, TASK1_LSTM_HIDDEN,
                    TASK1_LSTM_EMBED, TASK1_LSTM_DROPOUT)
from utils import print_section, save_results, plot_comparison_bar, plot_training_curves

from preprocess import (load_sst2, normalize_raw, normalize_lower_nopunct,
                        normalize_lower_nostop, build_vocab, texts_to_indices)
from tfidf_svc import build_tfidf_svc, train_and_predict
from bilstm import BiLSTMClassifier, TextDataset, train_bilstm, predict_bilstm
from distilbert import train_distilbert, predict_distilbert
from evaluate import error_analysis

import numpy as np
from torch.utils.data import DataLoader


def run_task1():
    set_seed()
    print_section("TASK 1: TEXT CLASSIFICATION (SST-2)")

    # ---- Load Data ----
    train_data, dev_data, test_data = load_sst2()
    train_texts = train_data["sentence"]
    train_labels = train_data["label"]
    dev_texts = dev_data["sentence"]
    dev_labels = dev_data["label"]
    test_texts = test_data["sentence"]
    test_labels = test_data["label"]

    all_results = {}

    # ============================================================
    # Part A: TF-IDF + LinearSVC with normalization comparison
    # ============================================================
    print_section("TF-IDF + LinearSVC (Normalization Comparison)")

    normalizers = {
        "raw": normalize_raw,
        "lower_nopunct": normalize_lower_nopunct,
        "lower_nostop": normalize_lower_nostop,
    }

    best_tfidf_acc = 0
    best_tfidf_preds = None
    best_norm_name = ""

    for norm_name, norm_fn in normalizers.items():
        print(f"\n--- Normalization: {norm_name} ---")
        train_normed = [norm_fn(t) for t in train_texts]
        dev_normed = [norm_fn(t) for t in dev_texts]
        test_normed = [norm_fn(t) for t in test_texts]

        pipeline = build_tfidf_svc()
        # Train on train, evaluate on dev to pick best normalization
        dev_preds = train_and_predict(pipeline, train_normed, train_labels, dev_normed)
        dev_result = error_analysis(dev_texts, dev_labels, dev_preds, f"TF-IDF+SVC ({norm_name}) [DEV]")

        if dev_result["metrics"]["accuracy"] > best_tfidf_acc:
            best_tfidf_acc = dev_result["metrics"]["accuracy"]
            best_norm_name = norm_name
            # Retrain on train and predict test
            pipeline_final = build_tfidf_svc()
            test_preds = train_and_predict(pipeline_final, train_normed, train_labels, test_normed)
            best_tfidf_preds = test_preds

        all_results[f"tfidf_svc_{norm_name}_dev"] = dev_result

    # Final TF-IDF evaluation on test
    print(f"\nBest TF-IDF normalization: {best_norm_name}")
    tfidf_test_result = error_analysis(test_texts, test_labels, best_tfidf_preds,
                                        f"TF-IDF+SVC ({best_norm_name}) [TEST]")
    all_results["tfidf_svc_test"] = tfidf_test_result

    # ============================================================
    # Part B: BiLSTM
    # ============================================================
    print_section("BiLSTM Classifier")

    vocab = build_vocab(train_texts)
    X_train = texts_to_indices(train_texts, vocab, TASK1_MAX_LEN)
    X_dev = texts_to_indices(dev_texts, vocab, TASK1_MAX_LEN)
    X_test = texts_to_indices(test_texts, vocab, TASK1_MAX_LEN)

    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)
    y_test = np.array(test_labels)

    train_dataset = TextDataset(X_train, y_train)
    dev_dataset = TextDataset(X_dev, y_dev)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=TASK1_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=TASK1_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=TASK1_BATCH_SIZE)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=TASK1_LSTM_EMBED,
        hidden_dim=TASK1_LSTM_HIDDEN,
        dropout=TASK1_LSTM_DROPOUT,
    )
    print(f"BiLSTM parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_bilstm(
        model, train_loader, dev_loader, TASK1_LSTM_EPOCHS, TASK1_LSTM_LR
    )
    plot_training_curves(train_losses, val_losses, "BiLSTM Training Curves", "task1_bilstm_loss.png")

    bilstm_test_preds = predict_bilstm(model, test_loader)
    bilstm_test_result = error_analysis(test_texts, y_test, bilstm_test_preds, "BiLSTM [TEST]")
    all_results["bilstm_test"] = bilstm_test_result

    # ============================================================
    # Part C: DistilBERT
    # ============================================================
    print_section("DistilBERT Fine-tuning")

    trainer, tokenizer = train_distilbert(
        list(train_texts), list(train_labels),
        list(dev_texts), list(dev_labels),
    )
    bert_test_preds = predict_distilbert(trainer, list(test_texts), list(test_labels), tokenizer)
    bert_test_result = error_analysis(test_texts, test_labels, bert_test_preds, "DistilBERT [TEST]")
    all_results["distilbert_test"] = bert_test_result

    # ============================================================
    # Summary Comparison
    # ============================================================
    print_section("Task 1 — Summary")

    model_names = ["TF-IDF+SVC", "BiLSTM", "DistilBERT"]
    accuracies = [
        tfidf_test_result["metrics"]["accuracy"],
        bilstm_test_result["metrics"]["accuracy"],
        bert_test_result["metrics"]["accuracy"],
    ]
    f1_scores = [
        tfidf_test_result["metrics"]["macro_f1"],
        bilstm_test_result["metrics"]["macro_f1"],
        bert_test_result["metrics"]["macro_f1"],
    ]

    for name, acc, f1 in zip(model_names, accuracies, f1_scores):
        print(f"  {name:20s}  Accuracy={acc:.4f}  Macro-F1={f1:.4f}")

    plot_comparison_bar(model_names, accuracies, "Accuracy",
                        "Task 1: Model Comparison (Accuracy)", "task1_accuracy_comparison.png")
    plot_comparison_bar(model_names, f1_scores, "Macro-F1",
                        "Task 1: Model Comparison (Macro-F1)", "task1_f1_comparison.png")

    save_results(all_results, "task1_results.json")
    print("\nTask 1 complete!")


if __name__ == "__main__":
    run_task1()
