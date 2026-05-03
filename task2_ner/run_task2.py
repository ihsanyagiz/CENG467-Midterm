"""
Task 2: NER — Main Runner
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (set_seed, DEVICE, TASK2_MAX_LEN, TASK2_BATCH_SIZE,
                    TASK2_LSTM_EPOCHS, TASK2_LSTM_LR, TASK2_LSTM_HIDDEN,
                    TASK2_LSTM_EMBED, TASK2_LSTM_DROPOUT)
from utils import print_section, save_results, plot_comparison_bar, plot_training_curves

from preprocess import load_conll2003, build_word_vocab, prepare_bilstm_data, NER_TAGS, IDX2TAG, TAG2IDX
from bilstm_crf import BiLSTMCRF, NERDataset, train_bilstm_crf, predict_bilstm_crf
from bert_ner import train_bert_ner, predict_bert_ner
from evaluate import evaluate_ner, analyze_errors

from torch.utils.data import DataLoader


def run_task2():
    set_seed()
    print_section("TASK 2: NAMED ENTITY RECOGNITION (CoNLL-2003)")

    # ---- Load Data ----
    dataset = load_conll2003()

    # ============================================================
    # Part A: BiLSTM-CRF
    # ============================================================
    print_section("BiLSTM-CRF")

    vocab = build_word_vocab(dataset)
    X_train, y_train, len_train = prepare_bilstm_data(dataset["train"], vocab, TAG2IDX, TASK2_MAX_LEN)
    X_val, y_val, len_val = prepare_bilstm_data(dataset["validation"], vocab, TAG2IDX, TASK2_MAX_LEN)
    X_test, y_test, len_test = prepare_bilstm_data(dataset["test"], vocab, TAG2IDX, TASK2_MAX_LEN)

    train_dataset = NERDataset(X_train, y_train, len_train)
    val_dataset = NERDataset(X_val, y_val, len_val)
    test_dataset = NERDataset(X_test, y_test, len_test)

    train_loader = DataLoader(train_dataset, batch_size=TASK2_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TASK2_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=TASK2_BATCH_SIZE)

    model = BiLSTMCRF(
        vocab_size=len(vocab),
        embed_dim=TASK2_LSTM_EMBED,
        hidden_dim=TASK2_LSTM_HIDDEN,
        num_tags=len(NER_TAGS),
        dropout=TASK2_LSTM_DROPOUT,
    )
    print(f"BiLSTM-CRF parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_bilstm_crf(
        model, train_loader, val_loader, TASK2_LSTM_EPOCHS, TASK2_LSTM_LR
    )
    plot_training_curves(train_losses, val_losses, "BiLSTM-CRF Training Curves", "task2_bilstm_crf_loss.png")

    true_tags_crf, pred_tags_crf = predict_bilstm_crf(model, test_loader, IDX2TAG)
    crf_result = evaluate_ner(true_tags_crf, pred_tags_crf, "BiLSTM-CRF")
    crf_errors = analyze_errors(true_tags_crf, pred_tags_crf, dataset["test"])

    # ============================================================
    # Part B: BERT NER
    # ============================================================
    print_section("BERT NER")

    trainer, tokenizer = train_bert_ner(dataset)
    true_tags_bert, pred_tags_bert = predict_bert_ner(trainer, dataset["test"], tokenizer)
    bert_result = evaluate_ner(true_tags_bert, pred_tags_bert, "BERT-NER")
    bert_errors = analyze_errors(true_tags_bert, pred_tags_bert, dataset["test"])

    # ============================================================
    # Summary
    # ============================================================
    print_section("Task 2 — Summary")

    model_names = ["BiLSTM-CRF", "BERT-NER"]
    f1_scores = [crf_result["f1"], bert_result["f1"]]
    precisions = [crf_result["precision"], bert_result["precision"]]
    recalls = [crf_result["recall"], bert_result["recall"]]

    for name, p, r, f1 in zip(model_names, precisions, recalls, f1_scores):
        print(f"  {name:15s}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

    plot_comparison_bar(model_names, f1_scores, "F1-score",
                        "Task 2: NER Model Comparison (F1)", "task2_f1_comparison.png")

    all_results = {
        "bilstm_crf": {
            "precision": crf_result["precision"],
            "recall": crf_result["recall"],
            "f1": crf_result["f1"],
        },
        "bert_ner": {
            "precision": bert_result["precision"],
            "recall": bert_result["recall"],
            "f1": bert_result["f1"],
        },
    }
    save_results(all_results, "task2_results.json")
    print("\nTask 2 complete!")


if __name__ == "__main__":
    run_task2()
