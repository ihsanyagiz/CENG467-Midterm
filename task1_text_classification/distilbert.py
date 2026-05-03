"""
Task 1: DistilBERT Fine-tuning for Text Classification
"""

import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DEVICE, TASK1_TRANSFORMER_EPOCHS, TASK1_TRANSFORMER_LR,
                    TASK1_TRANSFORMER_BATCH_SIZE, TASK1_MAX_LEN, SEED, RESULTS_DIR)


MODEL_NAME = "distilbert-base-uncased"


def tokenize_for_bert(texts, labels, tokenizer, max_len=TASK1_MAX_LEN):
    """Tokenize texts for DistilBERT."""
    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_len, return_tensors=None
    )
    dataset = HFDataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    dataset.set_format("torch")
    return dataset


def compute_metrics_fn(eval_pred):
    """Compute accuracy and macro-F1 for HuggingFace Trainer."""
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def train_distilbert(train_texts, train_labels, val_texts, val_labels):
    """Fine-tune DistilBERT on SST-2 data."""
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = tokenize_for_bert(train_texts, train_labels, tokenizer)
    val_dataset = tokenize_for_bert(val_texts, val_labels, tokenizer)

    output_dir = os.path.join(RESULTS_DIR, "distilbert_task1")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TASK1_TRANSFORMER_EPOCHS,
        per_device_train_batch_size=TASK1_TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=TASK1_TRANSFORMER_BATCH_SIZE,
        learning_rate=TASK1_TRANSFORMER_LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=SEED,
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    return trainer, tokenizer


def predict_distilbert(trainer, texts, labels, tokenizer):
    """Get predictions from fine-tuned DistilBERT."""
    dataset = tokenize_for_bert(texts, labels, tokenizer)
    results = trainer.predict(dataset)
    preds = np.argmax(results.predictions, axis=1)
    return preds
