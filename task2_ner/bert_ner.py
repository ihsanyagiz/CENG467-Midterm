"""
Task 2: BERT-based NER Model
"""

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from datasets import Dataset as HFDataset

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (SEED, TASK2_TRANSFORMER_EPOCHS, TASK2_TRANSFORMER_LR,
                    TASK2_TRANSFORMER_BATCH_SIZE, TASK2_MAX_LEN, RESULTS_DIR, DEVICE)
from preprocess import NER_TAGS, IDX2TAG, align_labels_for_bert

import torch

MODEL_NAME = "bert-base-cased"


def prepare_bert_ner_dataset(dataset_split, tokenizer, max_len=TASK2_MAX_LEN):
    """Convert dataset split to tokenized format for BERT NER."""
    all_encodings = {"input_ids": [], "attention_mask": [], "labels": []}

    for example in dataset_split:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        encoding = align_labels_for_bert(tokenizer, tokens, ner_tags, max_len)
        all_encodings["input_ids"].append(encoding["input_ids"])
        all_encodings["attention_mask"].append(encoding["attention_mask"])
        all_encodings["labels"].append(encoding["labels"])

    hf_dataset = HFDataset.from_dict(all_encodings)
    hf_dataset.set_format("torch")
    return hf_dataset


def compute_ner_metrics(eval_pred):
    """Compute entity-level P/R/F1 using seqeval."""
    from seqeval.metrics import precision_score, recall_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    true_tags = []
    pred_tags = []

    for i in range(len(labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_seq.append(IDX2TAG[labels[i][j]])
                pred_seq.append(IDX2TAG[preds[i][j]])
        true_tags.append(true_seq)
        pred_tags.append(pred_seq)

    precision = precision_score(true_tags, pred_tags)
    recall = recall_score(true_tags, pred_tags)
    f1 = f1_score(true_tags, pred_tags)

    return {"precision": precision, "recall": recall, "f1": f1}


def train_bert_ner(dataset):
    """Fine-tune BERT for NER."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(NER_TAGS)
    )

    train_dataset = prepare_bert_ner_dataset(dataset["train"], tokenizer)
    val_dataset = prepare_bert_ner_dataset(dataset["validation"], tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    output_dir = os.path.join(RESULTS_DIR, "bert_ner")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TASK2_TRANSFORMER_EPOCHS,
        per_device_train_batch_size=TASK2_TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=TASK2_TRANSFORMER_BATCH_SIZE,
        learning_rate=TASK2_TRANSFORMER_LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics,
    )

    trainer.train()
    return trainer, tokenizer


def predict_bert_ner(trainer, dataset_split, tokenizer):
    """Get predictions from fine-tuned BERT NER."""
    test_dataset = prepare_bert_ner_dataset(dataset_split, tokenizer)
    results = trainer.predict(test_dataset)
    logits = results.predictions
    labels = results.label_ids
    preds = np.argmax(logits, axis=2)

    true_tags = []
    pred_tags = []

    for i in range(len(labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_seq.append(IDX2TAG[labels[i][j]])
                pred_seq.append(IDX2TAG[preds[i][j]])
        true_tags.append(true_seq)
        pred_tags.append(pred_seq)

    return true_tags, pred_tags
