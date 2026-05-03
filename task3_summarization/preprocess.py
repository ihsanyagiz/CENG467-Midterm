"""
Task 3: Text Summarization — Preprocessing
CNN/DailyMail dataset subset loader.
"""

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from datasets import load_dataset

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TASK3_SUBSET_TRAIN, TASK3_SUBSET_VAL, TASK3_SUBSET_TEST, SEED


def load_cnn_dailymail_subset():
    """Load a subset of CNN/DailyMail dataset for computational feasibility."""
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train = dataset["train"].shuffle(seed=SEED).select(range(TASK3_SUBSET_TRAIN))
    val = dataset["validation"].shuffle(seed=SEED).select(range(TASK3_SUBSET_VAL))
    test = dataset["test"].shuffle(seed=SEED).select(range(TASK3_SUBSET_TEST))

    print(f"CNN/DailyMail subset — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def get_articles_and_highlights(dataset_split):
    """Extract articles and reference summaries."""
    articles = dataset_split["article"]
    highlights = dataset_split["highlights"]
    return articles, highlights
