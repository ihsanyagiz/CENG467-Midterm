"""
Task 1: Text Classification — Preprocessing Pipeline
Compares word-level vs. subword tokenization and normalization strategies.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import load_dataset

# Ensure NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def load_sst2():
    """Load SST-2 dataset and create train/dev/test splits.
    
    Since SST-2 test labels are hidden in GLUE, we:
      - Split original train (90/10) → train / dev
      - Use official validation → test
    """
    dataset = load_dataset("glue", "sst2")
    
    # Official validation becomes our test set
    test_data = dataset["validation"]
    
    # Split training set: 90% train, 10% dev
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = train_val["train"]
    dev_data = train_val["test"]
    
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    return train_data, dev_data, test_data


# ============================================================
# Normalization Strategies
# ============================================================

def normalize_raw(text):
    """Strategy A: No normalization (raw text)."""
    return text


def normalize_lower_nopunct(text):
    """Strategy B: Lowercase + punctuation removal."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_lower_nostop(text):
    """Strategy C: Lowercase + stopword removal."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


# ============================================================
# Tokenization Strategies
# ============================================================

def tokenize_word_level(text):
    """Word-level tokenization using NLTK."""
    return word_tokenize(text.lower())


def build_vocab(texts, min_freq=2):
    """Build vocabulary from tokenized texts."""
    from collections import Counter
    counter = Counter()
    for text in texts:
        tokens = tokenize_word_level(text)
        counter.update(tokens)
    
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, count in counter.most_common():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def texts_to_indices(texts, vocab, max_len):
    """Convert texts to padded index sequences."""
    import numpy as np
    result = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tokenize_word_level(text)
        for j, token in enumerate(tokens[:max_len]):
            result[i, j] = vocab.get(token, vocab["<unk>"])
    return result
