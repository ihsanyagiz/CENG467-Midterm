"""
Task 5: Language Modeling — Preprocessing
WikiText-2 dataset loader and tokenization.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TASK5_VOCAB_SIZE, TASK5_BPTT

from datasets import load_dataset
from collections import Counter
import numpy as np
import torch


def load_wikitext2():
    """Load WikiText-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print(f"WikiText-2 — Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    return dataset


def build_lm_vocab(dataset_split, max_vocab=TASK5_VOCAB_SIZE):
    """Build vocabulary from training text."""
    counter = Counter()
    for example in dataset_split:
        text = example["text"].strip()
        if text:
            tokens = text.lower().split()
            counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    idx = 3
    for word, _ in counter.most_common(max_vocab - 3):
        vocab[word] = idx
        idx += 1

    print(f"LM vocabulary size: {len(vocab)}")
    return vocab


def tokenize_lm_data(dataset_split, vocab):
    """Tokenize all text into a single long sequence of indices."""
    all_ids = []
    for example in dataset_split:
        text = example["text"].strip()
        if text:
            tokens = text.lower().split()
            ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
            ids.append(vocab["<eos>"])
            all_ids.extend(ids)
    return np.array(all_ids, dtype=np.int64)


def batchify(data, batch_size):
    """Reshape data into [batch_size, seq_len] for BPTT."""
    nbatch = len(data) // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape(batch_size, -1)
    return torch.LongTensor(data)


def get_batch(source, i, bptt=TASK5_BPTT):
    """Get a batch for language modeling (input, target)."""
    seq_len = min(bptt, source.size(1) - 1 - i)
    data = source[:, i:i + seq_len]
    target = source[:, i + 1:i + 1 + seq_len]
    return data, target
