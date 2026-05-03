"""
Task 4: Machine Translation — Preprocessing
Multi30k dataset loader and tokenization.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED, TASK4_MAX_LEN

from datasets import load_dataset
from collections import Counter
import numpy as np


def load_multi30k():
    """Load Multi30k EN→DE dataset."""
    dataset = load_dataset("bentrevett/multi30k")
    print(f"Multi30k — Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    return dataset


def simple_tokenize(text, lang="en"):
    """Simple whitespace + lowercasing tokenizer as fallback."""
    return text.lower().strip().split()


def build_translation_vocab(dataset_split, lang_key, min_freq=2):
    """Build vocabulary for source or target language."""
    counter = Counter()
    for example in dataset_split:
        tokens = simple_tokenize(example[lang_key])
        counter.update(tokens)

    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    idx = 4
    for word, count in counter.most_common():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    print(f"  {lang_key} vocab size: {len(vocab)}")
    return vocab


def encode_sentence(sentence, vocab, max_len, add_special=True):
    """Encode a sentence as indices with <sos> and <eos>."""
    tokens = simple_tokenize(sentence)
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]

    if add_special:
        indices = [vocab["<sos>"]] + indices + [vocab["<eos>"]]

    # Truncate
    indices = indices[:max_len]
    length = len(indices)

    # Pad
    indices += [vocab["<pad>"]] * (max_len - length)

    return indices, length


def prepare_translation_data(dataset_split, src_vocab, tgt_vocab, max_len=TASK4_MAX_LEN):
    """Prepare parallel data for Seq2Seq model."""
    src_data, tgt_data = [], []
    src_lens, tgt_lens = [], []

    for example in dataset_split:
        src_ids, src_len = encode_sentence(example["en"], src_vocab, max_len)
        tgt_ids, tgt_len = encode_sentence(example["de"], tgt_vocab, max_len)
        src_data.append(src_ids)
        tgt_data.append(tgt_ids)
        src_lens.append(src_len)
        tgt_lens.append(tgt_len)

    return (np.array(src_data, dtype=np.int64),
            np.array(tgt_data, dtype=np.int64),
            np.array(src_lens, dtype=np.int64),
            np.array(tgt_lens, dtype=np.int64))
