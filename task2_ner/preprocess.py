"""
Task 2: Named Entity Recognition — Preprocessing
CoNLL-2003 dataset loader with BIO tagging and token–label alignment.
"""

from datasets import load_dataset


# CoNLL-2003 NER tag mapping
NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
TAG2IDX = {tag: idx for idx, tag in enumerate(NER_TAGS)}
IDX2TAG = {idx: tag for tag, idx in TAG2IDX.items()}


def load_conll2003():
    """Load CoNLL-2003 dataset with standard splits."""
    dataset = load_dataset("conll2003")
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    return dataset


def build_word_vocab(dataset, min_freq=1):
    """Build word vocabulary from CoNLL-2003 training set."""
    from collections import Counter
    counter = Counter()
    for example in dataset["train"]:
        for token in example["tokens"]:
            counter[token.lower()] += 1

    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, count in counter.most_common():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    print(f"Word vocabulary size: {len(vocab)}")
    return vocab


def align_labels_for_bert(tokenizer, tokens, ner_tags, max_len=128):
    """Align BIO labels with BERT subword tokens.
    
    Strategy: first subword token gets the original label,
    subsequent subword tokens get -100 (ignored in loss).
    """
    encoding = tokenizer(
        tokens, is_split_into_words=True, truncation=True,
        padding="max_length", max_length=max_len
    )
    word_ids = encoding.word_ids()
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)  # special tokens
        elif word_idx != previous_word_idx:
            aligned_labels.append(ner_tags[word_idx])
        else:
            aligned_labels.append(-100)  # subword continuation
        previous_word_idx = word_idx

    encoding["labels"] = aligned_labels
    return encoding


def prepare_bilstm_data(dataset_split, vocab, tag2idx, max_len=128):
    """Prepare data for BiLSTM-CRF: convert tokens to indices, pad sequences."""
    import numpy as np
    all_token_ids = []
    all_tag_ids = []
    all_lengths = []

    for example in dataset_split:
        tokens = example["tokens"]
        tags = example["ner_tags"]

        token_ids = [vocab.get(t.lower(), vocab["<unk>"]) for t in tokens[:max_len]]
        tag_ids = [tags[i] for i in range(min(len(tags), max_len))]
        length = len(token_ids)

        # Pad
        token_ids += [0] * (max_len - length)
        tag_ids += [0] * (max_len - length)

        all_token_ids.append(token_ids)
        all_tag_ids.append(tag_ids)
        all_lengths.append(length)

    return (np.array(all_token_ids, dtype=np.int64),
            np.array(all_tag_ids, dtype=np.int64),
            np.array(all_lengths, dtype=np.int64))
