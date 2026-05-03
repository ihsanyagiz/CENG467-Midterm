"""
Task 3: Extractive Summarization — LexRank
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TASK3_LEXRANK_SENTENCES


def lexrank_summarize(text, num_sentences=TASK3_LEXRANK_SENTENCES, threshold=0.1):
    """LexRank extractive summarization.
    
    Uses TF-IDF cosine similarity to build a graph of sentences,
    then applies a power method to compute LexRank scores.
    """
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Build TF-IDF matrix
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # If all sentences are stop words or empty
        return " ".join(sentences[:num_sentences])

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Apply threshold
    sim_matrix[sim_matrix < threshold] = 0

    # Normalize rows to create transition matrix
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    transition_matrix = sim_matrix / row_sums

    # Power method to compute stationary distribution (LexRank scores)
    n = len(sentences)
    scores = np.ones(n) / n
    damping = 0.85

    for _ in range(100):
        new_scores = (1 - damping) / n + damping * transition_matrix.T.dot(scores)
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    # Select top sentences (preserving original order)
    ranked_indices = np.argsort(scores)[::-1][:num_sentences]
    ranked_indices = sorted(ranked_indices)  # Preserve document order

    summary = " ".join([sentences[i] for i in ranked_indices])
    return summary


def generate_lexrank_summaries(articles, num_sentences=TASK3_LEXRANK_SENTENCES):
    """Generate LexRank summaries for a list of articles."""
    from tqdm import tqdm
    summaries = []
    for article in tqdm(articles, desc="LexRank"):
        summary = lexrank_summarize(article, num_sentences)
        summaries.append(summary)
    return summaries
