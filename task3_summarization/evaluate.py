"""
Task 3: Summarization Evaluation
ROUGE, BLEU, METEOR, BERTScore
"""

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from nltk.tokenize import word_tokenize
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_rouge(predictions, references):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)

    return {key: np.mean(vals) for key, vals in scores.items()}


def compute_bleu(predictions, references):
    """Compute average sentence-level BLEU score."""
    smoothie = SmoothingFunction().method1
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        if len(pred_tokens) == 0:
            bleu_scores.append(0.0)
            continue
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)

    return np.mean(bleu_scores)


def compute_meteor(predictions, references):
    """Compute average METEOR score."""
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        score = nltk_meteor([ref_tokens], pred_tokens)
        scores.append(score)
    return np.mean(scores)


def compute_bertscore(predictions, references):
    """Compute BERTScore."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def evaluate_summaries(predictions, references, model_name):
    """Run all summarization metrics."""
    print(f"\n--- {model_name} Evaluation ---")

    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    meteor = compute_meteor(predictions, references)
    bertscore = compute_bertscore(predictions, references)

    print(f"  ROUGE-1: {rouge['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge['rougeL']:.4f}")
    print(f"  BLEU:    {bleu:.4f}")
    print(f"  METEOR:  {meteor:.4f}")
    print(f"  BERTScore F1: {bertscore['f1']:.4f}")

    return {
        "model": model_name,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bleu": bleu,
        "meteor": meteor,
        "bertscore_f1": bertscore["f1"],
    }
