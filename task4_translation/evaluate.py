"""
Task 4: Machine Translation Evaluation
BLEU, METEOR, ChrF, BERTScore
"""

import sacrebleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_bleu_sacre(predictions, references):
    """Compute corpus-level BLEU using sacrebleu."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_chrf(predictions, references):
    """Compute ChrF score."""
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return chrf.score


def compute_meteor_mt(predictions, references):
    """Compute average METEOR score."""
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = word_tokenize(ref.lower())
        pred_tokens = word_tokenize(pred.lower())
        score = nltk_meteor([ref_tokens], pred_tokens)
        scores.append(score)
    return np.mean(scores)


def compute_bertscore_mt(predictions, references, lang="de"):
    """Compute BERTScore for translation."""
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def evaluate_translation(predictions, references, model_name):
    """Run all translation metrics."""
    print(f"\n--- {model_name} Translation Evaluation ---")

    bleu = compute_bleu_sacre(predictions, references)
    chrf = compute_chrf(predictions, references)
    meteor = compute_meteor_mt(predictions, references)
    bertscore = compute_bertscore_mt(predictions, references)

    print(f"  BLEU:        {bleu:.2f}")
    print(f"  ChrF:        {chrf:.2f}")
    print(f"  METEOR:      {meteor:.4f}")
    print(f"  BERTScore F1: {bertscore['f1']:.4f}")

    return {
        "model": model_name,
        "bleu": bleu,
        "chrf": chrf,
        "meteor": meteor,
        "bertscore_f1": bertscore["f1"],
    }
