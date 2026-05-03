"""
Task 3: Abstractive Summarization — BART
Uses facebook/bart-large-cnn (pre-trained on CNN/DailyMail).
"""

from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, TASK3_MAX_SOURCE_LEN, TASK3_MAX_TARGET_LEN

MODEL_NAME = "facebook/bart-large-cnn"


def load_bart_model():
    """Load pre-trained BART model for summarization."""
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    print(f"BART model loaded on {DEVICE}")
    return model, tokenizer


def generate_bart_summary(model, tokenizer, text, max_source=TASK3_MAX_SOURCE_LEN,
                          max_target=TASK3_MAX_TARGET_LEN):
    """Generate a single summary using BART."""
    inputs = tokenizer(text, return_tensors="pt", max_length=max_source,
                       truncation=True).to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_target,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def generate_bart_summaries(articles, batch_size=4):
    """Generate BART summaries for a list of articles."""
    model, tokenizer = load_bart_model()
    summaries = []

    for article in tqdm(articles, desc="BART Summarization"):
        summary = generate_bart_summary(model, tokenizer, article)
        summaries.append(summary)

    return summaries
