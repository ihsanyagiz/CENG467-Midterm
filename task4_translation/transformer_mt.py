"""
Task 4: Transformer-based Machine Translation
Uses Helsinki-NLP/opus-mt-en-de (pre-trained).
"""

from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"


def load_transformer_mt():
    """Load pre-trained Helsinki-NLP EN→DE model."""
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    print(f"Helsinki-NLP model loaded on {DEVICE}")
    return model, tokenizer


def translate_transformer(model, tokenizer, text, max_len=128):
    """Translate a single sentence with Helsinki-NLP model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=max_len,
                       truncation=True).to(DEVICE)
    with torch.no_grad():
        translated = model.generate(**inputs, max_length=max_len, num_beams=4)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def translate_batch(sentences, batch_size=32):
    """Translate a batch of sentences."""
    model, tokenizer = load_transformer_mt()
    translations = []

    for sent in tqdm(sentences, desc="Transformer Translation"):
        translation = translate_transformer(model, tokenizer, sent)
        translations.append(translation)

    return translations
