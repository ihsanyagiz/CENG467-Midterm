"""
Task 5: Transformer-based Language Model (GPT-2)
Uses pre-trained GPT-2 for perplexity evaluation and text generation.
"""

import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


MODEL_NAME = "gpt2"


def load_gpt2():
    """Load pre-trained GPT-2 model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    print(f"GPT-2 loaded on {DEVICE}")
    return model, tokenizer


def compute_gpt2_perplexity(model, tokenizer, dataset_split, max_len=512):
    """Compute perplexity of GPT-2 on WikiText-2 test set."""
    model.eval()

    # Concatenate all text
    text = "\n".join([ex["text"] for ex in dataset_split if ex["text"].strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]

    total_loss = 0.0
    num_tokens = 0
    stride = max_len

    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids) - 1, stride), desc="GPT-2 PPL"):
            end = min(i + max_len, len(input_ids))
            input_chunk = input_ids[i:end].unsqueeze(0).to(DEVICE)

            if input_chunk.size(1) < 2:
                continue

            outputs = model(input_chunk, labels=input_chunk)
            loss = outputs.loss
            total_loss += loss.item() * (input_chunk.size(1) - 1)
            num_tokens += input_chunk.size(1) - 1

    avg_loss = total_loss / max(num_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return perplexity


def generate_gpt2_text(model, tokenizer, prompt="The", max_len=100, temperature=0.8,
                       top_k=50, top_p=0.95):
    """Generate text using GPT-2."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text
