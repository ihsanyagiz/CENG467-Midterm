"""
Task 5: Language Modeling — Main Runner
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn

from config import (set_seed, DEVICE, TASK5_BATCH_SIZE, TASK5_BPTT,
                    TASK5_LSTM_EPOCHS, TASK5_LSTM_LR, TASK5_LSTM_HIDDEN,
                    TASK5_LSTM_EMBED, TASK5_LSTM_LAYERS, TASK5_LSTM_DROPOUT)
from utils import print_section, save_results, plot_comparison_bar, plot_training_curves

from preprocess import load_wikitext2, build_lm_vocab, tokenize_lm_data, batchify
from lstm_lm import LSTMLanguageModel, train_lstm_lm, evaluate_lm, generate_text
from transformer_lm import load_gpt2, compute_gpt2_perplexity, generate_gpt2_text
from evaluate import compute_perplexity


def run_task5():
    set_seed()
    print_section("TASK 5: LANGUAGE MODELING (WikiText-2)")

    # ---- Load Data ----
    dataset = load_wikitext2()
    vocab = build_lm_vocab(dataset["train"])

    train_ids = tokenize_lm_data(dataset["train"], vocab)
    val_ids = tokenize_lm_data(dataset["validation"], vocab)
    test_ids = tokenize_lm_data(dataset["test"], vocab)

    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}, Test tokens: {len(test_ids):,}")

    train_data = batchify(train_ids, TASK5_BATCH_SIZE)
    val_data = batchify(val_ids, TASK5_BATCH_SIZE)
    test_data = batchify(test_ids, TASK5_BATCH_SIZE)

    # ============================================================
    # LSTM Language Model
    # ============================================================
    print_section("LSTM Language Model")

    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embed_dim=TASK5_LSTM_EMBED,
        hidden_dim=TASK5_LSTM_HIDDEN,
        num_layers=TASK5_LSTM_LAYERS,
        dropout=TASK5_LSTM_DROPOUT,
    )
    print(f"LSTM LM parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_lstm_lm(
        model, train_data, val_data, TASK5_LSTM_EPOCHS, TASK5_LSTM_LR
    )
    plot_training_curves(train_losses, val_losses, "LSTM LM Training Curves", "task5_lstm_loss.png")

    # Test perplexity
    criterion = nn.CrossEntropyLoss()
    test_data_gpu = test_data.to(DEVICE)
    test_loss = evaluate_lm(model, test_data_gpu, criterion)
    lstm_ppl = compute_perplexity(test_loss)
    print(f"\nLSTM Test Perplexity: {lstm_ppl:.2f}")

    # Generate samples
    print("\nLSTM Generated Samples:")
    seed_texts = ["the", "in the beginning", "scientists have discovered",
                  "the united states", "natural language processing"]
    lstm_samples = []
    for seed in seed_texts:
        sample = generate_text(model, vocab, seed_text=seed, max_len=50)
        lstm_samples.append(sample)
        print(f"  Seed: '{seed}' → {sample[:150]}...")

    # ============================================================
    # GPT-2 Language Model
    # ============================================================
    print_section("GPT-2 Language Model")

    gpt2_model, gpt2_tokenizer = load_gpt2()
    gpt2_ppl = compute_gpt2_perplexity(gpt2_model, gpt2_tokenizer, dataset["test"])
    print(f"\nGPT-2 Test Perplexity: {gpt2_ppl:.2f}")

    # Generate samples
    print("\nGPT-2 Generated Samples:")
    prompts = ["The", "In the beginning", "Scientists have discovered",
               "The United States", "Natural language processing"]
    gpt2_samples = []
    for prompt in prompts:
        sample = generate_gpt2_text(gpt2_model, gpt2_tokenizer, prompt=prompt, max_len=80)
        gpt2_samples.append(sample)
        print(f"  Prompt: '{prompt}' → {sample[:150]}...")

    # ============================================================
    # Summary
    # ============================================================
    print_section("Task 5 — Summary")

    model_names = ["LSTM", "GPT-2"]
    perplexities = [lstm_ppl, gpt2_ppl]

    for name, ppl in zip(model_names, perplexities):
        print(f"  {name:10s}  Perplexity={ppl:.2f}")

    plot_comparison_bar(model_names, perplexities, "Perplexity (↓ better)",
                        "Task 5: Perplexity Comparison", "task5_perplexity_comparison.png")

    all_results = {
        "lstm": {"perplexity": lstm_ppl, "samples": lstm_samples},
        "gpt2": {"perplexity": gpt2_ppl, "samples": gpt2_samples},
    }
    save_results(all_results, "task5_results.json")
    print("\nTask 5 complete!")


if __name__ == "__main__":
    run_task5()
