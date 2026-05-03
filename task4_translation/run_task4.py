"""
Task 4: Machine Translation — Main Runner
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (set_seed, DEVICE, TASK4_MAX_LEN, TASK4_BATCH_SIZE,
                    TASK4_SEQ2SEQ_EPOCHS, TASK4_SEQ2SEQ_LR, TASK4_SEQ2SEQ_HIDDEN,
                    TASK4_SEQ2SEQ_EMBED, TASK4_SEQ2SEQ_DROPOUT)
from utils import print_section, save_results, plot_comparison_bar, plot_training_curves

from preprocess import load_multi30k, build_translation_vocab, prepare_translation_data
from seq2seq_attention import (Encoder, Decoder, Seq2Seq, TranslationDataset,
                                train_seq2seq, translate_seq2seq)
from transformer_mt import translate_batch
from evaluate import evaluate_translation

from torch.utils.data import DataLoader


def run_task4():
    set_seed()
    print_section("TASK 4: MACHINE TRANSLATION (Multi30k EN→DE)")

    # ---- Load Data ----
    dataset = load_multi30k()

    src_vocab = build_translation_vocab(dataset["train"], "en")
    tgt_vocab = build_translation_vocab(dataset["train"], "de")

    # Prepare data for Seq2Seq
    src_train, tgt_train, sl_train, tl_train = prepare_translation_data(dataset["train"], src_vocab, tgt_vocab)
    src_val, tgt_val, sl_val, tl_val = prepare_translation_data(dataset["validation"], src_vocab, tgt_vocab)
    src_test, tgt_test, sl_test, tl_test = prepare_translation_data(dataset["test"], src_vocab, tgt_vocab)

    train_dataset = TranslationDataset(src_train, tgt_train, sl_train, tl_train)
    val_dataset = TranslationDataset(src_val, tgt_val, sl_val, tl_val)
    test_dataset = TranslationDataset(src_test, tgt_test, sl_test, tl_test)

    train_loader = DataLoader(train_dataset, batch_size=TASK4_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TASK4_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=TASK4_BATCH_SIZE)

    # ============================================================
    # Seq2Seq + Attention
    # ============================================================
    print_section("Seq2Seq + Bahdanau Attention")

    encoder = Encoder(len(src_vocab), TASK4_SEQ2SEQ_EMBED, TASK4_SEQ2SEQ_HIDDEN, TASK4_SEQ2SEQ_DROPOUT)
    decoder = Decoder(len(tgt_vocab), TASK4_SEQ2SEQ_EMBED, TASK4_SEQ2SEQ_HIDDEN, TASK4_SEQ2SEQ_DROPOUT)
    seq2seq = Seq2Seq(encoder, decoder, DEVICE)
    print(f"Seq2Seq parameters: {sum(p.numel() for p in seq2seq.parameters()):,}")

    train_losses, val_losses = train_seq2seq(
        seq2seq, train_loader, val_loader, TASK4_SEQ2SEQ_EPOCHS, TASK4_SEQ2SEQ_LR,
        tgt_pad_idx=tgt_vocab["<pad>"]
    )
    plot_training_curves(train_losses, val_losses, "Seq2Seq Training Curves", "task4_seq2seq_loss.png")

    # Translate test set with Seq2Seq
    test_en_sentences = dataset["test"]["en"]
    test_de_references = dataset["test"]["de"]

    seq2seq_translations = []
    for sent in test_en_sentences:
        translation = translate_seq2seq(seq2seq, sent, src_vocab, tgt_vocab)
        seq2seq_translations.append(translation)

    seq2seq_result = evaluate_translation(seq2seq_translations, test_de_references, "Seq2Seq+Attention")

    # ============================================================
    # Transformer (Helsinki-NLP)
    # ============================================================
    print_section("Transformer (Helsinki-NLP)")

    transformer_translations = translate_batch(test_en_sentences)
    transformer_result = evaluate_translation(transformer_translations, test_de_references, "Helsinki-NLP Transformer")

    # ============================================================
    # Qualitative Examples
    # ============================================================
    print_section("Qualitative Examples")
    for i in range(min(3, len(test_en_sentences))):
        print(f"\nExample {i+1}:")
        print(f"  Source (EN):     {test_en_sentences[i]}")
        print(f"  Reference (DE):  {test_de_references[i]}")
        print(f"  Seq2Seq:         {seq2seq_translations[i]}")
        print(f"  Transformer:     {transformer_translations[i]}")

    # ============================================================
    # Summary
    # ============================================================
    print_section("Task 4 — Summary")

    model_names = ["Seq2Seq+Attn", "Helsinki-NLP"]
    bleu_scores = [seq2seq_result["bleu"], transformer_result["bleu"]]

    plot_comparison_bar(model_names, bleu_scores, "BLEU",
                        "Task 4: Translation BLEU Comparison", "task4_bleu_comparison.png")

    all_results = {
        "seq2seq": seq2seq_result,
        "transformer": transformer_result,
    }
    save_results(all_results, "task4_results.json")
    print("\nTask 4 complete!")


if __name__ == "__main__":
    run_task4()
