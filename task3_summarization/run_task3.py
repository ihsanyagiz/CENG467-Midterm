"""
Task 3: Text Summarization — Main Runner
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import set_seed
from utils import print_section, save_results, plot_comparison_bar

from preprocess import load_cnn_dailymail_subset, get_articles_and_highlights
from extractive import generate_lexrank_summaries
from abstractive import generate_bart_summaries
from evaluate import evaluate_summaries


def qualitative_examples(articles, references, lexrank_sums, bart_sums, n=3):
    """Print qualitative comparison examples."""
    print_section("Qualitative Examples")
    for i in range(min(n, len(articles))):
        print(f"\n{'='*60}")
        print(f"Example {i+1}")
        print(f"{'='*60}")
        print(f"\nArticle (first 300 chars): {articles[i][:300]}...")
        print(f"\nReference Summary: {references[i][:200]}...")
        print(f"\nLexRank Summary: {lexrank_sums[i][:200]}...")
        print(f"\nBART Summary: {bart_sums[i][:200]}...")


def run_task3():
    set_seed()
    print_section("TASK 3: TEXT SUMMARIZATION (CNN/DailyMail)")

    # ---- Load Data ----
    train, val, test = load_cnn_dailymail_subset()
    test_articles, test_references = get_articles_and_highlights(test)

    # ============================================================
    # Extractive: LexRank
    # ============================================================
    print_section("LexRank Extractive Summarization")
    lexrank_summaries = generate_lexrank_summaries(test_articles)
    lexrank_result = evaluate_summaries(lexrank_summaries, test_references, "LexRank")

    # ============================================================
    # Abstractive: BART
    # ============================================================
    print_section("BART Abstractive Summarization")
    bart_summaries = generate_bart_summaries(test_articles)
    bart_result = evaluate_summaries(bart_summaries, test_references, "BART")

    # ============================================================
    # Qualitative Comparison
    # ============================================================
    qualitative_examples(test_articles, test_references, lexrank_summaries, bart_summaries)

    # ============================================================
    # Summary
    # ============================================================
    print_section("Task 3 — Summary")

    model_names = ["LexRank", "BART"]
    rouge1 = [lexrank_result["rouge1"], bart_result["rouge1"]]
    rouge2 = [lexrank_result["rouge2"], bart_result["rouge2"]]
    rougeL = [lexrank_result["rougeL"], bart_result["rougeL"]]

    plot_comparison_bar(model_names, rouge1, "ROUGE-1",
                        "Task 3: ROUGE-1 Comparison", "task3_rouge1_comparison.png")
    plot_comparison_bar(model_names, rouge2, "ROUGE-2",
                        "Task 3: ROUGE-2 Comparison", "task3_rouge2_comparison.png")

    all_results = {
        "lexrank": lexrank_result,
        "bart": bart_result,
    }
    save_results(all_results, "task3_results.json")
    print("\nTask 3 complete!")


if __name__ == "__main__":
    run_task3()
