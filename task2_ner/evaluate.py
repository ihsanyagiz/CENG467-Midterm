"""
Task 2: NER Evaluation
"""

from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


def evaluate_ner(true_tags, pred_tags, model_name):
    """Evaluate NER model using seqeval entity-level metrics."""
    precision = precision_score(true_tags, pred_tags)
    recall = recall_score(true_tags, pred_tags)
    f1 = f1_score(true_tags, pred_tags)

    print(f"\n--- {model_name} NER Evaluation ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"\nDetailed Report:")
    report = classification_report(true_tags, pred_tags)
    print(report)

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
    }


def analyze_errors(true_tags, pred_tags, dataset_split, n=10):
    """Analyze common NER errors: boundary errors, entity confusions."""
    errors = {"boundary": [], "type_confusion": [], "missed": [], "spurious": []}

    for i in range(min(len(true_tags), len(dataset_split))):
        tokens = dataset_split[i]["tokens"]
        true_seq = true_tags[i]
        pred_seq = pred_tags[i]

        for j in range(min(len(true_seq), len(tokens))):
            t = true_seq[j]
            p = pred_seq[j]
            if t != p:
                token = tokens[j] if j < len(tokens) else "<UNK>"
                error_info = {
                    "sentence_idx": i,
                    "token": token,
                    "position": j,
                    "true_tag": t,
                    "pred_tag": p,
                }
                # Categorize error
                if t == "O" and p != "O":
                    errors["spurious"].append(error_info)
                elif t != "O" and p == "O":
                    errors["missed"].append(error_info)
                elif t.split("-")[-1] != p.split("-")[-1] and t != "O":
                    errors["type_confusion"].append(error_info)
                elif t.split("-")[0] != p.split("-")[0]:
                    errors["boundary"].append(error_info)

    # Print summary
    print(f"\nError Analysis Summary:")
    print(f"  Boundary errors:    {len(errors['boundary'])}")
    print(f"  Type confusions:    {len(errors['type_confusion'])}")
    print(f"  Missed entities:    {len(errors['missed'])}")
    print(f"  Spurious entities:  {len(errors['spurious'])}")

    # Show examples
    for err_type, err_list in errors.items():
        if err_list:
            print(f"\n  Sample {err_type} errors:")
            for e in err_list[:3]:
                print(f"    Token='{e['token']}', True={e['true_tag']}, Pred={e['pred_tag']}")

    return errors
