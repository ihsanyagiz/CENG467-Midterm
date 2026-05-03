# CENG 467 - Natural Language Understanding and Generation
## Take-Home Midterm Examination

**Student ID:** 300201071  
**Course:** CENG 467 — Natural Language Understanding and Generation  
**Instructor:** Prof. Dr. Aytuğ Onan

---

## Project Overview

This repository contains the complete implementation and report for the CENG 467 Take-Home Midterm. The project covers five core NLP tasks:

1. **Text Classification** — SST-2 dataset with TF-IDF+SVC, BiLSTM, and DistilBERT
2. **Named Entity Recognition** — CoNLL-2003 with BiLSTM-CRF and BERT
3. **Text Summarization** — CNN/DailyMail with LexRank and BART
4. **Machine Translation** — Multi30k (EN→DE) with Seq2Seq+Attention and Helsinki-NLP Transformer
5. **Language Modeling** — WikiText-2 with LSTM and GPT-2

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

## Running Experiments

Each task can be run independently:

```bash
python task1_text_classification/run_task1.py
python task2_ner/run_task2.py
python task3_summarization/run_task3.py
python task4_translation/run_task4.py
python task5_language_modeling/run_task5.py
```

## Reproducibility

- All random seeds are fixed to `42`
- Environment: see `requirements.txt`
- Device: CUDA if available, CPU otherwise
- All hyperparameters are centralized in `config.py`

## Report

The LaTeX report is located in `report/main.tex`. Compile with:
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## License

This project is for academic purposes only (CENG 467 coursework).
