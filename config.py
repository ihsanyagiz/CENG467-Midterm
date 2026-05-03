"""
CENG 467 - NLU&G Take-Home Midterm
Global Configuration
Student ID: 300201071
"""

import os
import random
import numpy as np
import torch

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "report", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Task 1: Text Classification
# ============================================================
TASK1_MAX_LEN = 128
TASK1_BATCH_SIZE = 32
TASK1_TRANSFORMER_BATCH_SIZE = 16
TASK1_LSTM_EPOCHS = 10
TASK1_TRANSFORMER_EPOCHS = 3
TASK1_LSTM_LR = 1e-3
TASK1_TRANSFORMER_LR = 2e-5
TASK1_LSTM_HIDDEN = 256
TASK1_LSTM_EMBED = 100
TASK1_LSTM_DROPOUT = 0.5

# ============================================================
# Task 2: NER
# ============================================================
TASK2_MAX_LEN = 128
TASK2_BATCH_SIZE = 32
TASK2_TRANSFORMER_BATCH_SIZE = 16
TASK2_LSTM_EPOCHS = 15
TASK2_TRANSFORMER_EPOCHS = 3
TASK2_LSTM_LR = 1e-3
TASK2_TRANSFORMER_LR = 3e-5
TASK2_LSTM_HIDDEN = 256
TASK2_LSTM_EMBED = 100
TASK2_LSTM_DROPOUT = 0.5

# ============================================================
# Task 3: Summarization
# ============================================================
TASK3_SUBSET_TRAIN = 5000
TASK3_SUBSET_VAL = 500
TASK3_SUBSET_TEST = 500
TASK3_MAX_SOURCE_LEN = 512
TASK3_MAX_TARGET_LEN = 150
TASK3_BATCH_SIZE = 8
TASK3_LEXRANK_SENTENCES = 3

# ============================================================
# Task 4: Machine Translation
# ============================================================
TASK4_MAX_LEN = 40
TASK4_BATCH_SIZE = 64
TASK4_SEQ2SEQ_EPOCHS = 8
TASK4_SEQ2SEQ_LR = 1e-3
TASK4_SEQ2SEQ_HIDDEN = 512
TASK4_SEQ2SEQ_EMBED = 256
TASK4_SEQ2SEQ_DROPOUT = 0.5
TASK4_TEACHER_FORCING = 0.5

# ============================================================
# Task 5: Language Modeling
# ============================================================
TASK5_BATCH_SIZE = 128
TASK5_BPTT = 64
TASK5_LSTM_EPOCHS = 5
TASK5_LSTM_LR = 1e-3
TASK5_LSTM_HIDDEN = 512
TASK5_LSTM_EMBED = 256
TASK5_LSTM_LAYERS = 2
TASK5_LSTM_DROPOUT = 0.5
TASK5_VOCAB_SIZE = 30000
